// Wave-7 Step 3 — mocked bakeoff:blend tests (AC-703/AC-704/AC-705) + the DB-backed
// golden A0-equivalence check.
//
// The mocked block injects ALL dependencies ({pool, fuse, rerankFn, embedQuery}) so the
// capture-once invariant, the arm math, and the verdict statistics are proven with ZERO
// network and zero live quota. The DB block (skipIf no DATABASE_URL) proves the offline
// scoring path (fuse capture → blendScores under A0) reproduces the LIVE searchMemory
// ordering byte-for-byte on seeded rows — the equivalence that makes offline arm scores
// trustworthy.

import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skipDb = !DATABASE_URL;

// Deterministic helpers, hoisted for the vi.mock factory (search-golden pattern).
const H = vi.hoisted(() => {
  function fnv1a(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }
  function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function fakeVec(key: string, dim = 1024): number[] {
    const rnd = mulberry32(fnv1a(key));
    return Array.from({ length: dim }, () =>
      Number((rnd() * 2 - 1).toFixed(3)),
    );
  }
  function fakeRerank(
    query: string,
    docs: string[],
    topK: number,
  ): Array<{ index: number; score: number }> {
    const scored = docs.map((d, i) => ({
      index: i,
      score: Number(
        (0.05 + 0.9 * mulberry32(fnv1a(`${query}|${d.length}|${i}`))()).toFixed(
          6,
        ),
      ),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.min(topK, docs.length));
  }
  return { fnv1a, mulberry32, fakeVec, fakeRerank };
});

// HARD RULE: never call live Voyage — module-mock the boundary (used by the DB block;
// the mocked block injects its own deps and never reaches this module).
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map((t) => H.fakeVec(`legacy:${t}`)),
  embedContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(t))),
  embedContextualSingle: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(t)),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  AFFECTED_EPS,
  BAKEOFF_BLEND_COST_NOTE,
  MIN_SPLIT_ROWS,
  SEED,
  parseBakeoffBlendArgs,
  predeclaredArms,
  runBakeoffBlend,
  writeBakeoffBlendArtifact,
  type BakeoffBlendDeps,
  type Queryable,
} from "../src/db/bakeoff-blend.js";
import {
  blendScores,
  formatHits,
  fuseCandidates,
  searchMemory,
  RERANK_DOC_TRUNCATION,
  type FusedCandidate,
  type MemoryType,
} from "../src/memory.js";
import { config } from "../src/config.js";
import {
  pairedBootstrapCI,
  signTest,
  type RecallEvalFileV2,
  type RecallEvalRowV2,
} from "../src/eval-core.js";
import { rerank } from "../src/voyage.js";

const here = dirname(fileURLToPath(import.meta.url));

// ── Mocked world ─────────────────────────────────────────────────────────────────────

const NOW = Date.UTC(2026, 6, 1, 12, 0, 0); // 2026-07-01T12:00:00Z
const DAY = 86_400_000;

function mkCand(over: {
  id: string;
  title: string;
  type?: MemoryType;
  createdDaysAgo: number;
  importance?: number;
  sourceKind?: string | null;
}): FusedCandidate {
  return {
    id: over.id,
    title: over.title,
    content: `content of ${over.id}`,
    summary: null,
    type: over.type ?? "semantic",
    importance: over.importance ?? 0.5,
    created_at: new Date(NOW - over.createdDaysAgo * DAY),
    event_date: null,
    source_kind: over.sourceKind ?? null,
    status: "active",
    rrf: 0.01,
    bm25_rank: 1,
    vec_rank: 1,
  };
}

// 16 rows (≥ MIN_SPLIT_ROWS): m-001 is the discriminating row; m-002..m-016 are
// trivial single-candidate rows where every arm scores nDCG 1. Rerank relevance is
// encoded in the candidate TITLE ("t|<score>") so the rerank mock is transparent.
//
// m-001: gold g-001 (type entity, 90d old, rel 0.60) vs distractor d-001 (fresh,
// rel 0.58). Hand-computed finals:
//   A0: gold 0.7·0.6 + 0.2·e⁻³ + 0.05 ≈ 0.47996 < distractor 0.656  → gold rank 2
//   A1/A2 (entity exempt): gold 0.42 + 0.2 + 0.05 = 0.67 > 0.656     → gold rank 1
//   A3 (multiplicative): 0.6·(1+0.2e⁻³+0.025) ≈ 0.62097 < 0.7105     → gold rank 2
//   A4 (relevance-only): 0.6 > 0.58                                   → gold rank 1
const GOLD_TYPE: Record<string, string> = { "g-001": "entity" };

function evalRows(n = 16): RecallEvalRowV2[] {
  return Array.from({ length: n }, (_, i) => {
    const id = `m-${String(i + 1).padStart(3, "0")}`;
    const gold = `g-${String(i + 1).padStart(3, "0")}`;
    return {
      id,
      query: `query ${id}`,
      relevantIds: [gold],
      split: "dev" as const,
      provenance: "seed-v1" as const,
      // Two temporal-facet rows exercise the temporal slice.
      ...(i === 1 || i === 2 ? { facet: "temporal" } : {}),
    };
  });
}

function evalFile(rows = evalRows()): RecallEvalFileV2 {
  return { version: 2, k: 10, changelog: ["mock (never committed)"], rows };
}

function candidatesFor(rowIndex: number): FusedCandidate[] {
  const gold = `g-${String(rowIndex + 1).padStart(3, "0")}`;
  if (rowIndex === 0) {
    return [
      mkCand({
        id: gold,
        title: "t|0.60",
        type: "entity",
        createdDaysAgo: 90,
      }),
      mkCand({ id: "d-001", title: "t|0.58", createdDaysAgo: 0 }),
    ];
  }
  return [mkCand({ id: gold, title: "t|0.90", createdDaysAgo: 10 })];
}

interface MockWorld {
  deps: BakeoffBlendDeps;
  counters: {
    fuse: number;
    rerank: number;
    embed: number;
    fusePerQuery: Map<string, number>;
    rerankPerQuery: Map<string, number>;
    embedPerQuery: Map<string, number>;
    rrfKsSeen: Set<number>;
  };
}

/** Fully-injected mock deps: identity gold chains, title-encoded rerank scores,
 *  call counters everywhere. Zero network by construction. */
function mockWorld(rows = evalRows()): MockWorld {
  const counters = {
    fuse: 0,
    rerank: 0,
    embed: 0,
    fusePerQuery: new Map<string, number>(),
    rerankPerQuery: new Map<string, number>(),
    embedPerQuery: new Map<string, number>(),
    rrfKsSeen: new Set<number>(),
  };
  const bump = (m: Map<string, number>, k: string) =>
    m.set(k, (m.get(k) ?? 0) + 1);
  const rowIndexByQuery = new Map(rows.map((r, i) => [r.query, i]));

  const pool: Queryable = {
    query: async (text: string, params: unknown[]) => {
      if (text.includes("WITH RECURSIVE chain")) {
        // Identity chains: every gold id resolves to itself.
        const ids = params[0] as string[];
        return { rows: ids.map((id) => ({ root: id, id })) };
      }
      if (text.includes("SELECT id, type FROM memory.memories")) {
        const ids = params[0] as string[];
        return {
          rows: ids.map((id) => ({ id, type: GOLD_TYPE[id] ?? "semantic" })),
        };
      }
      throw new Error(`mock pool: unexpected SQL: ${text}`);
    },
  };

  const deps: BakeoffBlendDeps = {
    pool,
    fuse: (async (input: {
      projectId: string;
      query: string;
      qvec?: number[];
      rrfK?: number;
    }) => {
      counters.fuse += 1;
      bump(counters.fusePerQuery, input.query);
      counters.rrfKsSeen.add(input.rrfK ?? 60);
      expect(input.qvec).toBeDefined(); // the embed is reused via qvec — never absent
      const idx = rowIndexByQuery.get(input.query);
      if (idx === undefined) throw new Error(`mock fuse: ${input.query}`);
      return candidatesFor(idx);
    }) as unknown as BakeoffBlendDeps["fuse"],
    rerankFn: async (query, docs, topK) => {
      counters.rerank += 1;
      bump(counters.rerankPerQuery, query);
      const scored = docs.map((d, index) => ({
        index,
        score: Number(d.split("|")[1]?.split("\n")[0] ?? 0),
      }));
      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, Math.min(topK, docs.length));
    },
    embedQuery: async (query) => {
      counters.embed += 1;
      bump(counters.embedPerQuery, query);
      return [0.1, 0.2, 0.3];
    },
    log: () => {},
    now: NOW,
  };
  return { deps, counters };
}

const BLEND_ARMS = ["A0", "A1", "A2", "A3", "A4"];

describe("bakeoff:blend — mocked injectables (zero network)", () => {
  it("AC-703: fuse + rerank run exactly ONCE per query while all 5 blend arms score", async () => {
    const { deps, counters } = mockWorld();
    const artifact = await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      arms: BLEND_ARMS,
    });
    expect(Object.keys(artifact.arms)).toEqual(BLEND_ARMS);
    expect(counters.fuse).toBe(16);
    expect(counters.rerank).toBe(16);
    expect(counters.embed).toBe(16);
    for (const [, n] of counters.fusePerQuery) expect(n).toBe(1);
    for (const [, n] of counters.rerankPerQuery) expect(n).toBe(1);
    expect([...counters.rrfKsSeen]).toEqual([60]);
  });

  it("RRF_K axis re-pools per k (capture-once WITHIN each k; qvec embed reused across k)", async () => {
    const { deps, counters } = mockWorld();
    await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      // default arms: A0..A4 + K20 + K120 → k ∈ {20, 60, 120}
    });
    expect([...counters.rrfKsSeen].sort((a, b) => a - b)).toEqual([
      20, 60, 120,
    ]);
    expect(counters.fuse).toBe(16 * 3);
    expect(counters.rerank).toBe(16 * 3);
    // The paid embed is computed once per query and REUSED across k.
    expect(counters.embed).toBe(16);
    for (const [, n] of counters.embedPerQuery) expect(n).toBe(1);
    for (const [, n] of counters.fusePerQuery) expect(n).toBe(3);
  });

  it("per-arm nDCG hand-check on the discriminating row (m-001)", async () => {
    const { deps } = mockWorld();
    const artifact = await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      arms: BLEND_ARMS,
    });
    const q = (arm: string) =>
      artifact.arms[arm].perQuery.find((p) => p.id === "m-001")!;
    const rank2 = 1 / Math.log2(3); // gold at rank 2 of 2 → nDCG@10 = 1/log2(3)
    expect(q("A0").ndcg).toBeCloseTo(rank2, 12);
    expect(q("A0").rank).toBe(2);
    expect(q("A1").ndcg).toBe(1); // entity exempt → gold outranks the fresh distractor
    expect(q("A2").ndcg).toBe(1);
    expect(q("A3").ndcg).toBeCloseTo(rank2, 12); // multiplicative keeps distractor first
    expect(q("A4").ndcg).toBe(1); // relevance-only: 0.60 > 0.58
    // Trivial rows score 1 everywhere → hand-computed arm means.
    expect(artifact.arms.A0.ndcgMean).toBeCloseTo((15 + rank2) / 16, 12);
    expect(artifact.arms.A1.ndcgMean).toBe(1);
    // Pool recall is arm-independent here (identical capture): gold always pooled.
    for (const arm of BLEND_ARMS) {
      expect(artifact.arms[arm].poolRecall).toBe(1);
    }
  });

  it("AC-705: CI / sign / affected / slices equal DIRECT eval-core calls at seed 42", async () => {
    const { deps } = mockWorld();
    const artifact = await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      arms: BLEND_ARMS,
    });
    const a0ById = new Map(artifact.arms.A0.perQuery.map((p) => [p.id, p]));
    for (const arm of ["A1", "A2", "A3", "A4"]) {
      const cmp = artifact.comparisons[arm];
      const deltas = artifact.arms[arm].perQuery.map(
        (p) => p.ndcg - a0ById.get(p.id)!.ndcg,
      );
      expect(cmp.ci).toEqual(pairedBootstrapCI(deltas, { seed: SEED }));
      expect(cmp.signTest).toEqual(signTest(deltas));
      expect(cmp.affected).toBe(
        deltas.filter((d) => Math.abs(d) > AFFECTED_EPS).length,
      );
      expect(cmp.meanDelta).toBeCloseTo(
        deltas.reduce((a, b) => a + b, 0) / deltas.length,
        12,
      );
    }
    // Slice shape: m-001's gold is type entity; all trivial rows are semantic.
    const a1 = artifact.comparisons.A1;
    expect(a1.sliceByType.entity).toEqual({
      meanDelta: expect.closeTo(1 - 1 / Math.log2(3), 10),
      n: 1,
    });
    expect(a1.sliceByType.semantic).toEqual({ meanDelta: 0, n: 15 });
    // Temporal slice: m-002/m-003 carry facet:"temporal" (both trivial → delta 0).
    expect(a1.sliceTemporal).toEqual({ meanDelta: 0, n: 2 });
    expect(a1.signTest).toMatchObject({ wins: 1, losses: 0, ties: 15 });
    expect(a1.affected).toBe(1);
    // One winning row in 16 cannot exclude zero → no qualifier → KEEP A0.
    expect(artifact.verdict.winner).toBe("A0");
  });

  it("refuses fewer than 15 split rows (throws before any capture)", async () => {
    const { deps, counters } = mockWorld(evalRows(14));
    await expect(
      runBakeoffBlend(deps, {
        evalFile: evalFile(evalRows(14)),
        projectId: "mock",
        split: "dev",
        arms: BLEND_ARMS,
      }),
    ).rejects.toThrow(new RegExp(`at least ${MIN_SPLIT_ROWS}`));
    expect(counters.fuse).toBe(0);
    expect(counters.embed).toBe(0);
  });

  it("A0-drift assertion fires on a perturbed live config (and aborts before capture)", async () => {
    const { deps, counters } = mockWorld();
    const orig = config.blendConfig.weights.relevance;
    config.blendConfig.weights.relevance = 0.75;
    try {
      await expect(
        runBakeoffBlend(deps, {
          evalFile: evalFile(),
          projectId: "mock",
          split: "dev",
          arms: BLEND_ARMS,
        }),
      ).rejects.toThrow(/drift/);
      expect(counters.fuse).toBe(0);
    } finally {
      config.blendConfig.weights.relevance = orig;
    }
  });

  it("--arms A0,A3 subsets correctly (control always present, k=60 capture only)", async () => {
    const { deps, counters } = mockWorld();
    const artifact = await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      arms: ["A0", "A3"],
    });
    expect(Object.keys(artifact.arms)).toEqual(["A0", "A3"]);
    expect(Object.keys(artifact.comparisons)).toEqual(["A3"]);
    expect([...counters.rrfKsSeen]).toEqual([60]);
    // A0 is forced in even when omitted — comparisons need the control.
    const again = await runBakeoffBlend(mockWorld().deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
      arms: ["A4"],
    });
    expect(Object.keys(again.arms)).toEqual(["A0", "A4"]);
  });

  it("unknown arm names fail loudly", async () => {
    const { deps } = mockWorld();
    await expect(
      runBakeoffBlend(deps, {
        evalFile: evalFile(),
        projectId: "mock",
        split: "dev",
        arms: ["A0", "A9"],
      }),
    ).rejects.toThrow(/unknown arm/);
  });

  it("AC-704: no --yes refuses (pure parse — before ANY dependency is invoked) and defaults to dev", () => {
    expect(parseBakeoffBlendArgs([])).toBe(BAKEOFF_BLEND_COST_NOTE);
    expect(parseBakeoffBlendArgs(["--split", "test"])).toBe(
      BAKEOFF_BLEND_COST_NOTE,
    );
    expect(parseBakeoffBlendArgs(["--yes"])).toEqual({
      yes: true,
      split: "dev", // the frozen test split is NOT the selection surface
      evalPath: undefined,
      arms: undefined,
      projectId: config.defaultProjectId,
      outDir: undefined,
    });
    expect(
      parseBakeoffBlendArgs([
        "--yes",
        "--split",
        "test",
        "--arms",
        "A0,A3",
        "--project-id",
        "p1",
        "--out-dir",
        "/tmp/x",
      ]),
    ).toEqual({
      yes: true,
      split: "test",
      evalPath: undefined,
      arms: ["A0", "A3"],
      projectId: "p1",
      outDir: "/tmp/x",
    });
    expect(parseBakeoffBlendArgs(["--yes", "--split", "prod"])).toMatch(
      /invalid --split/,
    );
  });

  it("artifact: seed 42, all predeclared arms, config snapshot — written ONLY to a temp dir", async () => {
    const { deps } = mockWorld();
    const artifact = await runBakeoffBlend(deps, {
      evalFile: evalFile(),
      projectId: "mock",
      split: "dev",
    });
    const runsBefore = readdirSync(join(here, "runs"));
    const tmp = mkdtempSync(join(tmpdir(), "w7-bakeoff-blend-"));
    try {
      const path = writeBakeoffBlendArtifact(artifact, tmp);
      expect(path.startsWith(tmp)).toBe(true);
      const parsed = JSON.parse(readFileSync(path, "utf8")) as typeof artifact;
      expect(parsed.seed).toBe(42);
      expect(Object.keys(parsed.arms)).toEqual([
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "K20",
        "K120",
      ]);
      expect(parsed.split).toBe("dev");
      expect(parsed.rows).toBe(16);
      expect(parsed.config).toMatchObject({
        service: "memory-mcp",
        scoringVersion: "blend-2",
      });
      expect(parsed.rejectedForNow).toEqual(["access-based decay"]);
      for (const arm of Object.values(parsed.arms)) {
        expect(arm.armConfig).toMatchObject({ rrfK: expect.any(Number) });
        expect(arm.perQuery).toHaveLength(16);
      }
      expect(parsed.arms.K20.armConfig.rrfK).toBe(20);
      expect(parsed.arms.K120.armConfig.rrfK).toBe(120);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
    // Nothing leaked into the committed test/runs directory.
    expect(readdirSync(join(here, "runs"))).toEqual(runsBefore);
  });

  it("zero network: a full mocked run never touches fetch", async () => {
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("bakeoff-blend test: network is forbidden");
    }) as typeof fetch;
    try {
      const { deps } = mockWorld();
      const artifact = await runBakeoffBlend(deps, {
        evalFile: evalFile(),
        projectId: "mock",
        split: "dev",
      });
      expect(artifact.rows).toBe(16);
    } finally {
      globalThis.fetch = origFetch;
    }
  });
});

// ── DB-backed golden A0 equivalence (skipped cleanly without DATABASE_URL) ──────────

const PROJ = "w7-blend-golden";
const gid = (n: number) =>
  `70000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;
const DBNOW = new Date("2026-07-01T12:00:00Z");

interface DbFixture {
  id: string;
  type: MemoryType;
  title: string;
  content: string;
  importance: number;
  createdAt: string;
  sourceKind: string | null;
}

const DB_FIXTURES: DbFixture[] = [
  {
    id: gid(1),
    type: "semantic",
    title: "Pool sizing guidance",
    content:
      "postgres connection pool sizing guidance: cap max at ten per service and tune idle timeout under load",
    importance: 0.9,
    createdAt: "2026-06-10T00:00:00Z",
    sourceKind: null,
  },
  {
    id: gid(2),
    type: "procedural",
    title: "Rollback steps",
    content:
      "rollback a bad deployment by redeploying the previous tag and verifying health checks first",
    importance: 0.7,
    createdAt: "2026-06-28T00:00:00Z",
    sourceKind: null,
  },
  {
    id: gid(3),
    type: "episodic",
    title: "Connection storm incident",
    content:
      "incident: postgres connections exhausted after a deploy loop opened one pool per request",
    importance: 0.6,
    createdAt: "2026-06-25T00:00:00Z",
    sourceKind: null,
  },
  {
    id: gid(4),
    type: "semantic",
    title: "Fusion decision record",
    content:
      "hybrid retrieval fuses bm25 and vector ranks with reciprocal rank fusion so neither arm dominates",
    importance: 0.8,
    createdAt: "2026-06-20T00:00:00Z",
    sourceKind: "decision",
  },
  {
    id: gid(5),
    type: "entity",
    title: "Voyage embeddings vendor",
    content:
      "voyage provides the contextual embedding and rerank models used by the memory pipeline",
    importance: 0.4,
    createdAt: "2026-06-27T00:00:00Z",
    sourceKind: null,
  },
  {
    id: gid(6),
    type: "semantic",
    title: "Index rebuild guidance",
    content:
      "hnsw ef search trades recall for latency and the index is rebuilt concurrently after mass re-embeds",
    importance: 0.65,
    createdAt: "2026-06-15T00:00:00Z",
    sourceKind: null,
  },
];

describe.skipIf(skipDb)("golden A0 equivalence (DB)", () => {
  let db: pg.Pool;

  beforeAll(async () => {
    vi.useFakeTimers({ toFake: ["Date"], now: DBNOW });
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    const sqlDir = join(here, "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await db.query(`DELETE FROM memory.memories WHERE project_id = $1`, [PROJ]);
    for (const f of DB_FIXTURES) {
      await db.query(
        `INSERT INTO memory.memories
           (id, project_id, type, title, content, importance, created_at, source_kind, embedding_v2)
         VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6, $7, $8, $9::halfvec)`,
        [
          f.id,
          PROJ,
          f.type,
          f.title,
          f.content,
          f.importance,
          f.createdAt,
          f.sourceKind,
          `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
        ],
      );
    }
  });

  afterAll(async () => {
    await db.query(`DELETE FROM memory.memories WHERE project_id = $1`, [PROJ]);
    await db.end();
    vi.useRealTimers();
  });

  it("blendScores(capture, A0) ordering === searchMemory() ordering on seeded rows", async () => {
    const a0 = predeclaredArms().find((a) => a.name === "A0")!.blend;
    for (const query of [
      "postgres connection pool sizing",
      "rollback deployment health checks",
      "reciprocal rank fusion decision",
    ]) {
      const input = { projectId: PROJ, query, limit: 5 };
      // Offline path: capture (fuse + rerank once) → pure blendScores under A0.
      const pool = await fuseCandidates(input);
      expect(pool.length).toBeGreaterThan(0);
      const docs = pool.map((r) =>
        `${r.title}\n${r.content}`.slice(0, RERANK_DOC_TRUNCATION),
      );
      const ranked = await rerank(query, docs, pool.length);
      const relById = new Map(ranked.map((r) => [pool[r.index].id, r.score]));
      const offline = blendScores(
        pool.map((r) => ({ ...r, relevance: relById.get(r.id) ?? 0 })),
        a0,
      );
      // Live path: the full pipeline.
      const oneShot = await searchMemory(input);
      expect(offline.slice(0, input.limit).map((c) => c.id)).toEqual(
        oneShot.hits.map((h) => h.id),
      );
    }
  });

  it("fuseCandidates carries source_kind for the exemption logic (wave-7 column add)", async () => {
    const pool = await fuseCandidates({
      projectId: PROJ,
      query: "reciprocal rank fusion decision",
    });
    const decision = pool.find((c) => c.id === gid(4));
    expect(decision).toBeTruthy();
    expect(decision!.source_kind).toBe("decision");
    const plain = pool.find((c) => c.id !== gid(4));
    if (plain) expect(plain.source_kind).toBeNull();
  });

  it("formatted output remains rank-only (no score leaks)", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "postgres connection pool sizing",
      limit: 5,
    });
    for (const h of result.hits) expect("score" in h).toBe(false);
    const formatted = formatHits(result.hits, result);
    expect(formatted.length).toBeGreaterThan(0);
    expect(formatted).not.toMatch(/score/i);
    // No 4-decimal blend scores anywhere in the rendered lines.
    expect(formatted).not.toMatch(/\b0\.\d{4}\b/);
  });
});
