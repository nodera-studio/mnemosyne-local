// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Waves 1–3 (memory-mcp half)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/index.md).
// Every assertion below was drafted from the EARS acceptance criteria BEFORE
// locating the exported symbols; this suite is the independent anti-reward-hacking
// gate, additive to the Implementer's own tests.
//
// AC map (memory-mcp scope):
//   AC-101 — composed pipeline output contract (snippet/rank-only/limit/[])
//            + determinism + two-phase composition ≡ one-shot searchMemory
//   AC-102 — fuseCandidates pool: ≤ candidatePool rows, per-arm RRF ranks,
//            live handler consumes exactly this pool (finals ⊆ pool)
//   AC-103 — ndcgAtK / pairedBootstrapCI vs hand-computed values; seed determinism
//   AC-104 — retrievalConfig() full snapshot; eval run artifact carries the
//            snapshot + per-query scores for BOTH layers; test/runs/ convention
//   AC-105 — gate math semantics (paired-bootstrap CI regression side; id-drift
//            fails loudly). The gate test's own skip ladder is verified at the
//            suite level (run with/without DATABASE_URL), not in-process.
//   AC-106 — resolveGoldIds walks superseded_by forward chains (+ cycle guard);
//            end-to-end: a superseded gold row is credited via its successor
//   AC-107 — search_log is fire-and-forget: row lands async; a held lock does
//            NOT block searchMemory; a missing table changes nothing
//   AC-108 — quota-spending scripts: npm-run entries exist, import is
//            side-effect-free, import.meta.url main-guard present
//   AC-109 — distilled candidates are approved:false and never touch gold;
//            v1 gold file gates v2 consumers off until human approval
//   AC-601 — memory-mcp vitest executes test files serially
//
// Deterministic throughout: Voyage is module-mocked (seeded PRNG vectors/scores),
// fixture vectors are inserted as literals, Date is frozen for the DB block.
// DB tests are self-contained under conf-w13-mem-* project ids and clean up.
// NO live Voyage/Anthropic quota is ever spent here.
// ─────────────────────────────────────────────────────────────────────────────

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic helpers, hoisted so the vi.mock factory can reference them.
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

// HARD RULE: never call live Voyage — module-mock the boundary.
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
  applyQuerySnippets,
  fuseCandidates,
  rerankAndBlend,
  retrievalConfig,
  searchMemory,
  RERANK_DOC_TRUNCATION,
  SNIPPET_CHARS,
  type MemoryType,
} from "../src/memory.js";
import {
  ndcgAtK,
  mulberry32 as prng,
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  resolveGoldIds,
  runRecallEval,
  tryLoadRecallEvalV2,
  writeRunArtifact,
  type RecallEvalFileV2,
  type RunArtifact,
} from "./recall.helper.js";
import { distillEval } from "../src/db/distill-eval.js";

const here = dirname(fileURLToPath(import.meta.url));
const serviceRoot = join(here, "..");

// Unique fixture namespace — never collides with other suites' project ids.
const P_SEARCH = "conf-w13-mem-search";
const P_POOL = "conf-w13-mem-pool";
const P_RUN = "conf-w13-mem-run";
const P_CHAIN = "conf-w13-mem-chain";
const P_EMPTY = "conf-w13-mem-empty";
const ALL_PROJECTS = [P_SEARCH, P_POOL, P_RUN, P_CHAIN, P_EMPTY];

const mid = (n: number) =>
  `20000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// Frozen instant AFTER every fixture created_at, so the recency blend is exact.
const NOW = new Date("2026-07-01T12:00:00Z");

// ── Pure: AC-103 — math layer vs hand-computed values ────────────────────────

describe("AC-103: ndcgAtK matches hand-computed fixtures (pure)", () => {
  it("relevant at rank 1 only → 1.0", () => {
    expect(ndcgAtK(["r", "x", "y"], ["r"], 10)).toBeCloseTo(1.0, 10);
  });

  it("single relevant at rank 3, k=10 → 1/log2(4) = 0.5", () => {
    expect(ndcgAtK(["a", "b", "r"], ["r"], 10)).toBeCloseTo(0.5, 10);
  });

  it("two relevant at ranks 2+4, k=10, |rel|=2 → (1/log2(3)+1/log2(5))/(1+1/log2(3))", () => {
    const expected =
      (1 / Math.log2(3) + 1 / Math.log2(5)) / (1 + 1 / Math.log2(3));
    expect(ndcgAtK(["a", "r1", "b", "r2"], ["r1", "r2"], 10)).toBeCloseTo(
      expected,
      10,
    );
  });

  it("relevant absent → 0; k=0 → 0; empty relevant → 0", () => {
    expect(ndcgAtK(["a", "b"], ["r"], 10)).toBe(0);
    expect(ndcgAtK(["r"], ["r"], 0)).toBe(0);
    expect(ndcgAtK(["a"], [], 10)).toBe(0);
  });

  it("IDCG truncates at k < |relevant|: 2 relevants at ranks 1+2, k=2, |rel|=3 → 1.0", () => {
    expect(ndcgAtK(["r1", "r2"], ["r1", "r2", "r3"], 2)).toBeCloseTo(1.0, 10);
  });
});

describe("AC-103: pairedBootstrapCI is exact on constants and seed-deterministic (pure)", () => {
  it("constant deltas [0.1 ×20] → mean = ciLow = ciHigh = 0.1", () => {
    const ci = pairedBootstrapCI(Array(20).fill(0.1), { seed: 7 });
    expect(ci.mean).toBeCloseTo(0.1, 10);
    expect(ci.ciLow).toBeCloseTo(0.1, 10);
    expect(ci.ciHigh).toBeCloseTo(0.1, 10);
  });

  it("same seed twice → identical output (determinism)", () => {
    const deltas = Array.from({ length: 30 }, (_, i) =>
      i % 2 === 0 ? 0.08 : -0.03,
    );
    const a = pairedBootstrapCI(deltas, { seed: 512, iterations: 2000 });
    const b = pairedBootstrapCI(deltas, { seed: 512, iterations: 2000 });
    expect(a).toEqual(b);
  });

  it("empty input → zeros", () => {
    expect(pairedBootstrapCI([], { seed: 1 })).toEqual({
      mean: 0,
      ciLow: 0,
      ciHigh: 0,
    });
  });

  it("mulberry32: same seed → same first 5 values", () => {
    const a = prng(1234);
    const b = prng(1234);
    const seqA = [a(), a(), a(), a(), a()];
    const seqB = [b(), b(), b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });
});

// ── Pure: AC-105 — gate math semantics (delta = fresh − baseline) ────────────

function syntheticArtifact(ndcgs: number[]): RunArtifact {
  return {
    retrievalConfig: {},
    evalVersion: 2,
    split: "dev",
    rows: ndcgs.length,
    perQuery: ndcgs.map((ndcg, i) => ({
      id: `m-${String(i + 1).padStart(3, "0")}`,
      query: `q${i + 1}`,
      poolRecall: 1,
      ndcg,
      mrr: ndcg,
      rank: 1,
    })),
    aggregates: { recallAt25: 1, ndcgAt10: 0, mrr10: 0 },
  };
}

describe("AC-105: paired-bootstrap gate semantics (pure)", () => {
  const base = Array.from({ length: 20 }, (_, i) => 0.5 + (i % 5) * 0.05);

  it("uniform regression (fresh 0.2 below baseline) → CI excludes zero on the regression side", () => {
    const baseline = syntheticArtifact(base);
    const fresh = syntheticArtifact(base.map((v) => v - 0.2));
    const deltas = perQueryDeltas(baseline, fresh);
    const ci = pairedBootstrapCI(deltas, { seed: 42 });
    expect(regressionExcluded(ci)).toBe(true);
  });

  it("identical runs → gate passes (CI does not exclude zero)", () => {
    const baseline = syntheticArtifact(base);
    const fresh = syntheticArtifact(base);
    const ci = pairedBootstrapCI(perQueryDeltas(baseline, fresh), { seed: 42 });
    expect(regressionExcluded(ci)).toBe(false);
  });

  it("noise straddling zero → gate passes", () => {
    const fresh = syntheticArtifact(
      base.map((v, i) => v + (i % 2 ? 0.02 : -0.02)),
    );
    const ci = pairedBootstrapCI(
      perQueryDeltas(syntheticArtifact(base), fresh),
      { seed: 42 },
    );
    expect(regressionExcluded(ci)).toBe(false);
  });

  it("row-id drift between baseline and fresh run fails loudly", () => {
    const baseline = syntheticArtifact(base);
    const fresh = syntheticArtifact(base.slice(0, 19));
    expect(() => perQueryDeltas(baseline, fresh)).toThrow(/drift|id/i);
  });
});

// ── Pure: AC-104 — retrievalConfig() snapshot contract ───────────────────────

describe("AC-104: retrievalConfig() full snapshot (pure)", () => {
  it("contains every plan-named key with the values the pipeline actually runs", () => {
    const cfg = retrievalConfig();
    expect(cfg.service).toBe("memory-mcp");
    expect(typeof cfg.rrfK).toBe("number");
    // Wave-7 (AC-702): nested blend config + top-level scoringVersion replace the
    // retired scalar recencyHalfLifeDays.
    expect(typeof cfg.scoringVersion).toBe("string");
    expect(cfg.blend).toEqual({
      form: expect.any(String),
      weights: {
        relevance: expect.any(Number),
        recency: expect.any(Number),
        importance: expect.any(Number),
      },
      decay: {
        shape: expect.any(String),
        tauDays: expect.any(Number),
        tauDaysByType: expect.any(Object),
        powerExponent: expect.any(Number),
        exempt: {
          types: expect.any(Array),
          sourceKinds: expect.any(Array),
        },
      },
    });
    expect("recencyHalfLifeDays" in cfg).toBe(false);
    expect(cfg.candidatePool).toBe(25);
    expect(typeof cfg.recallLimit).toBe("number");
    expect(typeof cfg.embedModel).toBe("string");
    expect(typeof cfg.contextModel).toBe("string");
    expect(typeof cfg.rerankModel).toBe("string");
    // The snapshot must equal the constants the pipeline uses (single source).
    expect(cfg.rerankDocTruncation).toBe(RERANK_DOC_TRUNCATION);
    expect(cfg.rerankDocTruncation).toBe(1200);
    expect(cfg.snippetChars).toBe(SNIPPET_CHARS);
    expect(cfg.snippetChars).toBe(180);
    // Wave-2 (AC-810): the doc composition is a ranking-bearing knob.
    expect(cfg.rerankDocIncludesSummary).toBe(true);
  });

  it("is JSON-safe (survives a JSON round-trip verbatim)", () => {
    const cfg = retrievalConfig();
    expect(JSON.parse(JSON.stringify(cfg))).toEqual(cfg);
  });

  it("test/runs/ artifact directory convention exists", () => {
    expect(existsSync(join(here, "runs"))).toBe(true);
  });
});

// ── Pure: AC-108 — quota-spending scripts run only via explicit npm run ──────

describe("AC-108: paid operator scripts (pure)", () => {
  const pkg = JSON.parse(
    readFileSync(join(serviceRoot, "package.json"), "utf8"),
  ) as { scripts: Record<string, string> };

  it("package.json exposes the operator script entries", () => {
    expect(pkg.scripts["distill-eval"]).toBeTruthy();
    expect(pkg.scripts["harvest-eval"]).toBeTruthy();
    expect(pkg.scripts["gold:migrate"]).toBeTruthy();
    expect(pkg.scripts["backfill:context"]).toBeTruthy();
  });

  const scriptFiles = [
    "src/db/distill-eval.ts",
    "src/db/harvest-eval.ts",
    "src/db/migrate-gold.ts",
    "src/db/backfill-context.ts",
  ];

  it("every operator script carries an import.meta.url main-guard", () => {
    for (const f of scriptFiles) {
      const src = readFileSync(join(serviceRoot, f), "utf8");
      expect(src, `${f} must main-guard on import.meta.url`).toMatch(
        /import\.meta\.url/,
      );
    }
  });

  it("importing the script modules executes nothing (no network, no exit)", async () => {
    const fetchCalls: unknown[][] = [];
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async (...args: unknown[]) => {
      fetchCalls.push(args);
      throw new Error("conformance: no network on import");
    }) as typeof fetch;
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((
      code?: number,
    ) => {
      throw new Error(`conformance: process.exit(${code}) during import`);
    }) as never);
    try {
      await import("../src/db/distill-eval.js");
      await import("../src/db/harvest-eval.js");
      await import("../src/db/migrate-gold.js");
      await import("../src/db/backfill-context.js");
      expect(fetchCalls).toHaveLength(0);
      expect(exitSpy).not.toHaveBeenCalled();
    } finally {
      globalThis.fetch = origFetch;
      exitSpy.mockRestore();
    }
  });
});

// ── Pure: AC-109 — human approval gates every gold mutation ──────────────────

describe("AC-109: distilled candidates never enter gold without approval (pure)", () => {
  const goldPath = join(here, "fixtures", "recall-eval.json");

  it("distillEval emits approved:false candidates and does not touch the gold file", async () => {
    const goldBefore = readFileSync(goldPath, "utf8");
    const tmp = mkdtempSync(join(tmpdir(), "conf-w13-distill-"));
    try {
      const outPath = join(tmp, "candidates.json");
      const sampled = [
        {
          id: mid(900),
          type: "semantic",
          source_kind: "docs",
          title: "Sample title one",
          excerpt: "sample excerpt one",
        },
        {
          id: mid(901),
          type: "episodic",
          source_kind: null,
          title: "Sample title two",
          excerpt: "sample excerpt two",
        },
      ];
      const modelReply = JSON.stringify([
        {
          query: "where is the sample one documented",
          suggestedGold: [mid(900)],
          archetype: "conceptual",
        },
        {
          query: "sample title two",
          suggestedGold: [mid(901)],
          archetype: "exact-title",
        },
      ]);
      const { candidates, path } = await distillEval({
        pool: { query: async () => ({ rows: sampled }) },
        complete: async () => modelReply,
        projectId: "conf-w13-distill",
        outPath,
        log: () => {},
      });
      expect(candidates.length).toBeGreaterThan(0);
      for (const c of candidates) {
        expect(c.approved).toBe(false);
        expect(c.provenance).toBe("distilled");
      }
      const written = JSON.parse(readFileSync(path, "utf8")) as Array<{
        approved: boolean;
      }>;
      // Wrapper-tolerant: candidates may be the array itself or under a key.
      const rows = Array.isArray(written)
        ? written
        : (Object.values(written).find(Array.isArray) as {
            approved: boolean;
          }[]);
      expect(rows.length).toBeGreaterThan(0);
      for (const r of rows) expect(r.approved).toBe(false);
      // The gold eval file is byte-identical — candidates NEVER merge themselves.
      expect(readFileSync(goldPath, "utf8")).toBe(goldBefore);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("gold file gates v2 consumers until the operator approves the migration", () => {
    const raw = JSON.parse(readFileSync(goldPath, "utf8")) as {
      version?: number;
      changelog?: string[];
      rows: Array<{ split?: string; id?: string; provenance?: string }>;
    };
    const v2 = tryLoadRecallEvalV2(goldPath);
    if (raw.version === 2) {
      // Post-G0 state: v2 header with changelog, every row split-assigned.
      expect(v2).not.toBeNull();
      expect(Array.isArray(raw.changelog)).toBe(true);
      for (const r of v2!.rows) {
        expect(r.id).toBeTruthy();
        expect(["dev", "test"]).toContain(r.split);
        expect(r.provenance).toBeTruthy();
      }
    } else {
      // Pre-G0 state: loader must return null so gates skip (no un-approved gold).
      expect(v2).toBeNull();
    }
  });
});

// ── Pure: AC-601 — memory-mcp vitest runs test files serially ────────────────

describe("AC-601: vitest serial file execution (pure)", () => {
  it("vitest.config.ts sets fileParallelism: false", () => {
    const cfg = readFileSync(join(serviceRoot, "vitest.config.ts"), "utf8");
    expect(cfg).toMatch(/fileParallelism:\s*false/);
  });
});

// ── DB-backed conformance (skipped cleanly without DATABASE_URL) ─────────────

interface SearchFixture {
  id: string;
  type: MemoryType;
  title: string;
  content: string;
  importance: number;
  createdAt: string;
}

// 8 rows: distinct content lengths (rerank determinism), distinct importance and
// created_at (tie-free blend). Row 3 carries raw newlines + runs of spaces so the
// whitespace-collapsed 180-char snippet contract is exercised for real.
const SEARCH_FIXTURES: SearchFixture[] = [
  {
    id: mid(1),
    type: "semantic",
    title: "Connection pool sizing",
    content:
      "postgres connection pool sizing: cap max at ten per service and watch idle timeouts under sustained load",
    importance: 0.9,
    createdAt: "2026-06-10T00:00:00Z",
  },
  {
    id: mid(2),
    type: "procedural",
    title: "Rollback runbook",
    content:
      "rollback a bad deployment by redeploying the previous image tag and verifying health checks before announcing",
    importance: 0.7,
    createdAt: "2026-06-28T00:00:00Z",
  },
  {
    id: mid(3),
    type: "episodic",
    title: "Incident connection storm",
    content:
      "incident report:\n  postgres connections exhausted   after a deploy loop\n  opened a new pool per request; fixed by sharing the singleton pool and lowering the per-service ceiling so the database stopped refusing connections during the retry storm window",
    importance: 0.6,
    createdAt: "2026-06-25T00:00:00Z",
  },
  {
    id: mid(4),
    type: "semantic",
    title: "RRF fusion decision",
    content:
      "hybrid retrieval fuses bm25 and vector ranks with reciprocal rank fusion so neither arm dominates the candidate pool",
    importance: 0.8,
    createdAt: "2026-06-20T00:00:00Z",
  },
  {
    id: mid(5),
    type: "semantic",
    title: "Contextual embedder choice",
    content:
      "voyage context four embeds each chunk in document context and is the corpus embedder for memory",
    importance: 0.4,
    createdAt: "2026-06-27T00:00:00Z",
  },
  {
    id: mid(6),
    type: "procedural",
    title: "Backup restore drill",
    content:
      "restore the nightly backup into a scratch database, verify row counts, then swap the connection string",
    importance: 0.55,
    createdAt: "2026-06-18T00:00:00Z",
  },
  {
    id: mid(7),
    type: "episodic",
    title: "Migration ordering note",
    content:
      "the decision log migration landed late and the batch runner applies files strictly in filename order",
    importance: 0.3,
    createdAt: "2026-06-29T00:00:00Z",
  },
  {
    id: mid(8),
    type: "semantic",
    title: "HNSW rebuild guidance",
    content:
      "hnsw ef search trades recall for latency; rebuild the index concurrently after any mass re-embed",
    importance: 0.65,
    createdAt: "2026-06-15T00:00:00Z",
  },
];
const searchById = new Map(SEARCH_FIXTURES.map((f) => [f.id, f]));

function iso(s: string): string {
  return new Date(s).toISOString();
}

// Supersession chain fixtures (AC-106): A → B → C forward via superseded_by.
const CHAIN_A = mid(101);
const CHAIN_B = mid(102);
const CHAIN_C = mid(103);
const CYCLE_X = mid(104);
const CYCLE_Y = mid(105);

// Runner fixtures (AC-104 + AC-106 end-to-end).
const RUN_GOOD = mid(201);
const RUN_SUPER_A = mid(202); // gold id; superseded, unfindable by search
const RUN_SUCC_B = mid(203); // its successor; what search actually returns

describe.skipIf(skip)("W1–W3 conformance — DB-backed (memory-mcp)", () => {
  let db: pg.Pool;

  async function insertMemory(row: {
    id: string;
    projectId: string;
    type: MemoryType;
    title: string;
    content: string;
    importance?: number;
    createdAt?: string;
    embedKey?: string | null;
    supersededBy?: string | null;
  }): Promise<void> {
    await db.query(
      `INSERT INTO memory.memories
         (id, project_id, type, title, content, importance, created_at, superseded_by, embedding_v2)
       VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6, $7, $8, $9::halfvec)`,
      [
        row.id,
        row.projectId,
        row.type,
        row.title,
        row.content,
        row.importance ?? 0.5,
        row.createdAt ?? "2026-06-20T00:00:00Z",
        row.supersededBy ?? null,
        row.embedKey ? `[${H.fakeVec(row.embedKey).join(",")}]` : null,
      ],
    );
  }

  async function cleanup(): Promise<void> {
    // Restore search_log if a crashed earlier run left it renamed.
    await db.query(`DO $$ BEGIN
      IF to_regclass('memory.search_log') IS NULL
         AND to_regclass('memory.search_log_confw13bak') IS NOT NULL THEN
        ALTER TABLE memory.search_log_confw13bak RENAME TO search_log;
      END IF;
    END $$;`);
    await db.query(
      `UPDATE memory.memories SET superseded_by = NULL WHERE project_id = ANY($1)`,
      [ALL_PROJECTS],
    );
    await db.query(`DELETE FROM memory.memories WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
    await db.query(`DELETE FROM memory.search_log WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
  }

  /** Poll a predicate on a real-time deadline (Date is frozen — use hrtime). */
  async function until(
    fn: () => Promise<boolean>,
    ms = 4000,
    step = 50,
  ): Promise<boolean> {
    const deadline = process.hrtime.bigint() + BigInt(ms) * 1_000_000n;
    while (process.hrtime.bigint() < deadline) {
      if (await fn()) return true;
      await new Promise((r) => setTimeout(r, step));
    }
    return fn();
  }

  beforeAll(async () => {
    vi.useFakeTimers({ toFake: ["Date"], now: NOW });
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Idempotent migration apply (HOLD files skipped) — self-sufficient on a fresh DB.
    const sqlDir = join(serviceRoot, "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();

    for (const f of SEARCH_FIXTURES) {
      await insertMemory({
        id: f.id,
        projectId: P_SEARCH,
        type: f.type,
        title: f.title,
        content: f.content,
        importance: f.importance,
        createdAt: f.createdAt,
        embedKey: `mem:${f.id}`,
      });
    }

    // Pool-cap project: 31 rows all matching one query token, all embedded.
    for (let i = 1; i <= 31; i++) {
      await insertMemory({
        id: mid(300 + i),
        projectId: P_POOL,
        type: "semantic",
        title: `Pool row ${i}`,
        content: `shared retrieval conformance token variant ${i} with filler text of varying width ${"x".repeat(i)}`,
        importance: 0.3 + (i % 7) * 0.09,
        createdAt: `2026-06-${String((i % 27) + 1).padStart(2, "0")}T00:00:00Z`,
        embedKey: `pool:${i}`,
      });
    }

    // Supersession chains: C first (FK), then B → C, then A → B; cycle X ↔ Y.
    for (const [id, sup] of [
      [CHAIN_C, null],
      [CHAIN_B, CHAIN_C],
      [CHAIN_A, CHAIN_B],
      [CYCLE_X, null],
      [CYCLE_Y, CYCLE_X],
    ] as Array<[string, string | null]>) {
      await insertMemory({
        id,
        projectId: P_CHAIN,
        type: "semantic",
        title: `chain ${id.slice(-3)}`,
        content: "supersession chain fixture row",
        supersededBy: sup,
      });
    }
    await db.query(
      `UPDATE memory.memories SET superseded_by = $1 WHERE id = $2`,
      [CYCLE_Y, CYCLE_X],
    );

    // Runner project: one plainly-findable gold row + one superseded gold row
    // whose SUCCESSOR is what search can find (AC-106 end-to-end).
    await insertMemory({
      id: RUN_GOOD,
      projectId: P_RUN,
      type: "procedural",
      title: "Graceful shutdown drain",
      content:
        "graceful shutdown drain sequence stops accepting traffic then flushes queues before exit",
      importance: 0.8,
      createdAt: "2026-06-22T00:00:00Z",
      embedKey: `run:${RUN_GOOD}`,
    });
    await insertMemory({
      id: RUN_SUCC_B,
      projectId: P_RUN,
      type: "semantic",
      title: "Corpus embedder migration",
      content:
        "voyage context corpus embedder migration notes: re-embed the whole column consistently on a swap",
      importance: 0.7,
      createdAt: "2026-06-26T00:00:00Z",
      embedKey: `run:${RUN_SUCC_B}`,
    });
    await insertMemory({
      id: RUN_SUPER_A,
      projectId: P_RUN,
      type: "semantic",
      title: "zqx wvut placeholder",
      content: "zqx wvut unrelated placeholder body",
      importance: 0.5,
      createdAt: "2026-06-01T00:00:00Z",
      embedKey: null, // no vector, no BM25 overlap — search can never return it
      supersededBy: RUN_SUCC_B,
    });
  });

  afterAll(async () => {
    await cleanup();
    await db.end();
    vi.useRealTimers();
  });

  // ── AC-101: composed pipeline output contract ──────────────────────────────

  function assertHitShape(
    hits: Awaited<ReturnType<typeof searchMemory>>["hits"],
    limit: number,
  ): void {
    expect(hits.length).toBeLessThanOrEqual(limit);
    for (const h of hits) {
      const f = searchById.get(h.id)!;
      expect(f).toBeTruthy();
      // Snippet contract (AC-801/802): a query-aware ts_headline extract with
      // `**` markers when the query lexically matches the content, otherwise the
      // deterministic 180-char whitespace-collapsed prefix. Both forms are
      // whitespace-collapsed; only the prefix form is char-bounded. Matches are
      // detected internally via private-use sentinels (never literal `**`, which
      // stored markdown may contain) and the sentinels must never reach display.
      // The `**` branch split below is sound because these fixtures contain none.
      expect(h.snippet).not.toMatch(/[\uE000\uE001]/);
      const collapsed = f.content.replace(/\s+/g, " ");
      if (h.snippet.includes("**")) {
        expect(h.snippet).not.toMatch(/\n| {2}/);
        // Every fragment (markers stripped, delimiter-split) is drawn verbatim
        // from this hit's own content.
        for (const fragment of h.snippet.split(" … ")) {
          expect(collapsed).toContain(fragment.replaceAll("**", ""));
        }
      } else {
        expect(h.snippet).toBe(collapsed.slice(0, 180));
        expect(h.snippet.length).toBeLessThanOrEqual(180);
      }
      expect(h.title).toBe(f.title);
      expect(h.type).toBe(f.type);
      expect("score" in h).toBe(false);
      expect(h.createdAt).toBe(iso(f.createdAt));
      expect(h.eventDate).toBeNull();
      expect(h.effectiveDate).toBe(h.createdAt);
      expect(h.status).toBe("active");
    }
  }

  it("AC-101/AC-203/AC-207: hit shape — snippet(headline or 180-prefix, collapsed), rank-only, dates/status, ≤ limit", async () => {
    const result = await searchMemory({
      projectId: P_SEARCH,
      query: "postgres connection pool",
      limit: 5,
    });
    const { hits } = result;
    expect(hits.length).toBeGreaterThan(0);
    assertHitShape(hits, 5);
  });

  it("AC-101: whitespace in content is collapsed in the snippet", async () => {
    const result = await searchMemory({
      projectId: P_SEARCH,
      query: "incident connection storm retry",
      limit: 8,
    });
    const { hits } = result;
    const noisy = hits.find((h) => h.id === mid(3));
    expect(noisy).toBeTruthy();
    expect(noisy!.snippet).not.toMatch(/\n| {2}/);
  });

  it("AC-101: type filter returns only that type", async () => {
    const result = await searchMemory({
      projectId: P_SEARCH,
      query: "embedder index rebuild",
      type: "semantic",
      limit: 5,
    });
    const { hits } = result;
    expect(hits.length).toBeGreaterThan(0);
    for (const h of hits) expect(h.type).toBe("semantic");
  });

  it("AC-101: empty corpus → []", async () => {
    const result = await searchMemory({
      projectId: P_EMPTY,
      query: "anything at all",
      limit: 5,
    });
    expect(result.hits).toEqual([]);
  });

  it("AC-101: identical inputs → identical output (deterministic pipeline)", async () => {
    const input = {
      projectId: P_SEARCH,
      query: "rollback deployment health checks",
      limit: 5,
    };
    const a = await searchMemory(input);
    const b = await searchMemory(input);
    expect(b.hits).toEqual(a.hits);
  });

  it("AC-101/AC-102: two-phase composition ≡ one-shot searchMemory", async () => {
    const input = {
      projectId: P_SEARCH,
      query: "reciprocal rank fusion candidate pool",
      limit: 5,
    };
    const oneShot = await searchMemory(input);
    const pool = await fuseCandidates(input);
    expect(pool.length).toBeGreaterThan(0);
    const composed = await rerankAndBlend(input.query, pool, input.limit);
    // The live handler is exactly this composition plus the display-only snippet
    // decoration pass (AC-801) — which never adds, drops, or reorders hits.
    const decorated = await applyQuerySnippets(
      input.query,
      composed.map(({ score: _score, ...hit }) => hit),
    );
    expect(decorated).toEqual(oneShot.hits);
    // Ranking-bearing fields are identical even before decoration.
    expect(composed.map((h) => h.id)).toEqual(oneShot.hits.map((h) => h.id));
  });

  // ── AC-102: fuseCandidates pool contract ───────────────────────────────────

  it("AC-102: pool caps at config.candidatePool rows with per-arm RRF ranks", async () => {
    const pool = await fuseCandidates({
      projectId: P_POOL,
      query: "shared retrieval conformance token",
    });
    expect(pool.length).toBe(retrievalConfig().candidatePool);
    for (const c of pool) {
      expect(typeof c.id).toBe("string");
      expect("bm25_rank" in c).toBe(true);
      expect("vec_rank" in c).toBe(true);
      expect(c.bm25_rank !== null || c.vec_rank !== null).toBe(true);
      expect(typeof c.rrf).toBe("number");
    }
    // RRF-ordered, non-increasing.
    for (let i = 1; i < pool.length; i++) {
      expect(pool[i - 1].rrf).toBeGreaterThanOrEqual(pool[i].rrf);
    }
  });

  it("AC-102: the live handler consumes exactly this pool (finals ⊆ pool ids)", async () => {
    const query = "shared retrieval conformance token";
    const pool = await fuseCandidates({ projectId: P_POOL, query });
    const poolIds = new Set(pool.map((c) => c.id));
    const result = await searchMemory({ projectId: P_POOL, query, limit: 10 });
    const { hits } = result;
    expect(hits.length).toBe(10);
    for (const h of hits) expect(poolIds.has(h.id)).toBe(true);
  });

  // ── AC-106: supersession forward-chain resolution ──────────────────────────

  it("AC-106: resolveGoldIds walks the full superseded_by forward chain", async () => {
    const chains = await resolveGoldIds(db, [CHAIN_A, CHAIN_B]);
    expect(chains.get(CHAIN_A)).toEqual(new Set([CHAIN_A, CHAIN_B, CHAIN_C]));
    expect(chains.get(CHAIN_B)).toEqual(new Set([CHAIN_B, CHAIN_C]));
  });

  it("AC-106: supersession cycles terminate (path-array guard)", async () => {
    const chains = await resolveGoldIds(db, [CYCLE_X]);
    expect(chains.get(CYCLE_X)).toEqual(new Set([CYCLE_X, CYCLE_Y]));
  });

  it("AC-106: ids absent from the DB resolve to themselves", async () => {
    const ghost = mid(999);
    const chains = await resolveGoldIds(db, [ghost]);
    expect(chains.get(ghost)).toEqual(new Set([ghost]));
  });

  // ── AC-104 (+ AC-106 e2e): eval run artifact — both layers + config ────────

  it("AC-104/AC-106: runRecallEval artifact carries the config snapshot + two-layer per-query scores; superseded gold credits its successor", async () => {
    const evalFile: RecallEvalFileV2 = {
      version: 2,
      k: 10,
      changelog: ["conformance-w13 synthetic gold (never committed)"],
      rows: [
        {
          id: "g1",
          query: "graceful shutdown drain sequence",
          relevantIds: [RUN_GOOD],
          split: "dev",
          provenance: "seed-v1",
        },
        {
          id: "g2",
          query: "voyage context corpus embedder migration",
          relevantIds: [RUN_SUPER_A], // superseded — only its successor is findable
          split: "dev",
          provenance: "seed-v1",
        },
      ],
    };
    const artifact = await runRecallEval(evalFile, {
      projectId: P_RUN,
      split: "dev",
    });

    // Full retrievalConfig() snapshot, verbatim.
    expect(artifact.retrievalConfig).toEqual(retrievalConfig());
    expect(artifact.evalVersion).toBe(2);
    expect(artifact.split).toBe("dev");
    expect(artifact.rows).toBe(2);
    expect(artifact.perQuery.map((q) => q.id)).toEqual(["g1", "g2"]);
    for (const q of artifact.perQuery) {
      expect(typeof q.poolRecall).toBe("number"); // pool layer (Recall@25)
      expect(typeof q.ndcg).toBe("number"); // final layer (nDCG@10)
      expect(typeof q.mrr).toBe("number");
    }
    expect(artifact.aggregates).toEqual({
      recallAt25: expect.any(Number),
      ndcgAt10: expect.any(Number),
      mrr10: expect.any(Number),
    });

    const g1 = artifact.perQuery.find((q) => q.id === "g1")!;
    expect(g1.poolRecall).toBe(1);
    expect(g1.ndcg).toBeGreaterThan(0);

    // AC-106 end-to-end: the superseded gold id scores via its successor.
    const g2 = artifact.perQuery.find((q) => q.id === "g2")!;
    expect(g2.poolRecall).toBe(1);
    expect(g2.ndcg).toBeGreaterThan(0);

    // Artifact writer: JSON file with the snapshot embedded.
    const tmp = mkdtempSync(join(tmpdir(), "conf-w13-runs-"));
    try {
      const path = writeRunArtifact(tmp, "conf-w13", artifact);
      expect(existsSync(path)).toBe(true);
      const parsed = JSON.parse(readFileSync(path, "utf8")) as RunArtifact;
      expect(parsed.retrievalConfig).toEqual(retrievalConfig());
      expect(parsed.perQuery).toHaveLength(2);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  // ── AC-107: fire-and-forget search_log ─────────────────────────────────────

  it("AC-107: a search eventually lands one search_log row with matching ids", async () => {
    const query = "conf-w13 log landing probe backup restore";
    const result = await searchMemory({
      projectId: P_SEARCH,
      query,
      limit: 3,
    });
    const { hits } = result;
    const landed = await until(async () => {
      const { rows } = await db.query(
        `SELECT pool_ids, final_ids, filters FROM memory.search_log
         WHERE project_id = $1 AND query = $2`,
        [P_SEARCH, query],
      );
      return rows.length === 1;
    });
    expect(landed).toBe(true);
    const { rows } = await db.query<{
      pool_ids: string[];
      final_ids: string[];
      filters: Record<string, unknown>;
    }>(
      `SELECT pool_ids, final_ids, filters FROM memory.search_log
       WHERE project_id = $1 AND query = $2`,
      [P_SEARCH, query],
    );
    expect(rows[0].final_ids).toEqual(hits.map((h) => h.id));
    for (const h of hits) expect(rows[0].pool_ids).toContain(h.id);
    expect(rows[0].filters).toEqual({});
  });

  it("AC-107: a blocked log insert does NOT block searchMemory (no added latency)", async () => {
    const locker = await db.connect();
    try {
      await locker.query("BEGIN");
      await locker.query(
        "LOCK TABLE memory.search_log IN ACCESS EXCLUSIVE MODE",
      );
      const query = "conf-w13 lock probe hnsw rebuild guidance";
      const result = await Promise.race([
        searchMemory({ projectId: P_SEARCH, query, limit: 3 }).then((r) => ({
          resolved: true as const,
          hits: r.hits,
        })),
        new Promise<{ resolved: false }>((r) =>
          setTimeout(() => r({ resolved: false }), 4000),
        ),
      ]);
      // If the implementation awaited the insert, this would time out.
      expect(result.resolved).toBe(true);
      if (result.resolved) expect(result.hits.length).toBeGreaterThan(0);
      await locker.query("ROLLBACK");
      // The deferred insert must still land once the lock releases.
      const landed = await until(async () => {
        const { rows } = await db.query(
          `SELECT 1 FROM memory.search_log WHERE project_id = $1 AND query = $2`,
          [P_SEARCH, query],
        );
        return rows.length === 1;
      });
      expect(landed).toBe(true);
    } finally {
      locker.release();
    }
  });

  it("AC-107: a FAILING log insert changes nothing — same results, no error", async () => {
    const input = {
      projectId: P_SEARCH,
      query: "rollback deployment health checks",
      limit: 5,
    };
    const baseline = await searchMemory(input);
    await db.query(
      "ALTER TABLE memory.search_log RENAME TO search_log_confw13bak",
    );
    try {
      const broken = await searchMemory(input);
      expect(broken.hits).toEqual(baseline.hits);
    } finally {
      await db.query(
        "ALTER TABLE memory.search_log_confw13bak RENAME TO search_log",
      );
    }
    // Give the swallowed rejection a tick so nothing leaks into later tests.
    await new Promise((r) => setTimeout(r, 50));
  });
});
