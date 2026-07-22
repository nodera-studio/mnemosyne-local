// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Wave 5 (memory consolidation)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/
// wave-5-consolidation.md, AC-301…AC-306 + the 2026-07-03 fix-pass guarantees).
// Every assertion below was drafted from the plan text BEFORE locating the
// exported symbols; this suite is the independent anti-reward-hacking gate,
// additive to the Implementer's own tests.
//
// AC map:
//   AC-301 — storeMemory sha256 exact-dup short-circuit: ONE row, duplicate:true,
//            ZERO embed calls on the dup path, project-scoped, importer hash
//            recipe (sha256 hex over content only), superseded rows never block.
//   AC-302 — top-s cosine dup-candidate flag (>0.90) on the NEW row's metadata;
//            sub-threshold neighbors not recorded; NO LLM import on the write path.
//   AC-303 — consolidate dry-run DEFAULT: zero row changes + report artifact;
//            --apply flips ONLY status/superseded_by (content byte-identical);
//            pinned never loses; decision rows never enter; report = undo log.
//   AC-305 — fail-open: transport throw → WHOLE batch UNJUDGED and re-enters
//            (no judged-marker); wholly-unparseable JSON → batch UNJUDGED, run
//            continues; per-pair schema violation (unknown int / missing keep /
//            pinned-loser keep) → ONLY that pair UNJUDGED, siblings stand.
//   AC-306 — the self-join sweep is the completeness source of truth: a direct-SQL
//            importer-style row (no content_sha256, no dupCandidates) is discovered
//            and CAN lose.
//   AC-106 — interplay: a superseded gold row resolves forward through the chain
//            consolidation extends; searchMemory/listMemories exclude losers.
//   Fix-pass (binding, plan-level):
//     • report file is the resumability checkpoint (complete:false during a run,
//       complete:true at the end; priorReport never re-judges judged pairs);
//     • resolveCliAction: --apply never judge-and-applies fresh, never overwrites
//       an apply record with a dry-run, re-applies an apply record idempotently;
//     • decision-kind stores never dedup-short-circuit (duplicate content +
//       supersedesId still inserts and flips the superseded decision);
//     • updateMemory content edits clear metadata.dupCandidates/dupFlaggedAt.
//
// Deterministic throughout: the judge is ALWAYS an injected mock; Voyage is
// module-mocked (seeded PRNG vectors); zero network calls anywhere (asserted).
// DB tests are self-contained under conf-w5-* project ids and clean up.
//
// Run:
//   npx tsc -p tsconfig.test.json --noEmit
//   DATABASE_URL=postgresql://postgres:postgres@localhost:5544/postgres \
//     npx vitest run conformance-w5
// ─────────────────────────────────────────────────────────────────────────────

import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic vector helpers, hoisted so the vi.mock factory can use them.
// Contents may carry a marker `[vec:<cluster>:<eps>:<salt>]`; the mocked embedder
// returns base(cluster) + eps·noise(salt), so cosine similarity between two rows
// of the same cluster is controlled: eps 0 vs 0.2 → ~0.98 (> 0.90 threshold),
// eps 0 vs 1.0 → ~0.70 (< 0.90). Rows without a marker get independent PRNG
// vectors (pairwise cosine ~0).
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
  function nearVec(cluster: string, eps: number, salt: string): number[] {
    const base = fakeVec(`base:${cluster}`);
    if (eps === 0) return base;
    const noise = fakeVec(`noise:${cluster}:${salt}`);
    return base.map((v, i) => Number((v + eps * noise[i]).toFixed(3)));
  }
  const MARKER = /\[vec:([a-z0-9]+):([0-9.]+):([a-z0-9]+)\]/;
  function vecFor(text: string): number[] {
    const m = MARKER.exec(text);
    return m ? nearVec(m[1], Number(m[2]), m[3]) : fakeVec(`t:${text}`);
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
  // Embed-call counters — AC-301 asserts ZERO embed spend on the dup path.
  const counters = { embed: 0, embedContextual: 0, embedContextualSingle: 0 };
  return { fnv1a, mulberry32, fakeVec, nearVec, vecFor, fakeRerank, counters };
});

// HARD RULE: never call live Voyage — module-mock the boundary (with counters).
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => {
    H.counters.embed++;
    return texts.map(H.vecFor);
  },
  embedContextual: async (docs: unknown[]) => {
    H.counters.embedContextual++;
    return docs.map((d) =>
      Array.isArray(d) ? (d as string[]).map(H.vecFor) : H.vecFor(d as string),
    );
  },
  embedContextualSingle: async (texts: string[]) => {
    H.counters.embedContextualSingle++;
    return texts.map(H.vecFor);
  },
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";
process.env.ANTHROPIC_API_KEY ??= "test-key";

import {
  applyFromReport,
  consolidate,
  guardConsolidateRun,
  resolveCliAction,
  type ConsolidateReport,
  type JudgedPair,
} from "../src/db/consolidate.js";
import {
  DUP_COSINE_THRESHOLD,
  listMemories,
  searchMemory,
  storeMemory,
  updateMemory,
} from "../src/memory.js";
import { resolveGoldIds } from "./recall.helper.js";

const here = dirname(fileURLToPath(import.meta.url));
const serviceRoot = join(here, "..");

const totalEmbedCalls = () =>
  H.counters.embed +
  H.counters.embedContextual +
  H.counters.embedContextualSingle;

// Unique fixture namespace — never collides with other suites' project ids.
const P_DEDUP = "conf-w5-dedup";
const P_DEDUP2 = "conf-w5-dedup-other";
const P_FLAG = "conf-w5-flag";
const P_DEC = "conf-w5-decision";
const P_CONS = "conf-w5-consolidate";
const P_FAIL = "conf-w5-failopen";
const P_MIX = "conf-w5-pairlevel";
const ALL_PROJECTS = [P_DEDUP, P_DEDUP2, P_FLAG, P_DEC, P_CONS, P_FAIL, P_MIX];

const wid = (n: number) =>
  `50000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// ── P_CONS corpus ids (direct-SQL seeds; N2 arrives via storeMemory) ─────────
const N1 = wid(1); //   near-dup pair, older  → the loser
const M1 = wid(3); //   mid pair (~0.70)      → never a candidate
const M2 = wid(4);
const PIN1 = wid(5); // pinned, older         → may compare, may NEVER lose
const PIN2 = wid(6);
const IMP = wid(7); //  importer-style, older → no sha/no flags; CAN lose (AC-306)
const IMP2 = wid(8);
const D1C = wid(9); //  decision-kind         → excluded from candidates entirely
const D2C = wid(10);
const A0 = wid(11); //  pre-superseded → N1 (AC-106 chain head)

// ── P_FAIL corpus ─────────────────────────────────────────────────────────────
const F1 = wid(21);
const F2 = wid(22);
const F3 = wid(23);
const F4 = wid(24);

// ── P_MIX corpus ──────────────────────────────────────────────────────────────
const V1A = wid(31); // pair judged with unknown integer verdict → UNJUDGED
const V1B = wid(32);
const V2A = wid(33); // pair judged verdict 1 but missing keep   → UNJUDGED
const V2B = wid(34);
const V3A = wid(35); // valid sibling pair                        → applies
const V3B = wid(36);

// Unique per-row prompt tokens: the injected judge maps numbered pairs back to
// fixtures by scanning the user prompt for these (pairs are NUMBERED 1..N in
// presentation order per the plan, and each pair carries both rows' contents).
const TOK = {
  n1: "tokqnonex",
  n2: "tokqntwox",
  pin1: "tokqponex",
  pin2: "tokqptwox",
  imp1: "tokqionex",
  imp2: "tokqitwox",
  m1: "tokqmonex",
  m2: "tokqmtwox",
  f1: "tokqfonex",
  f2: "tokqftwox",
  f3: "tokqfthreex",
  f4: "tokqffourx",
  v1a: "tokqvonex",
  v1b: "tokqvtwox",
  v2a: "tokqwonex",
  v2b: "tokqwtwox",
  v3a: "tokqxonex",
  v3b: "tokqxtwox",
  d1: "tokqdonex",
  d2: "tokqdtwox",
} as const;
const ALL_TOKENS = Object.values(TOK);

/** Group the known tokens found in a judge prompt into presentation-ordered
 *  pairs (each numbered pair contributes its two rows' contents in order). */
function promptPairGroups(user: string): string[][] {
  const found = ALL_TOKENS.map((t) => ({ t, i: user.indexOf(t) }))
    .filter((x) => x.i >= 0)
    .sort((x, y) => x.i - y.i);
  const groups: string[][] = [];
  for (let k = 0; k < found.length; k += 2) {
    groups.push(found.slice(k, k + 2).map((x) => x.t));
  }
  return groups;
}

/** Build an injected judge that answers strict numbered JSON derived from the
 *  planted tokens — zero network, deterministic, order-independent. */
function makeJudge(
  decide: (group: string[]) => Record<string, unknown>,
  onCall?: (priorCalls: number) => void,
): {
  judge: (system: string, user: string) => Promise<string>;
  calls: string[];
} {
  const calls: string[] = [];
  const judge = async (_system: string, user: string): Promise<string> => {
    onCall?.(calls.length);
    calls.push(user);
    const out: Record<string, unknown> = {};
    promptPairGroups(user).forEach((g, i) => {
      out[String(i + 1)] = decide(g);
    });
    return JSON.stringify(out);
  };
  return { judge, calls };
}

function cliReport(
  over: Partial<ConsolidateReport["config"]> = {},
): ConsolidateReport {
  return {
    config: {
      threshold: 0.9,
      model: "mock-judge",
      batch: 500,
      judgeBatch: 8,
      topS: 3,
      cursor: null,
      complete: true,
      apply: false,
      projectId: "conf-w5-cli",
      generatedAt: new Date().toISOString(),
      ...over,
    },
    pairs: [],
    judged: 0,
    unjudged: 0,
    wouldSupersede: 0,
    applied: 0,
  };
}

// ═════════════════════════════ PURE (no DB) ══════════════════════════════════

describe("W5 pure — resolveCliAction guarantees (fix-pass)", () => {
  it("--apply with NO report refuses (never judge-and-apply fresh)", () => {
    const action = resolveCliAction({ apply: true, existing: null });
    expect(action.kind).toBe("refuse");
  });

  it("--apply with an INCOMPLETE report refuses (not a reviewed complete dry-run)", () => {
    const action = resolveCliAction({
      apply: true,
      existing: cliReport({ complete: false, cursor: wid(99) }),
    });
    expect(action.kind).toBe("refuse");
  });

  it("--apply with a complete same-day dry-run report applies FROM the report (no fresh judging)", () => {
    const existing = cliReport();
    const action = resolveCliAction({ apply: true, existing });
    expect(action.kind).toBe("applyReport");
    if (action.kind === "applyReport") expect(action.report).toBe(existing);
  });

  it("--apply with an explicit report applies FROM that report", () => {
    const explicitReport = cliReport();
    const action = resolveCliAction({
      apply: true,
      existing: null,
      explicitReport,
    });
    expect(action.kind).toBe("applyReport");
    if (action.kind === "applyReport")
      expect(action.report).toBe(explicitReport);
  });

  it("a dry-run over an existing APPLY record refuses (never overwrite the undo log)", () => {
    const action = resolveCliAction({
      apply: false,
      existing: cliReport({ apply: true }),
    });
    expect(action.kind).toBe("refuse");
  });

  it("--apply over an existing APPLY record re-applies it idempotently", () => {
    const existing = cliReport({ apply: true });
    const action = resolveCliAction({ apply: true, existing });
    expect(action.kind).toBe("applyReport");
    if (action.kind === "applyReport") expect(action.report).toBe(existing);
  });

  it("no flags + no report → a fresh DRY-RUN (dry-run is the default)", () => {
    const action = resolveCliAction({ apply: false, existing: null });
    expect(action.kind).toBe("run");
    if (action.kind === "run") {
      expect(action.apply).toBe(false);
      expect(action.priorReport).toBeUndefined();
    }
  });

  it("a dry-run over an INCOMPLETE report resumes it (never re-buys judged pairs)", () => {
    const existing = cliReport({ complete: false, cursor: wid(98) });
    const action = resolveCliAction({ apply: false, existing });
    expect(action.kind).toBe("run");
    if (action.kind === "run") {
      expect(action.apply).toBe(false);
      expect(action.priorReport).toBe(existing);
    }
  });
});

describe("W5 pure — PAID gate + write-path hygiene", () => {
  it("guardConsolidateRun blocks without --yes and passes with it (PAID gate)", () => {
    const refusal = guardConsolidateRun([]);
    expect(refusal).toBeTruthy();
    expect(refusal).toMatch(/--yes/);
    expect(guardConsolidateRun(["--apply"])).toBeTruthy();
    expect(guardConsolidateRun(["--yes"])).toBeNull();
    expect(guardConsolidateRun(["--yes", "--apply"])).toBeNull();
  });

  it("package.json exposes the consolidate operator script; the script main-guards", () => {
    const pkg = JSON.parse(
      readFileSync(join(serviceRoot, "package.json"), "utf8"),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts["consolidate"]).toBeTruthy();
    const src = readFileSync(
      join(serviceRoot, "src", "db", "consolidate.ts"),
      "utf8",
    );
    expect(src).toMatch(/import\.meta\.url/);
  });

  it("AC-302: the write path (memory.ts) never imports the LLM judge client", () => {
    const src = readFileSync(join(serviceRoot, "src", "memory.ts"), "utf8");
    expect(src).not.toMatch(/from\s+["'][^"']*llm(\.js)?["']/);
  });

  it("the dup threshold constant is exported at 0.90 (shared with the batch job)", () => {
    expect(DUP_COSINE_THRESHOLD).toBe(0.9);
  });
});

// ═════════════════════════════ DB-BACKED ═════════════════════════════════════

describe.skipIf(skip)("W5 conformance — DB-backed", () => {
  let db: pg.Pool;
  let tmp: string;
  const fetchCalls: unknown[][] = [];
  let origFetch: typeof fetch;

  // Structural pool dep (dodges pg overload variance in strict mode).
  const poolDep = {
    query: (text: string, params?: unknown[]) =>
      db.query(text, params as unknown[] | undefined),
  };

  async function seedRow(r: {
    id: string;
    projectId: string;
    title: string;
    content: string;
    createdAt: string;
    vec?: number[] | null;
    pinned?: boolean;
    sourceKind?: string | null;
    decisionStatus?: string | null;
    status?: string;
    supersededBy?: string | null;
    sha?: string | null;
    metadata?: Record<string, unknown>;
  }): Promise<void> {
    await db.query(
      `INSERT INTO memory.memories
         (id, project_id, type, title, content, importance, created_at, pinned,
          source_kind, decision_status, status, superseded_by, content_sha256,
          metadata, embedding_v2)
       VALUES ($1,$2,'semantic',$3,$4,0.5,$5,$6,$7,$8,$9,$10,$11,$12,$13::halfvec)`,
      [
        r.id,
        r.projectId,
        r.title,
        r.content,
        r.createdAt,
        r.pinned ?? false,
        r.sourceKind ?? null,
        r.decisionStatus ?? null,
        r.status ?? "active",
        r.supersededBy ?? null,
        r.sha ?? null,
        JSON.stringify(r.metadata ?? {}),
        r.vec ? `[${r.vec.join(",")}]` : null,
      ],
    );
  }

  interface SnapRow {
    id: string;
    title: string;
    content: string;
    importance: number;
    type: string;
    status: string;
    superseded_by: string | null;
    pinned: boolean;
    source_kind: string | null;
    content_sha256: string | null;
    decision_status: string | null;
    metadata: Record<string, unknown>;
    created_at: Date;
  }

  async function snapshot(projectId: string): Promise<SnapRow[]> {
    const { rows } = await db.query<SnapRow>(
      `SELECT id, title, content, importance, type::text AS type, status,
              superseded_by, pinned, source_kind, content_sha256,
              decision_status, metadata, created_at
       FROM memory.memories WHERE project_id = $1 ORDER BY id`,
      [projectId],
    );
    return rows;
  }

  const rowOf = (snap: SnapRow[], id: string): SnapRow => {
    const r = snap.find((x) => x.id === id);
    expect(r, `row ${id} must exist`).toBeTruthy();
    return r!;
  };

  async function cleanup(): Promise<void> {
    await db.query(
      `UPDATE memory.memories SET superseded_by = NULL, supersedes_id = NULL
       WHERE project_id = ANY($1)`,
      [ALL_PROJECTS],
    );
    await db.query(`DELETE FROM memory.memories WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
    await db.query(`DELETE FROM memory.search_log WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Idempotent migration replay (HOLD files skipped) — self-sufficient on a fresh DB.
    const sqlDir = join(serviceRoot, "sql");
    const { readdirSync } = await import("node:fs");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();
    tmp = mkdtempSync(join(tmpdir(), "conf-w5-"));
    // Zero-network tripwire for the whole DB block (judge is injected; Voyage mocked).
    origFetch = globalThis.fetch;
    globalThis.fetch = (async (...args: unknown[]) => {
      fetchCalls.push(args);
      throw new Error("conformance-w5: network call escaped the mocks");
    }) as typeof fetch;
  }, 120_000);

  afterAll(async () => {
    globalThis.fetch = origFetch;
    await cleanup();
    await db.end();
    rmSync(tmp, { recursive: true, force: true });
  });

  // ── AC-301: sha256 exact-dup short-circuit ─────────────────────────────────

  describe("AC-301 — write-path exact dedup", () => {
    const CONTENT =
      "conformance dedup probe: cap connection pools at ten per service";

    it("storing identical content twice → ONE row, duplicate:true, zero embed calls, importer hash recipe", async () => {
      const first = await storeMemory({
        projectId: P_DEDUP,
        type: "semantic",
        title: "Dedup probe",
        content: CONTENT,
      });
      expect(first.duplicate).not.toBe(true);

      const embedsBefore = totalEmbedCalls();
      const second = await storeMemory({
        projectId: P_DEDUP,
        type: "semantic",
        title: "Dedup probe RETITLED", // title is NOT part of the dedup key
        content: CONTENT,
      });
      expect(second.duplicate).toBe(true);
      expect(second.id).toBe(first.id);
      // The dup path spends ZERO embed quota.
      expect(totalEmbedCalls()).toBe(embedsBefore);

      const { rows } = await db.query<{ id: string; content_sha256: string }>(
        `SELECT id, content_sha256 FROM memory.memories WHERE project_id = $1`,
        [P_DEDUP],
      );
      expect(rows).toHaveLength(1);
      // Importer recipe cross-check: sha256 hex over content ONLY.
      expect(rows[0].content_sha256).toBe(
        createHash("sha256").update(CONTENT).digest("hex"),
      );
    }, 30_000);

    it("dedup is project-scoped: the same content in another project inserts fresh", async () => {
      const other = await storeMemory({
        projectId: P_DEDUP2,
        type: "semantic",
        title: "Dedup probe other project",
        content: CONTENT,
      });
      expect(other.duplicate).not.toBe(true);
      const { rows } = await db.query(
        `SELECT id FROM memory.memories WHERE project_id = $1`,
        [P_DEDUP2],
      );
      expect(rows).toHaveLength(1);
    }, 30_000);

    it("a superseded row with the same hash does NOT block a re-store", async () => {
      await db.query(
        `UPDATE memory.memories SET status = 'superseded' WHERE project_id = $1`,
        [P_DEDUP],
      );
      const restored = await storeMemory({
        projectId: P_DEDUP,
        type: "semantic",
        title: "Dedup probe re-store",
        content: CONTENT,
      });
      expect(restored.duplicate).not.toBe(true);
      const { rows } = await db.query(
        `SELECT id FROM memory.memories WHERE project_id = $1`,
        [P_DEDUP],
      );
      expect(rows).toHaveLength(2);
    }, 30_000);
  });

  // ── AC-302: cosine dup-candidate flag (no LLM) ─────────────────────────────

  describe("AC-302 — write-path dup-candidate flag + fix-pass clearing", () => {
    let aId: string;
    let bId: string;

    it("a >0.90 neighbor is flagged into the NEW row's metadata; sub-threshold is not", async () => {
      const a = await storeMemory({
        projectId: P_FLAG,
        type: "semantic",
        title: "Flag anchor",
        content: "retention anchor claim [vec:c30:0:fa]",
      });
      aId = a.id;
      const b = await storeMemory({
        projectId: P_FLAG,
        type: "semantic",
        title: "Flag near dup",
        content: "retention anchor claim restated [vec:c30:0.2:fb]",
      });
      bId = b.id;
      const { rows: bRows } = await db.query<{
        metadata: {
          dupCandidates?: Array<{ id: string; sim: number }>;
          dupFlaggedAt?: string;
        };
      }>(`SELECT metadata FROM memory.memories WHERE id = $1`, [bId]);
      const bMeta = bRows[0].metadata;
      expect(Array.isArray(bMeta.dupCandidates)).toBe(true);
      const flagged = bMeta.dupCandidates!.find((c) => c.id === aId);
      expect(flagged).toBeTruthy();
      expect(flagged!.sim).toBeGreaterThan(0.9);
      expect(flagged!.sim).toBeLessThanOrEqual(1);
      expect(typeof bMeta.dupFlaggedAt).toBe("string");
      expect(Number.isNaN(Date.parse(bMeta.dupFlaggedAt!))).toBe(false);

      // Sub-threshold (~0.70) neighbor is NOT recorded.
      const c = await storeMemory({
        projectId: P_FLAG,
        type: "semantic",
        title: "Flag mid similarity",
        content: "retention adjacent topic [vec:c30:1:fc]",
      });
      const { rows: cRows } = await db.query<{
        metadata: { dupCandidates?: unknown };
      }>(`SELECT metadata FROM memory.memories WHERE id = $1`, [c.id]);
      expect(cRows[0].metadata.dupCandidates).toBeUndefined();
    }, 30_000);

    it("fix-pass: a CONTENT edit clears metadata.dupCandidates/dupFlaggedAt", async () => {
      const updated = await updateMemory(bId, {
        content: "entirely different assertion after the edit",
      });
      expect(updated).not.toBeNull();
      const { rows } = await db.query<{
        metadata: { dupCandidates?: unknown; dupFlaggedAt?: unknown };
      }>(`SELECT metadata FROM memory.memories WHERE id = $1`, [bId]);
      expect(rows[0].metadata.dupCandidates).toBeUndefined();
      expect(rows[0].metadata.dupFlaggedAt).toBeUndefined();
    }, 30_000);
  });

  // ── Fix-pass: decision-kind stores never dedup-short-circuit ───────────────

  describe("fix-pass — decision stores bypass the exact-dup short-circuit", () => {
    it("duplicate content + supersedesId still inserts and flips the superseded decision", async () => {
      const DEC_CONTENT = "we adopt drizzle for the memory service data layer";
      const d1 = await storeMemory({
        projectId: P_DEC,
        type: "semantic",
        title: "Decision v1",
        content: DEC_CONTENT,
        sourceKind: "decision",
        decisionStatus: "active",
      });
      expect(d1.duplicate).not.toBe(true);

      const d2 = await storeMemory({
        projectId: P_DEC,
        type: "semantic",
        title: "Decision v2",
        content: DEC_CONTENT, // byte-identical — would dedup for any other kind
        sourceKind: "decision",
        decisionStatus: "active",
        supersedesId: d1.id,
      });
      expect(d2.duplicate).not.toBe(true);
      expect(d2.id).not.toBe(d1.id);

      const { rows } = await db.query<{
        id: string;
        decision_status: string | null;
        superseded_by: string | null;
      }>(
        `SELECT id, decision_status, superseded_by FROM memory.memories
         WHERE project_id = $1 ORDER BY created_at`,
        [P_DEC],
      );
      expect(rows).toHaveLength(2);
      const old = rows.find((r) => r.id === d1.id)!;
      expect(old.decision_status).toBe("superseded");
      expect(old.superseded_by).toBe(d2.id);
    }, 30_000);
  });

  // ── AC-303 / AC-306 / AC-106 + resume/apply fix-pass ───────────────────────

  describe("AC-303/305/306/106 — the consolidate pipeline", () => {
    let n2Id: string; // cluster-1 winner (stored via storeMemory → carries flags)
    let R1: ConsolidateReport; // dry-run 1 (pinned pair UNJUDGED)
    let R2: ConsolidateReport; // resumed dry-run (all pairs judged)
    const r1Path = () => join(tmp, "r1.json");

    const pairWith = (rep: ConsolidateReport, id: string): JudgedPair => {
      const p = rep.pairs.find((x) => x.a === id || x.b === id);
      expect(p, `pair containing ${id} must exist in the report`).toBeTruthy();
      return p!;
    };

    beforeAll(async () => {
      await seedRow({
        id: N1,
        projectId: P_CONS,
        title: "Pool ceiling claim",
        content: `shared retrieval consolidation pool ceiling claim ${TOK.n1} [vec:c10:0:n1]`,
        createdAt: "2026-06-01T00:00:00Z",
        vec: H.nearVec("c10", 0, "n1"),
      });
      await seedRow({
        id: M1,
        projectId: P_CONS,
        title: "Midband warmup",
        content: `midband cache warmup interval ${TOK.m1} [vec:c11:0:m1]`,
        createdAt: "2026-06-02T00:00:00Z",
        vec: H.nearVec("c11", 0, "m1"),
      });
      await seedRow({
        id: M2,
        projectId: P_CONS,
        title: "Midband cadence",
        content: `midband cache warmup cadence ${TOK.m2} [vec:c11:1:m2]`,
        createdAt: "2026-06-03T00:00:00Z",
        vec: H.nearVec("c11", 1, "m2"),
      });
      await seedRow({
        id: PIN1,
        projectId: P_CONS,
        title: "Pinned retention baseline",
        content: `pinned retention baseline policy ${TOK.pin1} [vec:c12:0:p1]`,
        createdAt: "2026-06-01T00:00:00Z",
        pinned: true,
        vec: H.nearVec("c12", 0, "p1"),
      });
      await seedRow({
        id: PIN2,
        projectId: P_CONS,
        title: "Pinned retention refined",
        content: `pinned retention baseline policy refined ${TOK.pin2} [vec:c12:0.2:p2]`,
        createdAt: "2026-06-05T00:00:00Z",
        vec: H.nearVec("c12", 0.2, "p2"),
      });
      // AC-306: importer-style rows — direct SQL, NO content_sha256, NO flags.
      await seedRow({
        id: IMP,
        projectId: P_CONS,
        title: "Importer payload",
        content: `importer sweep discovery payload ${TOK.imp1} [vec:c13:0:i1]`,
        createdAt: "2026-06-01T00:00:00Z",
        vec: H.nearVec("c13", 0, "i1"),
      });
      await seedRow({
        id: IMP2,
        projectId: P_CONS,
        title: "Importer payload enriched",
        content: `importer sweep discovery payload enriched ${TOK.imp2} [vec:c13:0.2:i2]`,
        createdAt: "2026-06-05T00:00:00Z",
        vec: H.nearVec("c13", 0.2, "i2"),
      });
      await seedRow({
        id: D1C,
        projectId: P_CONS,
        title: "Decision log ordering",
        content: `decision log ordering guarantee ${TOK.d1} [vec:c14:0:d1]`,
        createdAt: "2026-06-01T00:00:00Z",
        sourceKind: "decision",
        decisionStatus: "active",
        vec: H.nearVec("c14", 0, "d1"),
      });
      await seedRow({
        id: D2C,
        projectId: P_CONS,
        title: "Decision log ordering restated",
        content: `decision log ordering guarantee restated ${TOK.d2} [vec:c14:0.2:d2]`,
        createdAt: "2026-06-05T00:00:00Z",
        sourceKind: "decision",
        decisionStatus: "active",
        vec: H.nearVec("c14", 0.2, "d2"),
      });
      // AC-106 chain head: A0 was superseded by N1 BEFORE consolidation ran.
      await seedRow({
        id: A0,
        projectId: P_CONS,
        title: "Ancient pool claim",
        content: "ancient pool ceiling placeholder",
        createdAt: "2026-05-01T00:00:00Z",
        status: "superseded",
        supersededBy: N1,
        vec: null,
      });
      // N2 arrives through the real write path (newer + richer → the winner).
      const n2 = await storeMemory({
        projectId: P_CONS,
        type: "semantic",
        title: "Pool ceiling claim (richer)",
        content: `shared retrieval consolidation pool ceiling claim with retry ceiling detail ${TOK.n2} [vec:c10:0.2:n2]`,
      });
      n2Id = n2.id;
    }, 60_000);

    it("dry-run (DEFAULT): judges candidate pairs, writes the report, changes ZERO rows; the report file is a complete:false checkpoint during the run", async () => {
      const before = await snapshot(P_CONS);
      const checkpoints: Array<{ exists: boolean; complete: boolean | null }> =
        [];
      // judgeBatch:1 → one paid batch per pair; after batch 1's verdict is
      // bought, a crash-safe checkpoint MUST already exist on disk.
      const { judge } = makeJudge(
        () => ({ verdict: 1, keep: "b", reason: "same claim" }),
        (priorCalls) => {
          if (priorCalls === 0) return;
          if (!existsSync(r1Path())) {
            checkpoints.push({ exists: false, complete: null });
            return;
          }
          const parsed = JSON.parse(
            readFileSync(r1Path(), "utf8"),
          ) as ConsolidateReport;
          checkpoints.push({ exists: true, complete: parsed.config.complete });
        },
      );

      const { report, reportPath } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_CONS,
        reportPath: r1Path(),
        judgeBatch: 1,
        log: () => {},
      });
      R1 = report;

      // Zero row changes — the dry-run is the DEFAULT (no apply flag passed).
      expect(await snapshot(P_CONS)).toEqual(before);

      // Resumability checkpoint: mid-run rewrites carry complete:false.
      expect(checkpoints.length).toBeGreaterThanOrEqual(1);
      for (const c of checkpoints) {
        expect(c.exists).toBe(true);
        expect(c.complete).toBe(false);
      }

      // Completed run: complete:true, dry-run config, nothing applied.
      expect(report.config.complete).toBe(true);
      expect(report.config.apply).toBe(false);
      expect(report.config.projectId).toBe(P_CONS);
      expect(report.config.threshold).toBe(0.9);
      expect(report.applied).toBe(0);
      expect(reportPath).toBe(r1Path());

      // Exactly the three >0.90 clusters — mid pair (~0.70) absent, decision
      // rows excluded entirely.
      expect(report.pairs).toHaveLength(3);
      const touched = new Set(report.pairs.flatMap((p) => [p.a, p.b]));
      for (const banned of [M1, M2, D1C, D2C, A0]) {
        expect(touched.has(banned)).toBe(false);
      }
      for (const p of report.pairs) {
        expect(p.sim).toBeGreaterThan(0.9);
        expect([1, 0, -1]).toContain(p.verdict); // report verdict vocabulary
      }

      // Slot normalization: a is ALWAYS the older row.
      const nPair = pairWith(report, N1);
      expect(nPair.a).toBe(N1);
      expect(nPair.b).toBe(n2Id);
      expect(nPair.verdict).toBe(1);
      expect(nPair.keep).toBe("b");
      // Cluster 1's winner came through storeMemory → dupCandidates provenance.
      expect(["dupCandidates", "both"]).toContain(nPair.source);

      // AC-306: the pure importer-style pair is discovered by the self-join
      // sweep alone (neither row has flags or a sha).
      const impPair = pairWith(report, IMP);
      expect(impPair.a).toBe(IMP);
      expect(impPair.b).toBe(IMP2);
      expect(impPair.verdict).toBe(1);
      expect(impPair.source).toBe("selfJoin");

      // AC-305 pair-level violation: the judge's keep would make a PINNED row
      // lose → that pair is UNJUDGED while siblings stand.
      const pinPair = pairWith(report, PIN1);
      expect(pinPair.a).toBe(PIN1);
      expect(pinPair.verdict).toBe(-1);

      expect(report.judged).toBe(2);
      expect(report.unjudged).toBe(1);
      expect(report.wouldSupersede).toBe(2);

      // The artifact on disk IS the report.
      const onDisk = JSON.parse(readFileSync(r1Path(), "utf8"));
      expect(onDisk).toEqual(JSON.parse(JSON.stringify(report)));
    }, 60_000);

    it("resume with a prior report NEVER re-judges judged pairs; UNJUDGED pairs re-enter", async () => {
      const before = await snapshot(P_CONS);
      const { judge, calls } = makeJudge(() => ({ verdict: 0 }));
      const { report } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_CONS,
        reportPath: join(tmp, "r2.json"),
        priorReport: R1,
        log: () => {},
      });
      R2 = report;

      // Only the previously-UNJUDGED pinned pair is re-adjudicated.
      expect(calls).toHaveLength(1);
      expect(calls[0]).toContain(TOK.pin1);
      expect(calls[0]).toContain(TOK.pin2);
      for (const t of [TOK.n1, TOK.n2, TOK.imp1, TOK.imp2]) {
        expect(calls[0]).not.toContain(t);
      }

      expect(report.judged).toBe(3);
      expect(report.unjudged).toBe(0);
      expect(report.wouldSupersede).toBe(2);
      expect(report.config.complete).toBe(true);
      expect(pairWith(report, PIN1).verdict).toBe(0);
      expect(pairWith(report, N1).verdict).toBe(1);
      expect(pairWith(report, IMP).verdict).toBe(1);

      // Still a dry-run: zero row changes.
      expect(await snapshot(P_CONS)).toEqual(before);
    }, 60_000);

    it("apply flips ONLY status/superseded_by on losers — content byte-identical; pinned/decision/mid untouched", async () => {
      const before = await snapshot(P_CONS);
      const res = await applyFromReport(poolDep, R2, () => {}, P_CONS);
      expect(res.superseded).toBe(2);
      const after = await snapshot(P_CONS);

      // Losers: superseded + forward pointer, everything else byte-identical.
      for (const [loser, winner] of [
        [N1, n2Id],
        [IMP, IMP2],
      ] as const) {
        const b = rowOf(before, loser);
        const a = rowOf(after, loser);
        expect(a.status).toBe("superseded");
        expect(a.superseded_by).toBe(winner);
        expect(a.content).toBe(b.content);
        expect(a.title).toBe(b.title);
        expect(a.importance).toBe(b.importance);
        expect(a.type).toBe(b.type);
        expect(a.pinned).toBe(b.pinned);
        expect(a.created_at).toEqual(b.created_at);
      }

      // Winners: fully untouched.
      for (const winner of [n2Id, IMP2]) {
        expect(rowOf(after, winner)).toEqual(rowOf(before, winner));
      }

      // Pinned comparand: never loses (metadata may gain a judged-marker; the
      // row itself stays active and unedited).
      const pinAfter = rowOf(after, PIN1);
      expect(pinAfter.status).toBe("active");
      expect(pinAfter.superseded_by).toBeNull();
      expect(pinAfter.content).toBe(rowOf(before, PIN1).content);
      expect(rowOf(after, PIN2).status).toBe("active");

      // Mid pair, decision rows, and the pre-existing chain head: untouched.
      for (const id of [M1, M2, D1C, D2C, A0]) {
        expect(rowOf(after, id)).toEqual(rowOf(before, id));
      }
    }, 60_000);

    it("re-applying the same apply record is idempotent", async () => {
      const before = await snapshot(P_CONS);
      const res = await applyFromReport(poolDep, R2, () => {}, P_CONS);
      expect(res.superseded).toBe(0);
      expect(await snapshot(P_CONS)).toEqual(before);
    }, 60_000);

    it("a fresh run after apply makes ZERO judge calls (judged pairs are never re-bought)", async () => {
      const { judge, calls } = makeJudge(() => ({ verdict: 0 }));
      const { report } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_CONS,
        reportPath: join(tmp, "r3.json"),
        log: () => {},
      });
      expect(calls).toHaveLength(0);
      expect(report.judged).toBe(0);
      expect(report.unjudged).toBe(0);
    }, 60_000);

    it("AC-106: the extended chain resolves forward (A0 → N1 → N2) for gold credit", async () => {
      const chains = await resolveGoldIds(db, [A0]);
      expect(chains.get(A0)).toEqual(new Set([A0, N1, n2Id]));
    }, 30_000);

    it("searchMemory and listMemories exclude the superseded losers", async () => {
      const { hits } = await searchMemory({
        projectId: P_CONS,
        query: "shared retrieval consolidation pool ceiling claim",
        limit: 10,
      });
      expect(hits.length).toBeGreaterThan(0);
      const hitIds = hits.map((h) => h.id);
      expect(hitIds).toContain(n2Id); // the winner is findable
      expect(hitIds).not.toContain(N1); // the loser is not
      expect(hitIds).not.toContain(IMP);

      const listed = await listMemories({ projectId: P_CONS, limit: 50 });
      const listedIds = listed.rows.map((r) => r.id);
      expect(listedIds).toContain(n2Id);
      expect(listedIds).not.toContain(N1);
      expect(listedIds).not.toContain(IMP);
    }, 30_000);

    it("the report is the undo log: flipping losers back restores them", async () => {
      const losers = R2.pairs
        .filter((p) => p.verdict === 1)
        .map((p) => (p.keep === "a" ? p.b : p.a));
      expect(new Set(losers)).toEqual(new Set([N1, IMP]));
      for (const loser of losers) {
        await db.query(
          `UPDATE memory.memories SET status = 'active', superseded_by = NULL
           WHERE id = $1 AND status = 'superseded'`,
          [loser],
        );
      }
      const snap = await snapshot(P_CONS);
      for (const loser of [N1, IMP]) {
        expect(rowOf(snap, loser).status).toBe("active");
        expect(rowOf(snap, loser).superseded_by).toBeNull();
      }
    }, 30_000);
  });

  // ── AC-305: fail-open judge hardening ──────────────────────────────────────

  describe("AC-305 — fail-open (transport + unparseable)", () => {
    beforeAll(async () => {
      await seedRow({
        id: F1,
        projectId: P_FAIL,
        title: "Failover claim",
        content: `failover claim baseline ${TOK.f1} [vec:c20:0:f1]`,
        createdAt: "2026-06-01T00:00:00Z",
        vec: H.nearVec("c20", 0, "f1"),
      });
      await seedRow({
        id: F2,
        projectId: P_FAIL,
        title: "Failover claim restated",
        content: `failover claim baseline restated ${TOK.f2} [vec:c20:0.2:f2]`,
        createdAt: "2026-06-05T00:00:00Z",
        vec: H.nearVec("c20", 0.2, "f2"),
      });
      await seedRow({
        id: F3,
        projectId: P_FAIL,
        title: "GC pause claim",
        content: `gc pause tuning claim ${TOK.f3} [vec:c21:0:f3]`,
        createdAt: "2026-06-02T00:00:00Z",
        vec: H.nearVec("c21", 0, "f3"),
      });
      await seedRow({
        id: F4,
        projectId: P_FAIL,
        title: "GC pause claim restated",
        content: `gc pause tuning claim restated ${TOK.f4} [vec:c21:0.2:f4]`,
        createdAt: "2026-06-06T00:00:00Z",
        vec: H.nearVec("c21", 0.2, "f4"),
      });
    }, 60_000);

    it("transport failure → the WHOLE batch is UNJUDGED, zero rows change, the run still completes", async () => {
      const before = await snapshot(P_FAIL);
      const judge = async (): Promise<string> => {
        throw new Error("conformance-w5: judge transport down");
      };
      const { report } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_FAIL,
        reportPath: join(tmp, "fail1.json"),
        log: () => {},
      });
      expect(report.pairs).toHaveLength(2);
      for (const p of report.pairs) expect(p.verdict).toBe(-1);
      expect(report.judged).toBe(0);
      expect(report.unjudged).toBe(2);
      expect(report.applied).toBe(0);
      expect(report.config.complete).toBe(true); // fail-OPEN: the run finishes
      expect(await snapshot(P_FAIL)).toEqual(before);
    }, 60_000);

    it("UNJUDGED pairs re-enter the next run (no judged-marker); unparseable JSON marks the batch UNJUDGED and the run continues to the next batch", async () => {
      const before = await snapshot(P_FAIL);
      const calls: string[] = [];
      const judge = async (_s: string, user: string): Promise<string> => {
        calls.push(user);
        return "]]] this is not json {{{";
      };
      const { report } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_FAIL,
        reportPath: join(tmp, "fail2.json"),
        judgeBatch: 1,
        log: () => {},
      });
      // Both pairs re-presented (nothing was marked judged by the failed run),
      // and batch 2 was still attempted after batch 1 came back unparseable.
      expect(calls).toHaveLength(2);
      expect(report.unjudged).toBe(2);
      for (const p of report.pairs) expect(p.verdict).toBe(-1);
      expect(await snapshot(P_FAIL)).toEqual(before);
    }, 60_000);
  });

  // ── AC-305: per-pair schema violations with applying siblings ──────────────

  describe("AC-305 — pair-level violations leave siblings standing", () => {
    beforeAll(async () => {
      const seeds: Array<[string, string, string, number, string, string]> = [
        [V1A, "Verdict-seven probe", TOK.v1a, 0, "v1", "2026-06-01T00:00:00Z"],
        [
          V1B,
          "Verdict-seven probe restated",
          TOK.v1b,
          0.2,
          "v2",
          "2026-06-05T00:00:00Z",
        ],
        [V2A, "Missing-keep probe", TOK.v2a, 0, "w1", "2026-06-01T00:00:00Z"],
        [
          V2B,
          "Missing-keep probe restated",
          TOK.v2b,
          0.2,
          "w2",
          "2026-06-05T00:00:00Z",
        ],
        [V3A, "Valid sibling probe", TOK.v3a, 0, "x1", "2026-06-01T00:00:00Z"],
        [
          V3B,
          "Valid sibling probe restated",
          TOK.v3b,
          0.2,
          "x2",
          "2026-06-05T00:00:00Z",
        ],
      ];
      const clusterOf: Record<string, string> = {
        [TOK.v1a]: "c22",
        [TOK.v1b]: "c22",
        [TOK.v2a]: "c23",
        [TOK.v2b]: "c23",
        [TOK.v3a]: "c24",
        [TOK.v3b]: "c24",
      };
      for (const [id, title, tok, eps, salt, createdAt] of seeds) {
        const cluster = clusterOf[tok];
        await seedRow({
          id,
          projectId: P_MIX,
          title,
          content: `${title.toLowerCase()} ${tok} [vec:${cluster}:${eps}:${salt}]`,
          createdAt,
          vec: H.nearVec(cluster, eps, salt),
        });
      }
    }, 60_000);

    it("unknown integer / missing keep mark ONLY that pair UNJUDGED; the valid sibling in the SAME batch still applies", async () => {
      const { judge, calls } = makeJudge((group) => {
        if (group.includes(TOK.v1a) || group.includes(TOK.v1b)) {
          return { verdict: 7 }; // unknown integer → pair UNJUDGED
        }
        if (group.includes(TOK.v2a) || group.includes(TOK.v2b)) {
          return { verdict: 1 }; // verdict 1 without keep → pair UNJUDGED
        }
        return { verdict: 1, keep: "b", reason: "same claim" };
      });
      const { report } = await consolidate({
        pool: poolDep,
        judge,
        projectId: P_MIX,
        apply: true, // plan step 4: the exported function with apply:true
        reportPath: join(tmp, "mix.json"),
        log: () => {},
      });
      expect(calls).toHaveLength(1); // one batch of three numbered pairs
      expect(report.pairs).toHaveLength(3);
      expect(report.config.apply).toBe(true);

      const byId = (id: string) =>
        report.pairs.find((p) => p.a === id || p.b === id)!;
      expect(byId(V1A).verdict).toBe(-1);
      expect(byId(V2A).verdict).toBe(-1);
      expect(byId(V3A).verdict).toBe(1);
      expect(report.judged).toBe(1);
      expect(report.unjudged).toBe(2);
      expect(report.applied).toBe(1);

      const snap = await snapshot(P_MIX);
      // The valid sibling applied: older row superseded by the newer.
      expect(rowOf(snap, V3A).status).toBe("superseded");
      expect(rowOf(snap, V3A).superseded_by).toBe(V3B);
      expect(rowOf(snap, V3B).status).toBe("active");
      // The violated pairs: zero rows modified.
      for (const id of [V1A, V1B, V2A, V2B]) {
        expect(rowOf(snap, id).status).toBe("active");
        expect(rowOf(snap, id).superseded_by).toBeNull();
      }
    }, 60_000);
  });

  it("zero network calls escaped the mocks across the whole DB suite", () => {
    expect(fetchCalls).toHaveLength(0);
  });
});
