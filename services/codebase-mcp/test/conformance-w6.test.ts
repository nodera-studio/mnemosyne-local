// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Wave 6 (WS5: rerank-2.5 gated swap)
//
// Plan-derived, implementation-blind behavior tests for wave-6 of the retrieval
// improvement program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/
// wave-6-rerank-swap.md, Steps 1/2/3/5 + AC-501/AC-502). Assertions were drafted
// from the plan BEFORE locating exported symbols; independent of the
// Implementer's own tests (test/bakeoff-rerank.test.ts — deliberately not read).
//
// Check map:
//   Step 1  — rerank() gains a trailing optional model param in BOTH services;
//             default = config.rerankModel; the override reaches the request
//             body (mocked fetch, zero live Voyage).
//   AC-502  — the swap is CONFIG-ONLY: setting VOYAGE_RERANK_MODEL alone changes
//             the model in the rerank request body of both services.
//   Step 2  — `npm run bakeoff:rerank` exists; PAID gate refuses without --yes
//             BEFORE any dependency call; TEST split only; <15-row guard;
//             candidatePool must be 25 (Recall@25 label integrity, mirroring the
//             wave-2 runner); embed once per query SHARED across arms; both arms
//             receive IDENTICAL doc lists; per-arm model strings rerank-2.5-lite
//             vs rerank-2.5; docs truncated at the SAME 1500 chars as live;
//             `embedding_ctx` (dropped by sql/005) appears in NO SQL issued;
//             rank-based scoring (nDCG@10 post-cut first-occurrence path dedupe,
//             pool Recall@25) matching hand-computed values; paired deltas + CI
//             match pairedBootstrapCI(seed 42) called directly; verdict
//             boundaries SWAP / KEEP LITE / NO SIGNIFICANT DIFFERENCE; artifact
//             test/runs/<date>-bakeoff-rerank.json carries retrievalConfig() +
//             both models + per-query scores.
//   AC-108  — importing the bakeoff module executes nothing.
//   Step 5  — README PAID inventory lists bakeoff:rerank; memory-mcp retune.md's
//             reranker row points at the wave-6 decision record in bakeoff.md.
//
// Deterministic throughout: fetch is replaced by a throwing sentinel (any
// unexpected network call fails loudly); paid deps are injected mocks. DB tests
// are self-contained under conf-w6-* ids and clean up after themselves.
// ─────────────────────────────────────────────────────────────────────────────

import {
  readFileSync,
  readdirSync,
  mkdtempSync,
  rmSync,
  existsSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { basename, dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import type { Pool } from "pg";

process.env.VOYAGE_API_KEY ??= "conf-w6-test-key";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

import {
  ndcgAtK,
  pairedBootstrapCI,
  POOL_RECALL_K,
  type CodeEvalFile,
  type CodeEvalRow,
} from "../src/recall-math.js";
import { retrievalConfig } from "../src/search.js";
import { rerank as codebaseRerank } from "../src/voyage.js";
import { rerank as memoryRerank } from "../../memory-mcp/src/voyage.js";
import {
  bakeoffArtifact,
  runRerankBakeoff,
  scoreRerankArm,
  writeBakeoffArtifact,
  type RerankBakeoffResult,
  type RerankFn,
} from "../src/db/bakeoff-rerank.js";

const here = dirname(fileURLToPath(import.meta.url));
const serviceRoot = join(here, "..");
const repoRoot = join(serviceRoot, "..", "..");

// Plan-pinned model strings (wave-6 context: config default rerank-2.5-lite;
// the challenger is rerank-2.5).
const LITE = "rerank-2.5-lite";
const FULL = "rerank-2.5";
/** nDCG@10 of a single relevant doc at rank 2 (1 relevant → IDCG = 1). */
const NDCG_RANK2 = 1 / Math.log2(3);

// ── zero-network sentinel ─────────────────────────────────────────────────────
const realFetch = globalThis.fetch;
const netCalls: string[] = [];
const sentinelFetch = (async (...args: unknown[]) => {
  netCalls.push(String(args[0]));
  throw new Error(
    `conformance-w6: unexpected network call to ${String(args[0])}`,
  );
}) as typeof fetch;

beforeAll(() => {
  globalThis.fetch = sentinelFetch;
});
afterAll(() => {
  globalThis.fetch = realFetch;
  expect(netCalls, "no live network call anywhere in the suite").toEqual([]);
});

// ── deterministic helpers ─────────────────────────────────────────────────────

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
function mulberry(seed: number): () => number {
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
  const rnd = mulberry(fnv1a(key));
  return Array.from({ length: dim }, () => Number((rnd() * 2 - 1).toFixed(3)));
}

/** A fetch mock that records each request body and answers a valid empty-ish
 *  Voyage rerank response ({ok:true}, so the retry helper returns immediately). */
function capturingVoyageFetch() {
  const bodies: Array<{ url: string; body: Record<string, unknown> }> = [];
  const fn = (async (url: unknown, init?: { body?: string }) => {
    const body = JSON.parse(init?.body ?? "{}") as Record<string, unknown>;
    bodies.push({ url: String(url), body });
    return {
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => ({ data: [{ index: 0, relevance_score: 0.9 }] }),
      text: async () => "",
    } as unknown as Response;
  }) as typeof fetch;
  return { fn, bodies };
}

// Gold marker used by mock rerankers to spot the relevant doc.
const GOLD_MARKER = "confw6-gold-alpha-marker";

/** Mock RerankFn factory: `winner` decides which arm ranks the gold doc first.
 *  Records every call so the shared-fuse / per-arm-model invariants are
 *  observable. `alternate` flips the winner per query index (CI spans zero). */
function makeRerankMock(winner: "full" | "lite" | "same" | "alternate"): {
  fn: RerankFn;
  calls: Array<{ query: string; docs: string[]; topK: number; model: string }>;
} {
  const calls: Array<{
    query: string;
    docs: string[];
    topK: number;
    model: string;
  }> = [];
  const fn: RerankFn = async (query, docs, topK, model) => {
    calls.push({ query, docs: [...docs], topK, model });
    const goldIdx = docs.findIndex((d) => d.includes(GOLD_MARKER));
    const rest = docs.map((_, i) => i).filter((i) => i !== goldIdx);
    let order: number[];
    if (goldIdx === -1) {
      order = docs.map((_, i) => i);
    } else {
      const qNum = Number((query.match(/(\d+)\s*$/) ?? [])[1] ?? 0);
      const fullWins =
        winner === "full" || (winner === "alternate" && qNum % 2 === 0);
      const liteWins =
        winner === "lite" || (winner === "alternate" && qNum % 2 === 1);
      const goldFirst =
        winner === "same" ||
        (fullWins && model === FULL) ||
        (liteWins && model === LITE);
      order = goldFirst
        ? [goldIdx, ...rest]
        : rest.length > 0
          ? [rest[0], goldIdx, ...rest.slice(1)]
          : [goldIdx];
    }
    return order
      .slice(0, Math.min(topK, docs.length))
      .map((index, pos) => ({ index, score: 1 - pos * 0.01 }));
  };
  return { fn, calls };
}

/** Stub pg pool: answers EVERY query with the given candidate rows and records
 *  each SQL text (the injectable-pool contract from the plan). */
function makeStubPool(
  rows: Array<{ file_path: string; content: string; rrf: number }>,
) {
  const sqls: string[] = [];
  const pool = {
    query: async (sql: unknown) => {
      sqls.push(
        typeof sql === "string"
          ? sql
          : String((sql as { text?: string })?.text ?? sql),
      );
      return { rows, rowCount: rows.length };
    },
  } as unknown as Pool;
  return { pool, sqls };
}

function testEvalFile(nTest: number, nDev = 0): CodeEvalFile {
  const rows: CodeEvalRow[] = [];
  for (let i = 1; i <= nTest; i++) {
    rows.push({
      id: `c-w6t-${String(i).padStart(2, "0")}`,
      query: `confw6 retrieval probe ${i}`,
      relevantPaths: ["src/alpha.ts"],
      split: "test",
      provenance: "seed-v1",
    });
  }
  for (let i = 1; i <= nDev; i++) {
    rows.push({
      id: `c-w6d-${String(i).padStart(2, "0")}`,
      query: `confw6 devonly probe ${i}`,
      relevantPaths: ["src/alpha.ts"],
      split: "dev",
      provenance: "seed-v1",
    });
  }
  return {
    version: 2,
    changelog: ["conf-w6 synthetic gold (never committed)"],
    k: 10,
    rows,
  };
}

/** Three-candidate corpus for the stub-pool (pure) runs: gold + two decoys. */
const STUB_CANDIDATES = [
  { file_path: "src/beta.ts", content: "confw6 decoy beta content", rrf: 0.03 },
  {
    file_path: "src/alpha.ts",
    // > 1500 chars so the live 1500-char doc truncation is observable.
    content: `${GOLD_MARKER} ${"alpha-padding ".repeat(200)}`,
    rrf: 0.02,
  },
  {
    file_path: "src/gamma.ts",
    content: "confw6 decoy gamma content",
    rrf: 0.01,
  },
];

const embedCounter = () => {
  const queries: string[] = [];
  const fn = async (query: string) => {
    queries.push(query);
    return fakeVec(`q:${query}`);
  };
  return { fn, queries };
};

// ═══════════════════════════════════════════════════════════════════════════════
// Step 1 — rerank() optional model param (both services), mocked fetch
// ═══════════════════════════════════════════════════════════════════════════════

describe("Step 1: rerank() model parameter — codebase-mcp", () => {
  it("default call sends config default (rerank-2.5-lite) in the request body", async () => {
    const { fn, bodies } = capturingVoyageFetch();
    globalThis.fetch = fn;
    try {
      await codebaseRerank("q", ["doc one", "doc two"], 2);
      expect(bodies).toHaveLength(1);
      expect(bodies[0].url).toContain("/rerank");
      expect(bodies[0].body.model).toBe(LITE);
      expect(bodies[0].body.query).toBe("q");
      expect(bodies[0].body.documents).toEqual(["doc one", "doc two"]);
    } finally {
      globalThis.fetch = sentinelFetch;
    }
  });

  it("explicit model override reaches the request body unchanged", async () => {
    const { fn, bodies } = capturingVoyageFetch();
    globalThis.fetch = fn;
    try {
      await codebaseRerank("q", ["doc"], 1, FULL);
      expect(bodies[0].body.model).toBe(FULL);
    } finally {
      globalThis.fetch = sentinelFetch;
    }
  });
});

describe("Step 1: rerank() model parameter — memory-mcp", () => {
  it("default call sends config default (rerank-2.5-lite) in the request body", async () => {
    const { fn, bodies } = capturingVoyageFetch();
    globalThis.fetch = fn;
    try {
      await memoryRerank("q", ["doc one", "doc two"], 2);
      expect(bodies).toHaveLength(1);
      expect(bodies[0].url).toContain("/rerank");
      expect(bodies[0].body.model).toBe(LITE);
    } finally {
      globalThis.fetch = sentinelFetch;
    }
  });

  it("explicit model override reaches the request body unchanged", async () => {
    const { fn, bodies } = capturingVoyageFetch();
    globalThis.fetch = fn;
    try {
      await memoryRerank("q", ["doc"], 1, FULL);
      expect(bodies[0].body.model).toBe(FULL);
    } finally {
      globalThis.fetch = sentinelFetch;
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// AC-502 — the swap is a VOYAGE_RERANK_MODEL pin, no code change
// ═══════════════════════════════════════════════════════════════════════════════

describe("AC-502: VOYAGE_RERANK_MODEL env pin alone flips the default model", () => {
  for (const [service, modPath] of [
    ["codebase-mcp", "../src/voyage.js"],
    ["memory-mcp", "../../memory-mcp/src/voyage.js"],
  ] as const) {
    it(`${service}: env-pinned rerank-2.5 becomes the no-arg default in the request body`, async () => {
      vi.stubEnv("VOYAGE_RERANK_MODEL", FULL);
      vi.stubEnv("VOYAGE_API_KEY", "conf-w6-test-key");
      vi.resetModules();
      const { fn, bodies } = capturingVoyageFetch();
      globalThis.fetch = fn;
      try {
        const mod = (await import(modPath)) as {
          rerank: (q: string, d: string[], k: number) => Promise<unknown>;
        };
        await mod.rerank("q", ["doc"], 1);
        expect(bodies[0].body.model).toBe(FULL);
      } finally {
        globalThis.fetch = sentinelFetch;
        vi.unstubAllEnvs();
        vi.resetModules();
      }
    });
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// Step 2 — operator gating (AC-108, --yes, npm script)
// ═══════════════════════════════════════════════════════════════════════════════

describe("Step 2: PAID operator gating", () => {
  it("package.json exposes bakeoff:rerank running the bakeoff via tsx", () => {
    const pkg = JSON.parse(
      readFileSync(join(serviceRoot, "package.json"), "utf8"),
    ) as {
      scripts: Record<string, string>;
    };
    expect(pkg.scripts["bakeoff:rerank"]).toBeTruthy();
    expect(pkg.scripts["bakeoff:rerank"]).toContain("bakeoff-rerank");
    expect(pkg.scripts["bakeoff:rerank"]).toContain("tsx");
  });

  it("AC-108: importing the bakeoff module executes nothing (no network, no exit)", async () => {
    const before = netCalls.length;
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((
      code?: number,
    ) => {
      throw new Error(`conformance-w6: process.exit(${code}) during import`);
    }) as never);
    try {
      vi.resetModules();
      const mod = await import("../src/db/bakeoff-rerank.js");
      expect(typeof mod.scoreRerankArm).toBe("function");
      expect(typeof mod.runRerankBakeoff).toBe("function");
      expect(netCalls.length).toBe(before);
      expect(exitSpy).not.toHaveBeenCalled();
    } finally {
      exitSpy.mockRestore();
      vi.resetModules();
    }
  });

  it("running as main WITHOUT --yes refuses before any dependency call", () => {
    const tsxBin = join(serviceRoot, "node_modules", ".bin", "tsx");
    const res = spawnSync(tsxBin, ["src/db/bakeoff-rerank.ts"], {
      cwd: serviceRoot,
      env: {
        ...process.env,
        // Closed port: if the script touched the DB before refusing, the output
        // would carry ECONNREFUSED instead of the --yes refusal.
        DATABASE_URL: "postgresql://postgres:postgres@127.0.0.1:9/conf_w6_none",
        VOYAGE_API_KEY: "conf-w6-test-key",
      },
      encoding: "utf8",
      timeout: 60_000,
    });
    const out = `${res.stdout ?? ""}\n${res.stderr ?? ""}`;
    expect(
      res.status,
      `script must terminate on its own; output:\n${out}`,
    ).not.toBeNull();
    expect(out).toMatch(/--yes/);
    expect(out).not.toMatch(/ECONNREFUSED/);
    // A refused PAID run must not report success.
    expect(res.status).not.toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Step 2 — misconfiguration guards
// ═══════════════════════════════════════════════════════════════════════════════

describe("Step 2: misconfiguration guards", () => {
  it("refuses an eval file whose TEST split has <15 rows, before any dependency call", async () => {
    const { pool, sqls } = makeStubPool(STUB_CANDIDATES);
    const embeds = embedCounter();
    const rr = makeRerankMock("same");
    await expect(
      scoreRerankArm({
        pool,
        evalFile: testEvalFile(3, 20), // plenty of dev rows — only test rows count
        model: LITE,
        rerankFn: rr.fn,
        embedQuery: embeds.fn,
        projectId: "conf-w6-guard",
      }),
    ).rejects.toThrow(/15/);
    expect(sqls).toHaveLength(0);
    expect(embeds.queries).toHaveLength(0);
    expect(rr.calls).toHaveLength(0);
  });

  it("refuses when config.candidatePool drifts from 25 (Recall@25 label integrity)", async () => {
    vi.stubEnv("CANDIDATE_POOL", "30");
    vi.stubEnv("VOYAGE_API_KEY", "conf-w6-test-key");
    vi.resetModules();
    try {
      const mod = await import("../src/db/bakeoff-rerank.js");
      const { pool, sqls } = makeStubPool(STUB_CANDIDATES);
      const embeds = embedCounter();
      const rr = makeRerankMock("same");
      await expect(
        mod.runRerankBakeoff({
          pool,
          evalFile: testEvalFile(16),
          rerankFn: rr.fn,
          embedQuery: embeds.fn,
          projectId: "conf-w6-guard",
        }),
      ).rejects.toThrow(/25|candidatePool|CANDIDATE_POOL/i);
      expect(sqls).toHaveLength(0);
      expect(embeds.queries).toHaveLength(0);
      expect(rr.calls).toHaveLength(0);
    } finally {
      vi.unstubAllEnvs();
      vi.resetModules();
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Step 2/3 — bakeoff pipeline behavior (stub pool: pure, no DB)
// ═══════════════════════════════════════════════════════════════════════════════

async function runStubBakeoff(winner: "full" | "lite" | "same" | "alternate") {
  const { pool, sqls } = makeStubPool(STUB_CANDIDATES);
  const embeds = embedCounter();
  const rr = makeRerankMock(winner);
  const result = await runRerankBakeoff({
    pool,
    evalFile: testEvalFile(16, 2),
    rerankFn: rr.fn,
    embedQuery: embeds.fn,
    projectId: "conf-w6-pure",
    log: () => {},
  });
  return { result, sqls, embeds, rr };
}

describe("Step 2: pipeline invariants (stub pool)", () => {
  it("scores ONLY the frozen TEST split; dev rows never reach embed or rerank", async () => {
    const { result, embeds, rr } = await runStubBakeoff("full");
    expect(result.lite.rows).toBe(16);
    expect(result.full.rows).toBe(16);
    expect(result.lite.perQuery.map((q) => q.id)).toEqual(
      Array.from(
        { length: 16 },
        (_, i) => `c-w6t-${String(i + 1).padStart(2, "0")}`,
      ),
    );
    for (const q of embeds.queries) expect(q).not.toContain("devonly");
    for (const c of rr.calls) expect(c.query).not.toContain("devonly");
  });

  it("embeds each query exactly ONCE, shared across both arms", async () => {
    const { embeds } = await runStubBakeoff("full");
    expect(embeds.queries).toHaveLength(16);
    expect(new Set(embeds.queries).size).toBe(16);
  });

  it("both arms receive IDENTICAL doc lists; model strings are per-arm", async () => {
    const { rr } = await runStubBakeoff("full");
    expect(rr.calls).toHaveLength(32);
    const models = new Set(rr.calls.map((c) => c.model));
    expect(models).toEqual(new Set([LITE, FULL]));
    const byQuery = new Map<string, Array<{ model: string; docs: string[] }>>();
    for (const c of rr.calls) {
      (byQuery.get(c.query) ?? byQuery.set(c.query, []).get(c.query)!).push(c);
    }
    expect(byQuery.size).toBe(16);
    for (const [query, calls] of byQuery) {
      expect(calls, query).toHaveLength(2);
      expect(new Set(calls.map((c) => c.model))).toEqual(new Set([LITE, FULL]));
      expect(calls[0].docs, `identical docs for "${query}"`).toEqual(
        calls[1].docs,
      );
    }
  });

  it("docs handed to the reranker are truncated at the SAME 1500 chars as live", async () => {
    const { rr } = await runStubBakeoff("full");
    for (const c of rr.calls) {
      for (const d of c.docs) expect(d.length).toBeLessThanOrEqual(1500);
      // The gold doc's source is > 1500 chars → its doc must be cut at exactly 1500.
      const gold = c.docs.find((d) => d.includes(GOLD_MARKER));
      expect(gold).toBeTruthy();
      expect(gold!.length).toBe(1500);
    }
  });

  it("never touches embedding_ctx in any SQL issued (AC-501; column dropped by sql/005)", async () => {
    const { sqls } = await runStubBakeoff("full");
    expect(sqls.length).toBeGreaterThan(0);
    for (const sql of sqls) expect(sql).not.toMatch(/embedding_ctx/i);
  });

  it("rank-based scoring: per-query nDCG@10 matches hand-computed values; pool Recall@25 = 1", async () => {
    const { result } = await runStubBakeoff("full");
    for (const q of result.full.perQuery) {
      expect(q.ndcgAt10).toBeCloseTo(1.0, 10); // gold at rank 1
      expect(q.poolRecallAt25).toBe(1);
      expect(q.rank).toBe(1);
    }
    for (const q of result.lite.perQuery) {
      expect(q.ndcgAt10).toBeCloseTo(NDCG_RANK2, 10); // gold at rank 2
      expect(q.poolRecallAt25).toBe(1);
      expect(q.rank).toBe(2);
    }
    expect(result.full.aggregates.ndcgAt10).toBeCloseTo(1.0, 10);
    expect(result.lite.aggregates.ndcgAt10).toBeCloseTo(NDCG_RANK2, 10);
    expect(result.full.aggregates.recallAt25).toBe(1);
    expect(result.lite.aggregates.recallAt25).toBe(1);
  });

  it("paired deltas are full−lite per query and the CI matches pairedBootstrapCI(seed 42) directly", async () => {
    const { result } = await runStubBakeoff("full");
    expect(result.deltas).toHaveLength(16);
    const liteById = new Map(
      result.lite.perQuery.map((q) => [q.id, q.ndcgAt10]),
    );
    const fullById = new Map(
      result.full.perQuery.map((q) => [q.id, q.ndcgAt10]),
    );
    const expected = [...liteById.keys()].map(
      (id) => fullById.get(id)! - liteById.get(id)!,
    );
    expect([...result.deltas].sort()).toEqual([...expected].sort());
    expect(result.ci).toEqual(pairedBootstrapCI(result.deltas, { seed: 42 }));
  });
});

describe("Step 2: verdict boundaries", () => {
  it("full wins uniformly → mean>0, CI excludes zero on the win side → SWAP", async () => {
    const { result } = await runStubBakeoff("full");
    expect(result.ci.mean).toBeGreaterThan(0);
    expect(result.ci.ciLow).toBeGreaterThan(0);
    expect(result.verdict).toBe("SWAP");
  });

  it("lite wins uniformly → CI excludes zero on the loss side → KEEP LITE", async () => {
    const { result } = await runStubBakeoff("lite");
    expect(result.ci.mean).toBeLessThan(0);
    expect(result.ci.ciHigh).toBeLessThan(0);
    expect(result.verdict).toBe("KEEP LITE");
  });

  it("identical orderings → zero deltas → NO SIGNIFICANT DIFFERENCE (zero is not 'excluded')", async () => {
    const { result } = await runStubBakeoff("same");
    expect(result.deltas.every((d) => d === 0)).toBe(true);
    expect(result.ci).toEqual({ mean: 0, ciLow: 0, ciHigh: 0 });
    expect(result.verdict).toMatch(/NO SIGNIFICANT DIFFERENCE/);
  });

  it("alternating winners → CI spans zero → NO SIGNIFICANT DIFFERENCE", async () => {
    const { result } = await runStubBakeoff("alternate");
    expect(result.ci.ciLow).toBeLessThan(0);
    expect(result.ci.ciHigh).toBeGreaterThan(0);
    expect(result.verdict).toMatch(/NO SIGNIFICANT DIFFERENCE/);
  });
});

describe("Step 2: rank-scoring parity with the eval runner (post-cut first-occurrence dedupe)", () => {
  it("a relevant doc pushed past rank 10 by duplicate paths scores nDCG 0 (cut BEFORE dedupe), while pool recall stays 1", async () => {
    // 10 chunks of the same file ahead of the single relevant doc: the top-10
    // cut happens first (duplicates burn rank slots), THEN first-occurrence
    // dedupe — so the relevant doc must NOT be credited.
    const dupRows = Array.from({ length: 10 }, (_, i) => ({
      file_path: "src/dup.ts",
      content: `confw6 dup chunk ${i}`,
      rrf: 0.5 - i * 0.01,
    }));
    const rows = [
      ...dupRows,
      { file_path: "src/rel.ts", content: "confw6 rel", rrf: 0.01 },
    ];
    const { pool } = makeStubPool(rows);
    const embeds = embedCounter();
    const identityRerank: RerankFn = async (_q, docs, topK) =>
      docs
        .map((_, index) => ({ index, score: 1 - index * 0.01 }))
        .slice(0, Math.min(topK, docs.length));

    const evalFile: CodeEvalFile = {
      version: 2,
      changelog: ["conf-w6 dedupe fixture"],
      k: 10,
      rows: Array.from({ length: 16 }, (_, i) => ({
        id: `c-w6dup-${String(i + 1).padStart(2, "0")}`,
        query: `confw6 dedupe probe ${i + 1}`,
        relevantPaths: ["src/rel.ts"],
        split: "test" as const,
        provenance: "seed-v1",
      })),
    };
    const arm = await scoreRerankArm({
      pool,
      evalFile,
      model: LITE,
      rerankFn: identityRerank,
      embedQuery: embeds.fn,
      projectId: "conf-w6-dedupe",
    });
    for (const q of arm.perQuery) {
      expect(q.ndcgAt10).toBe(0);
      expect(q.rank).toBeNull();
      expect(q.poolRecallAt25).toBe(1); // rel.ts IS in the 25-pool — recall layer unaffected
      // deduped hits carry no duplicate paths
      expect(new Set(q.hits).size).toBe(q.hits.length);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Step 2 — artifact contract
// ═══════════════════════════════════════════════════════════════════════════════

describe("Step 2: run artifact (retrievalConfig + both models + per-query scores)", () => {
  let result: RerankBakeoffResult;
  beforeAll(async () => {
    ({ result } = await runStubBakeoff("full"));
  });

  it("bakeoffArtifact carries the full retrievalConfig() snapshot, both model strings, verdict, CI, per-query scores", () => {
    const artifact = bakeoffArtifact(result);
    expect(artifact.retrievalConfig).toEqual(retrievalConfig());
    expect(artifact.models).toEqual({ lite: LITE, full: FULL });
    expect(artifact.split).toBe("test");
    expect(artifact.verdict).toBe(result.verdict);
    expect(artifact.pairedBootstrap).toEqual(result.ci);
    expect(artifact.perQuery).toHaveLength(16);
    for (const q of artifact.perQuery) {
      expect(typeof q.lite.ndcgAt10).toBe("number");
      expect(typeof q.full.ndcgAt10).toBe("number");
      expect(typeof q.lite.poolRecallAt25).toBe("number");
      expect(typeof q.full.poolRecallAt25).toBe("number");
      expect(q.fullMinusLite).toBeCloseTo(
        q.full.ndcgAt10 - q.lite.ndcgAt10,
        10,
      );
    }
  });

  it("writeBakeoffArtifact persists <date>-bakeoff-rerank.json (JSON round-trip safe)", () => {
    const tmp = mkdtempSync(join(tmpdir(), "conf-w6-runs-"));
    try {
      const path = writeBakeoffArtifact(result, tmp);
      expect(existsSync(path)).toBe(true);
      expect(basename(path)).toMatch(
        /^\d{4}-\d{2}-\d{2}-bakeoff-rerank\.json$/,
      );
      const parsed = JSON.parse(readFileSync(path, "utf8")) as ReturnType<
        typeof bakeoffArtifact
      >;
      expect(parsed.retrievalConfig).toEqual(retrievalConfig());
      expect(parsed.models).toEqual({ lite: LITE, full: FULL });
      expect(parsed.perQuery).toHaveLength(16);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("the committed artifact directory convention (test/runs/) exists", () => {
    expect(existsSync(join(here, "runs"))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Step 5 — documentation closes the loop
// ═══════════════════════════════════════════════════════════════════════════════

describe("Step 5: documentation", () => {
  it("repo README lists bakeoff:rerank in the PAID script inventory", () => {
    const readme = readFileSync(join(repoRoot, "README.md"), "utf8");
    expect(readme).toContain("bakeoff:rerank");
  });

  it("memory-mcp retune.md's reranker row points at the wave-6 decision record in bakeoff.md", () => {
    const retune = readFileSync(
      join(repoRoot, "services", "memory-mcp", "test", "retune.md"),
      "utf8",
    );
    const rerankerLines = retune
      .split("\n")
      .filter((l) => /reranker/i.test(l) && /\|/.test(l));
    expect(rerankerLines.length).toBeGreaterThan(0);
    expect(rerankerLines.some((l) => /bakeoff\.md/i.test(l))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// DB-backed conformance — the real fuse SQL feeding both arms
// (skipped cleanly without DATABASE_URL)
// ═══════════════════════════════════════════════════════════════════════════════

const P_W6 = "conf-w6-rerank";
const R_W6 = "conf-w6-repo";
const cid = (n: number) =>
  `60000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

interface W6Chunk {
  id: string;
  path: string;
  symbolName: string;
  content: string;
}

const W6_CHUNKS: W6Chunk[] = [
  {
    id: cid(1),
    path: "src/alpha.ts",
    symbolName: "alphaGold",
    // > 1500 chars so live truncation is observable through the real SQL path.
    content: `${GOLD_MARKER} confw6 retrieval corpus alpha ${"padding-alpha ".repeat(160)}`,
  },
  {
    id: cid(2),
    path: "src/beta.ts",
    symbolName: "betaDecoy",
    content: "confw6 retrieval corpus beta decoy",
  },
  {
    id: cid(3),
    path: "src/gamma.ts",
    symbolName: "gammaDecoy",
    content: "confw6 retrieval corpus gamma decoy",
  },
  {
    id: cid(4),
    path: "src/delta.ts",
    symbolName: "deltaDecoy",
    content: "confw6 retrieval corpus delta decoy",
  },
  {
    id: cid(5),
    path: "src/epsilon.ts",
    symbolName: "epsilonDecoy",
    content: "confw6 retrieval corpus epsilon decoy",
  },
];

describe.skipIf(skip)("W6 conformance — DB-backed (real fuse SQL)", () => {
  let db: pg.Pool;
  let result: RerankBakeoffResult;
  let sqls: string[];
  let embeds: ReturnType<typeof embedCounter>;
  let rr: ReturnType<typeof makeRerankMock>;

  async function cleanup(): Promise<void> {
    await db.query(`DELETE FROM codebase.files WHERE repository_id = $1`, [
      R_W6,
    ]);
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    await db.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    const sqlDir = join(serviceRoot, "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();

    for (const c of W6_CHUNKS) {
      const { rows } = await db.query<{ id: string }>(
        `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
         VALUES ($1, $2, $3, 'typescript', $4)
         ON CONFLICT (repository_id, path)
           DO UPDATE SET content_sha256 = EXCLUDED.content_sha256
         RETURNING id`,
        [R_W6, P_W6, c.path, `conf-w6-sha-${c.path}`],
      );
      await db.query(
        `INSERT INTO codebase.code_chunks
           (id, file_id, repository_id, project_id, file_path, language,
            symbol_name, start_line, end_line, content, content_sha256, embedding)
         VALUES ($1, $2, $3, $4, $5, 'typescript', $6, 1, 20, $7, $8, $9::halfvec)`,
        [
          c.id,
          rows[0].id,
          R_W6,
          P_W6,
          c.path,
          c.symbolName,
          c.content,
          `conf-w6-chunk-sha-${c.id}`,
          `[${fakeVec(`chunk:${c.id}`).join(",")}]`,
        ],
      );
    }

    // Instrumented REAL pool: capture every SQL text handed to it.
    sqls = [];
    const captured = sqls;
    const instrumented = new Proxy(db, {
      get(target, prop) {
        if (prop === "query") {
          return (...args: unknown[]) => {
            const q = args[0];
            captured.push(
              typeof q === "string"
                ? q
                : String((q as { text?: string })?.text ?? q),
            );
            return (target.query as unknown as (...a: unknown[]) => unknown)(
              ...args,
            );
          };
        }
        const v = Reflect.get(target, prop, target);
        return typeof v === "function"
          ? (v as (...a: unknown[]) => unknown).bind(target)
          : v;
      },
    }) as pg.Pool;

    embeds = embedCounter();
    rr = makeRerankMock("full");
    result = await runRerankBakeoff({
      pool: instrumented,
      evalFile: testEvalFile(16, 2),
      rerankFn: rr.fn,
      embedQuery: embeds.fn,
      projectId: P_W6,
      repo: R_W6,
      log: () => {},
    });
  });

  afterAll(async () => {
    await cleanup();
    await db.end();
  });

  it("AC-501: no SQL issued against the live DB mentions embedding_ctx", () => {
    expect(sqls.length).toBeGreaterThan(0);
    expect(sqls.some((s) => /code_chunks/i.test(s))).toBe(true);
    for (const sql of sqls) expect(sql).not.toMatch(/embedding_ctx/i);
  });

  it("scores the frozen TEST split only — 16 rows, dev rows untouched", () => {
    expect(result.lite.rows).toBe(16);
    expect(result.full.rows).toBe(16);
    for (const q of embeds.queries) expect(q).not.toContain("devonly");
    for (const c of rr.calls) expect(c.query).not.toContain("devonly");
  });

  it("embeds once per query, shared across arms; both arms see IDENTICAL doc lists from the real fuse", () => {
    expect(embeds.queries).toHaveLength(16);
    expect(rr.calls).toHaveLength(32);
    const byQuery = new Map<string, Array<{ model: string; docs: string[] }>>();
    for (const c of rr.calls) {
      (byQuery.get(c.query) ?? byQuery.set(c.query, []).get(c.query)!).push(c);
    }
    for (const [query, calls] of byQuery) {
      expect(calls, query).toHaveLength(2);
      expect(new Set(calls.map((c) => c.model))).toEqual(new Set([LITE, FULL]));
      expect(calls[0].docs, `identical docs for "${query}"`).toEqual(
        calls[1].docs,
      );
      expect(calls[0].docs.length).toBeGreaterThan(0);
    }
  });

  it("real docs are truncated at 1500 chars; the >1500-char gold chunk is cut to exactly 1500", () => {
    let sawGold = false;
    for (const c of rr.calls) {
      for (const d of c.docs) expect(d.length).toBeLessThanOrEqual(1500);
      const gold = c.docs.find((d) => d.includes(GOLD_MARKER));
      if (gold) {
        sawGold = true;
        expect(gold.length).toBe(1500);
      }
    }
    expect(sawGold, "gold doc must surface in the candidate pool").toBe(true);
  });

  it("hand-computed rank scoring holds end-to-end: full=1.0, lite=1/log2(3), Recall@25=1, verdict SWAP", () => {
    for (const q of result.full.perQuery) {
      expect(q.ndcgAt10).toBeCloseTo(1.0, 10);
      expect(q.poolRecallAt25).toBe(1);
    }
    for (const q of result.lite.perQuery) {
      expect(q.ndcgAt10).toBeCloseTo(NDCG_RANK2, 10);
      expect(q.poolRecallAt25).toBe(1);
    }
    expect(result.ci).toEqual(pairedBootstrapCI(result.deltas, { seed: 42 }));
    expect(result.ci.ciLow).toBeGreaterThan(0);
    expect(result.verdict).toBe("SWAP");
  });

  it("the artifact from the DB run carries retrievalConfig() + both models + 16 per-query scores", () => {
    const artifact = bakeoffArtifact(result);
    expect(artifact.retrievalConfig).toEqual(retrievalConfig());
    expect(artifact.retrievalConfig.candidatePool).toBe(POOL_RECALL_K);
    expect(artifact.models).toEqual({ lite: LITE, full: FULL });
    expect(artifact.perQuery).toHaveLength(16);
  });
});
