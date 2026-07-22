// Wave-2 Step 4 verify: the two-layer v2 code runner measures DIFFERENT things per
// layer. A gold path is planted in the candidate pool but the (mocked, deterministic)
// reranker demotes it below k=10 → poolRecall = 1 while ndcg@10 = 0; a control gold the
// reranker favors scores ndcg = 1. Path-keyed throughout (paths are the stable code
// gold id) — no supersession resolution for code.
//
// DB-backed on the disposable :5544 Postgres, skipped without DATABASE_URL; Voyage is
// module-mocked (NO live quota).

import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  runCodeEval,
  writeRunArtifact,
  type CodeEvalFile,
  type RunArtifact,
} from "./code-eval.helper.js";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

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
  /** Deterministic rerank with planted extremes: docs carrying "demoteme" always sink
   *  to the bottom, docs carrying "favorme" always rise to the top; fillers get
   *  strictly distinct mid-range scores by index. */
  function fakeRerank(
    _query: string,
    docs: string[],
    topK: number,
  ): Array<{ index: number; score: number }> {
    const scored = docs.map((d, i) => ({
      index: i,
      score: d.includes("demoteme")
        ? 0.011
        : d.includes("favorme")
          ? 0.99
          : 0.5 - i * 0.01,
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.min(topK, docs.length));
  }
  return { fakeVec, fakeRerank };
});

vi.mock("../src/voyage.js", () => ({
  embedCode: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(`code:${t}`)),
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`code:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

const here = dirname(fileURLToPath(import.meta.url));
const PROJ = "runner-code";
const REPO = "runner-repo";
const N = 12; // > k=10 so the demoted gold falls OUT of the final top-k
const pathOf = (n: number) => `src/f${String(n).padStart(2, "0")}.ts`;
const GOLD_DEMOTED = pathOf(1);
const GOLD_FAVORED = pathOf(2);

describe.skipIf(skip)("runCodeEval v2 (two-layer divergence)", () => {
  let pool: pg.Pool;
  const tmp = mkdtempSync(join(tmpdir(), "code-runner-"));

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    const sqlDir = join(here, "..", "sql");
    await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
      PROJ,
    ]);

    let favoredFileId = "";
    for (let n = 1; n <= N; n++) {
      const p = pathOf(n);
      const content =
        n === 1
          ? "export function demoteme() { /* pool surfaces this, rerank buries it */ }"
          : n === 2
            ? "export function favorme() { /* rerank puts this first */ }"
            : `export function filler${n}() { return ${n}; } // padding line ${"x".repeat(n)}`;
      const { rows } = await pool.query<{ id: string }>(
        `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
         VALUES ($1, $2, $3, 'typescript', 'sha-' || $3) RETURNING id`,
        [REPO, PROJ, p],
      );
      if (n === 2) favoredFileId = rows[0].id;
      await pool.query(
        `INSERT INTO codebase.code_chunks
           (file_id, repository_id, project_id, file_path, language, symbol_name,
            start_line, end_line, content, content_sha256, embedding)
         VALUES ($1, $2, $3, $4, 'typescript', $5, 1, 3, $6, 'csha-' || $4, $7::halfvec)`,
        [
          rows[0].id,
          REPO,
          PROJ,
          p,
          `sym${n}`,
          content,
          `[${H.fakeVec(`chunk:${p}`).join(",")}]`,
        ],
      );
    }
    // HIGH-001 regression: a SECOND chunk of the favored file (also "favorme", so the
    // mocked reranker puts BOTH at the top). searchCode then returns the same file path
    // at ranks 1+2; the runner must dedupe to the first occurrence before path-keyed
    // scoring — c-002 below pins ndcg EXACTLY 1 (per-occurrence crediting gave ≈ 1.63).
    await pool.query(
      `INSERT INTO codebase.code_chunks
         (file_id, repository_id, project_id, file_path, language, symbol_name,
          start_line, end_line, content, content_sha256, embedding)
       VALUES ($1, $2, $3, $4, 'typescript', 'sym2b', 4, 6, $5, 'csha2-' || $4, $6::halfvec)`,
      [
        favoredFileId,
        REPO,
        PROJ,
        GOLD_FAVORED,
        "export function favormeToo() { /* favorme: duplicate-path chunk */ }",
        `[${H.fakeVec(`chunk2:${GOLD_FAVORED}`).join(",")}]`,
      ],
    );
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
      PROJ,
    ]);
    await pool.end();
    rmSync(tmp, { recursive: true, force: true });
  });

  const evalFile: CodeEvalFile = {
    version: 2,
    k: 10,
    rows: [
      {
        id: "c-001",
        query: "pool hit that rerank demotes",
        relevantPaths: [GOLD_DEMOTED],
        split: "dev",
        provenance: "seed-v1",
      },
      {
        id: "c-002",
        query: "gold the reranker favors",
        relevantPaths: [GOLD_FAVORED],
        split: "dev",
        provenance: "seed-v1",
      },
      {
        id: "c-003",
        query: "frozen test-split row (must not run)",
        relevantPaths: [GOLD_FAVORED],
        split: "test",
        provenance: "seed-v1",
      },
    ],
  };

  it("pool layer recalls what the reranker later buries (poolRecall=1, ndcg=0)", async () => {
    const result = await runCodeEval(evalFile, {
      projectId: PROJ,
      split: "dev",
    });

    // split filter: the test-split row did NOT run
    expect(result.rows).toBe(2);
    expect(result.perQuery.map((q) => q.id)).toEqual(["c-001", "c-002"]);

    const demoted = result.perQuery[0];
    // Layer 1: the gold path IS in the candidate pool …
    expect(demoted.poolRecall).toBe(1);
    // … Layer 2: the reranker buried it below k=10 (12 candidates, gold sunk last).
    expect(demoted.rank).toBeNull();
    expect(demoted.ndcg).toBe(0);
    expect(demoted.mrr).toBe(0);

    const favored = result.perQuery[1];
    expect(favored.poolRecall).toBe(1);
    expect(favored.rank).toBe(1);
    // Two "favorme" chunks of the SAME file sit at final ranks 1+2 — the duplicate path
    // is deduped before scoring, so the single gold credits exactly once (never > 1).
    expect(favored.ndcg).toBe(1);
    expect(favored.mrr).toBe(1);

    // the two layers measured different things on the same corpus
    expect(result.aggregates.recallAt25).toBe(1);
    expect(result.aggregates.ndcgAt10).toBeCloseTo(0.5, 10);
  });

  it("writes the artifact with the full retrievalConfig() snapshot (AC-104)", async () => {
    const result = await runCodeEval(evalFile, {
      projectId: PROJ,
      split: "dev",
    });
    const path = writeRunArtifact(tmp, "code-runner-verify", result);
    const parsed = JSON.parse(readFileSync(path, "utf8")) as RunArtifact;
    const { retrievalConfig } = await import("../src/search.js");
    expect(parsed.retrievalConfig).toEqual(retrievalConfig());
    expect(parsed.evalVersion).toBe(2);
    expect(parsed.split).toBe("dev");
    expect(parsed.perQuery).toEqual(result.perQuery);
  });
});
