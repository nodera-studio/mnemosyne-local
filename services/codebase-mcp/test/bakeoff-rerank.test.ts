// Wave 6 rerank bakeoff tests: DB-backed candidate fusion with fully MOCKED query
// embedding + reranking. No live Voyage quota, and the post-drop scratch column name
// must never appear in SQL issued by the bakeoff path.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import pg from "pg";
import {
  FULL_RERANK_MODEL,
  LITE_RERANK_MODEL,
  runRerankBakeoff,
  scoreRerankArm,
  type QueryEmbedder,
  type RerankFn,
} from "../src/db/bakeoff-rerank.js";
import type { QueryResult } from "pg";
import {
  ndcgAtK,
  pairedBootstrapCI,
  type CodeEvalFile,
} from "../src/recall-math.js";

const HAS_DB = !!process.env.DATABASE_URL;

function oneHot(key: number): number[] {
  const v = Array.from({ length: 1024 }, () => 0);
  v[((key % 1024) + 1024) % 1024] = 100;
  return v;
}

function keyForText(t: string): number {
  const m = t.match(/K(\d+)/);
  return m ? Number(m[1]) : 0;
}

function evalFile(split: "dev" | "test", n = 15): CodeEvalFile {
  return {
    version: 2,
    k: 10,
    rows: Array.from({ length: n }, (_, i) => ({
      id: `c-${String(i + 1).padStart(3, "0")}`,
      query: `K${i + 1} lookup`,
      relevantPaths: [`src/file${i + 1}.ts`],
      split,
      provenance: "seed-v1",
    })),
  };
}

function makeRerankMock(calls: {
  query: string;
  docs: string[];
  topK: number;
  model: string;
}[]): RerankFn {
  return async (query, docs, topK, model) => {
    calls.push({ query, docs: [...docs], topK, model });
    const key = keyForText(query);
    const target = `src/file${key}.ts`;
    const targetIndex = docs.findIndex((d) => d.startsWith(`${target}\n`));
    if (targetIndex < 0) throw new Error(`missing target doc ${target}`);
    const rest = docs.map((_, index) => index).filter((i) => i !== targetIndex);
    const order =
      model === FULL_RERANK_MODEL
        ? [targetIndex, ...rest]
        : [rest[0], rest[1], targetIndex, ...rest.slice(2)];
    return order.slice(0, Math.min(topK, docs.length)).map((index, rank) => ({
      index,
      score: 1 - rank / 100,
    }));
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe.skipIf(!HAS_DB)(
  "bakeoff-rerank scoring (mocked Voyage, test DB)",
  () => {
    let pool: pg.Pool;
    let seeded = false;
    const projectId = `rerank-bakeoff-${Date.now()}`;
    const repositoryId = `${projectId}-repo`;
    const observedSql: string[] = [];

    beforeAll(async () => {
      pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 4,
      });
      const here = dirname(fileURLToPath(import.meta.url));
      const sqlDir = join(here, "..", "sql");
      await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
      for (const f of readdirSync(sqlDir)
        .filter((x) => x.endsWith(".sql"))
        .sort()) {
        // The rerank bakeoff targets the post-005 schema and must not resurrect the old
        // scratch column while bootstrapping this test's tables.
        if (f.startsWith("004_") || f.startsWith("005_")) continue;
        const sql = readFileSync(join(sqlDir, f), "utf8");
        if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
        await pool.query(sql);
      }
      await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
        projectId,
      ]);
      for (let i = 1; i <= 15; i++) {
        const path = `src/file${i}.ts`;
        const { rows } = await pool.query<{ id: string }>(
          `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
           VALUES ($1, $2, $3, 'typescript', $4) RETURNING id`,
          [repositoryId, projectId, path, `sha-${i}`],
        );
        await pool.query(
          `INSERT INTO codebase.code_chunks
             (file_id, repository_id, project_id, file_path, language, symbol_name,
              start_line, end_line, content, content_sha256, embedding)
           VALUES ($1, $2, $3, $4, 'typescript', NULL, 1, 3, $5, $6, $7::halfvec)`,
          [
            rows[0].id,
            repositoryId,
            projectId,
            path,
            `token K${i} function file${i} returns useful value`,
            `csha-${i}`,
            `[${oneHot(i).join(",")}]`,
          ],
        );
      }
      seeded = true;
    });

    afterAll(async () => {
      if (seeded) {
        await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
          projectId,
        ]);
      }
      await pool?.end();
    });

    function instrumentedPool(): pg.Pool {
      return {
        query: ((text: unknown, params?: unknown[]) => {
          observedSql.push(
            typeof text === "string"
              ? text
              : String((text as { text?: unknown }).text ?? ""),
          );
          return pool.query(text as any, params as any);
        }) as pg.Pool["query"],
      } as pg.Pool;
    }

    it("scores lite vs full with shared docs and hand-computed nDCG deltas", async () => {
      observedSql.length = 0;
      const calls: {
        query: string;
        docs: string[];
        topK: number;
        model: string;
      }[] = [];
      const embedCalls: string[] = [];
      const embedQuery: QueryEmbedder = async (q) => {
        embedCalls.push(q);
        return oneHot(keyForText(q));
      };

      const result = await runRerankBakeoff({
        pool: instrumentedPool(),
        evalFile: evalFile("test"),
        rerankFn: makeRerankMock(calls),
        embedQuery,
        projectId,
        repo: repositoryId,
      });

      expect(result.lite.aggregates.ndcgAt10).toBeCloseTo(0.5, 10);
      expect(result.full.aggregates.ndcgAt10).toBe(1);
      expect(result.lite.aggregates.recallAt25).toBe(1);
      expect(result.full.aggregates.recallAt25).toBe(1);
      expect(result.deltas).toEqual(Array(15).fill(0.5));
      expect(result.ci).toEqual(
        pairedBootstrapCI(result.deltas, { seed: 42 }),
      );
      expect(result.verdict).toBe("SWAP");

      expect(
        ndcgAtK(["x.ts", "y.ts", "src/file1.ts"], ["src/file1.ts"], 10),
      ).toBe(0.5);
      expect(embedCalls).toHaveLength(15);

      const byQuery = new Map<string, { lite?: string[]; full?: string[] }>();
      for (const c of calls) {
        const item = byQuery.get(c.query) ?? {};
        if (c.model === LITE_RERANK_MODEL) item.lite = c.docs;
        if (c.model === FULL_RERANK_MODEL) item.full = c.docs;
        byQuery.set(c.query, item);
      }
      expect(byQuery.size).toBe(15);
      for (const item of byQuery.values()) {
        expect(item.lite).toBeDefined();
        expect(item.full).toEqual(item.lite);
      }
      expect(new Set(calls.map((c) => c.model))).toEqual(
        new Set([LITE_RERANK_MODEL, FULL_RERANK_MODEL]),
      );
      expect(observedSql.join("\n")).not.toContain("embedding_ctx");
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });
  },
);

describe("bakeoff-rerank pure guards (no DB, no network)", () => {
  it("throws on a dev-only eval file before issuing SQL", async () => {
    const query = vi.fn();
    await expect(
      scoreRerankArm({
        pool: { query } as unknown as pg.Pool,
        evalFile: evalFile("dev"),
        model: LITE_RERANK_MODEL,
        rerankFn: vi.fn(),
        embedQuery: vi.fn(),
        projectId: "unused",
      }),
    ).rejects.toThrow("at least 15 test rows");
    expect(query).not.toHaveBeenCalled();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("mirrors runCodeEval's post-cut dedupe: a duplicate path burns its rank slot but cannot double-credit", async () => {
    // Pool has TWO chunks for a.ts (occupying the top two rank slots) then one for
    // b.ts, the only relevant path. Rerank returns the identity order (no reordering),
    // so the raw top-10 cut is [a.ts, a.ts, b.ts]. Hand-computed: dedupe-then-score
    // promotes b.ts to position 1 (0-based) -> nDCG@10 = 1/log2(3) / (1/log2(2)).
    const rows = [
      { file_path: "a.ts", content: "A", rrf: 0.9 },
      { file_path: "a.ts", content: "A2", rrf: 0.8 },
      { file_path: "b.ts", content: "B", rrf: 0.7 },
    ];
    const query = vi.fn(
      async () => ({ rows }) as unknown as QueryResult<any>,
    );
    const identityRerank: RerankFn = async (_query, docs, topK) =>
      docs
        .map((_, index) => ({ index, score: 1 - index / 100 }))
        .slice(0, topK);

    const dupEvalFile: CodeEvalFile = {
      ...evalFile("test"),
      rows: evalFile("test").rows.map((r) => ({ ...r, relevantPaths: ["b.ts"] })),
    };
    const result = await scoreRerankArm({
      pool: { query } as unknown as pg.Pool,
      evalFile: dupEvalFile,
      model: LITE_RERANK_MODEL,
      rerankFn: identityRerank,
      embedQuery: async () => [0],
      projectId: "unused",
    });

    const expectedNdcg = 1 / Math.log2(3) / (1 / Math.log2(2));
    for (const q of result.perQuery) {
      expect(q.hits).toEqual(["a.ts", "b.ts"]);
      expect(q.rank).toBe(2);
      expect(q.ndcgAt10).toBeCloseTo(expectedNdcg, 10);
    }
    // Without the post-cut dedupe, b.ts would stay at its raw index 2 -> nDCG 0.5.
    expect(result.perQuery[0].ndcgAt10).not.toBeCloseTo(0.5, 10);
  });

  it("throws loudly when the rerank result carries an out-of-range index", async () => {
    const rows = [{ file_path: "a.ts", content: "A", rrf: 0.9 }];
    const query = vi.fn(
      async () => ({ rows }) as unknown as QueryResult<any>,
    );
    const outOfRangeRerank: RerankFn = async () => [
      { index: 7, score: 1 },
    ];

    await expect(
      scoreRerankArm({
        pool: { query } as unknown as pg.Pool,
        evalFile: evalFile("test"),
        model: LITE_RERANK_MODEL,
        rerankFn: outOfRangeRerank,
        embedQuery: async () => [0],
        projectId: "unused",
      }),
    ).rejects.toThrow("out-of-range index");
  });

  it("throws before any pool/embed call when CANDIDATE_POOL mismatches the fixed pool metric", async () => {
    vi.resetModules();
    vi.stubEnv("CANDIDATE_POOL", "50");
    try {
      const { runRerankBakeoff } = await import("../src/db/bakeoff-rerank.js");
      const query = vi.fn();
      const embedQuery = vi.fn();
      await expect(
        runRerankBakeoff({
          pool: { query } as unknown as pg.Pool,
          evalFile: evalFile("test"),
          rerankFn: vi.fn(),
          embedQuery,
          projectId: "unused",
        }),
      ).rejects.toThrow("pool-recall cutoff mismatch");
      expect(query).not.toHaveBeenCalled();
      expect(embedQuery).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllEnvs();
      vi.resetModules();
    }
  });
});
