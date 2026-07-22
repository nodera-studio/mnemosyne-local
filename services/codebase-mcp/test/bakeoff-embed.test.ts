// AC-030/AC-031: the bake-off scratch-embed writes the embedding_ctx column (file-grouped,
// batched, resumable) and scoreArm computes Recall@k/MRR over the SCRATCH column through
// the FULL RRF+rerank pipeline — WITHOUT touching the live `embedding` column. DB-backed
// (disposable test DB) but the Voyage embedder + reranker are MOCKED via dependency
// injection / vi.mock — NO live quota, NO live data. Skipped gracefully without DATABASE_URL.

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const HAS_DB = !!process.env.DATABASE_URL;

// Mock the Voyage module so searchArm's rerank() is deterministic (identity: keep RRF
// order) and never hits the network. embedCode/embedCodeContextual are only used by the
// CLI path (not under test); we still stub them so importing the module is hermetic.
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0)),
  ),
  embedCodeContextual: vi.fn(async (docs: string[][]) =>
    docs.map((chunks) =>
      chunks.map(() => Array.from({ length: 1024 }, () => 0)),
    ),
  ),
  // Identity reranker: preserve the candidate (RRF) order so the test asserts the SQL
  // ranking, not a Voyage model. Returns {index, score} in input order.
  rerank: vi.fn(async (_q: string, documents: string[]) =>
    documents.map((_d, index) => ({ index, score: 1 - index / 1000 })),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

// Imported AFTER vi.mock so the mocked voyage module is in effect.
const { bakeoffEmbedContextual, countScratchPending, scoreArm } =
  await import("../src/db/bakeoff-embed.js");
import type {
  ContextualFileEmbedder,
  QueryEmbedder,
} from "../src/db/bakeoff-embed.js";
import type { CodeEvalFile } from "../src/recall-math.js";

// ── Deterministic geometry: embed text → a sparse "one-hot" 1024-vector keyed on a token.
// Cosine distance is minimized when query and chunk share the same key, so a query embeds
// nearest to the chunk whose content carries the matching token. Lets us assert recall
// deterministically with NO live Voyage. (int8-ish small integers; halfvec-safe.)
function oneHot(key: number): number[] {
  const v = Array.from({ length: 1024 }, () => 0);
  v[((key % 1024) + 1024) % 1024] = 100;
  return v;
}
function keyForText(t: string): number {
  // Hash the FIRST token (e.g. "K7") so seeded chunks and queries line up by key.
  const m = t.match(/K(\d+)/);
  return m ? Number(m[1]) : 0;
}

describe.skipIf(!HAS_DB)(
  "bakeoff scratch-embed + scoreArm (AC-030/AC-031, mocked Voyage, test DB)",
  () => {
    let pool: pg.Pool;
    const projectId = `bakeoff-test-${Date.now()}`;
    const repositoryId = `${projectId}-repo`;

    beforeAll(async () => {
      pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 4,
      });
      // Self-bootstrap the schema (incl. the 004 scratch column) so this file does not
      // depend on which DB-setup test ran before it. Apply migrations HOLD-aware — exactly
      // like the real runner — so 005 (the scratch drop) never fires here.
      const { readFileSync, readdirSync } = await import("node:fs");
      const { fileURLToPath } = await import("node:url");
      const { dirname, join } = await import("node:path");
      const sqlDir = join(dirname(fileURLToPath(import.meta.url)), "..", "sql");
      await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
      for (const f of readdirSync(sqlDir)
        .filter((x) => x.endsWith(".sql"))
        .sort()) {
        const sql = readFileSync(join(sqlDir, f), "utf8");
        if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
        await pool.query(sql);
      }
    });

    afterAll(async () => {
      // FK cascade from files → code_chunks removes the chunks too.
      await pool.query(`DELETE FROM codebase.files WHERE project_id = $1`, [
        projectId,
      ]);
      await pool.end();
    });

    // Seed N files, each with `chunksPerFile` chunks. Chunk content carries a unique token
    // "K<key>" so the one-hot geometry lines up. The incumbent `embedding` is seeded
    // one-hot (the incumbent arm is already embedded in the live column); embedding_ctx is
    // left NULL for the bake-off re-embed to fill.
    async function seedCorpus(
      nFiles: number,
      chunksPerFile: number,
    ): Promise<{ keys: number[] }> {
      const keys: number[] = [];
      let key = 1;
      for (let f = 0; f < nFiles; f++) {
        const { rows } = await pool.query<{ id: string }>(
          `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
           VALUES ($1,$2,$3,'typescript',$4) RETURNING id`,
          [repositoryId, projectId, `src/file${f}.ts`, `sha-${f}`],
        );
        const fileId = rows[0].id;
        for (let c = 0; c < chunksPerFile; c++) {
          const k = key++;
          keys.push(k);
          await pool.query(
            `INSERT INTO codebase.code_chunks
               (file_id,repository_id,project_id,file_path,language,start_line,end_line,content,content_sha256,embedding)
             VALUES ($1,$2,$3,$4,'typescript',$5,$6,$7,$8,$9::halfvec)`,
            [
              fileId,
              repositoryId,
              projectId,
              `src/file${f}.ts`,
              c * 10 + 1,
              c * 10 + 9,
              `token K${k} function doThing${k} returns value`,
              `csha-${k}`,
              `[${oneHot(k).join(",")}]`,
            ],
          );
        }
      }
      return { keys };
    }

    // A file-grouped MOCK embedder: one-hot per chunk by its K-token, recording the doc
    // (file) shapes so we can assert grouping + batching. NEVER touches the network.
    function mockFileEmbedder(): {
      fn: ContextualFileEmbedder;
      requests: number[][];
    } {
      const requests: number[][] = [];
      const fn: ContextualFileEmbedder = async (docs) => {
        requests.push(docs.map((d) => d.length)); // chunk counts per file in this request
        return docs.map((chunks) => chunks.map((c) => oneHot(keyForText(c))));
      };
      return { fn, requests };
    }

    it("re-embeds the corpus into embedding_ctx, FILE-GROUPED and batched, leaving `embedding` untouched", async () => {
      const { keys } = await seedCorpus(3, 2); // 3 files × 2 chunks = 6 chunks

      expect(await countScratchPending(pool, projectId, repositoryId)).toBe(6);

      const { fn, requests } = mockFileEmbedder();
      const res = await bakeoffEmbedContextual({
        pool,
        embed: fn,
        projectId,
        repo: repositoryId,
        filesPerRequest: 2, // force ≥2 requests so batching is exercised
        log: () => {},
      });

      expect(res.files).toBe(3);
      expect(res.chunks).toBe(6);
      // 3 files at 2 files/request → requests of [2 files, 1 file]
      expect(res.requests).toBe(2);
      // each request is a NESTED file→chunks shape; every file had 2 chunks
      expect(requests).toEqual([[2, 2], [2]]);

      // every chunk now has embedding_ctx; none pending
      expect(await countScratchPending(pool, projectId, repositoryId)).toBe(0);
      const { rows } = await pool.query<{ ctx: number; emb: number }>(
        `SELECT count(*) FILTER (WHERE embedding_ctx IS NOT NULL) AS ctx,
                count(*) FILTER (WHERE embedding IS NOT NULL) AS emb
         FROM codebase.code_chunks WHERE project_id = $1`,
        [projectId],
      );
      expect(Number(rows[0].ctx)).toBe(6);
      // the live `embedding` column is INTACT (non-destructive requirement)
      expect(Number(rows[0].emb)).toBe(6);
      // sanity: keys exist
      expect(keys.length).toBe(6);
    });

    it("is RESUMABLE — a second run embeds nothing (no chunks left pending)", async () => {
      const { fn, requests } = mockFileEmbedder();
      const res = await bakeoffEmbedContextual({
        pool,
        embed: fn,
        projectId,
        repo: repositoryId,
        log: () => {},
      });
      expect(res.files).toBe(0);
      expect(res.chunks).toBe(0);
      expect(res.requests).toBe(0);
      expect(requests).toEqual([]); // embedder never called again
    });

    it("scoreArm computes Recall@k/MRR over the SCRATCH column via the full pipeline", async () => {
      // Eval set: each query targets the file holding a known K-token. With one-hot
      // geometry the matching chunk is the nearest neighbor, so the relevant file lands
      // at/near rank 1 → Recall@10 should be 1 and MRR high.
      const evalFile: CodeEvalFile = {
        _seed: true,
        k: 10,
        rows: [
          { query: "K1", relevantPaths: ["src/file0.ts"] },
          { query: "K3", relevantPaths: ["src/file1.ts"] },
          { query: "K5", relevantPaths: ["src/file2.ts"] },
        ],
      };
      // Query embedder for the CTX arm: one-hot on the query's K-token (no live Voyage).
      const ctxQuery: QueryEmbedder = async (q) => oneHot(keyForText(q));

      const result = await scoreArm({
        pool,
        evalFile,
        column: "embedding_ctx",
        embedQuery: ctxQuery,
        projectId,
        repo: repositoryId,
      });

      expect(result.recallAtK).toBe(1); // every relevant file surfaced in top-10
      expect(result.mrr).toBeGreaterThan(0);
      expect(result.perQuery).toHaveLength(3);
      for (const q of result.perQuery) {
        expect(q.recall).toBe(1);
        expect(q.rank).not.toBeNull();
      }
    });

    it("scoreArm against the incumbent `embedding` column also resolves (apples-to-apples path)", async () => {
      const evalFile: CodeEvalFile = {
        _seed: true,
        k: 10,
        rows: [{ query: "K1", relevantPaths: ["src/file0.ts"] }],
      };
      const codeQuery: QueryEmbedder = async (q) => oneHot(keyForText(q));
      const result = await scoreArm({
        pool,
        evalFile,
        column: "embedding",
        embedQuery: codeQuery,
        projectId,
        repo: repositoryId,
      });
      expect(result.recallAtK).toBe(1);
    });

    it("countScratchPending never throws and reflects the project/repo scope", async () => {
      const n = await countScratchPending(pool, projectId, repositoryId);
      expect(typeof n).toBe("number");
      expect(n).toBe(0); // all embedded by now
    });
  },
);

// A pure (no-DB) guard so the file always has a running assertion in infra-less CI.
describe("bakeoff one-hot geometry (no network)", () => {
  it("oneHot keys map query and chunk to the same basis vector", () => {
    expect(keyForText("K7 function")).toBe(7);
    const v = oneHot(7);
    expect(v[7]).toBe(100);
    expect(v).toHaveLength(1024);
    expect(v.filter((x) => x !== 0)).toHaveLength(1);
  });
});
