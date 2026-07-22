// Wave 5 bake-off: scratch-embed + per-arm scoring for the voyage-code-3 vs
// voyage-context-3 code embedder decision (AC-030/AC-031).
//
// NON-DESTRUCTIVE by design: the contextual arm is written into the SCRATCH column
// codebase.code_chunks.embedding_ctx (added by sql/004_bakeoff_scratch.sql), NEVER the
// live `embedding`. The running code_search keeps reading `embedding` throughout, so
// search never breaks while the operator measures. The incumbent arm (voyage-code-3) is
// already in `embedding` from the normal index, so only the contextual arm needs a
// scratch re-embed.
//
// This is a DELIBERATE ops action that spends Voyage quota — it is NEVER run on container
// start. Gate: `npm run bakeoff`. The embedder is INJECTED (ContextualFileEmbedder) so the
// batching + grouping + SQL logic is unit-testable with a MOCK — no live quota in tests.
//
// Scoring (`scoreArm`) parametrizes the embedding column so BOTH arms run through the
// SAME RRF + rerank pipeline as the live searchCode — the embedder interacts with fusion,
// so raw cosine is the wrong yardstick (AC-030 gotcha).

import type { Pool } from "pg";
import { pool as defaultPool } from "./pool.js";
import {
  embedCode,
  embedCodeContextual,
  rerank,
  toVectorLiteral,
} from "../voyage.js";
import { config } from "../config.js";
import {
  firstRelevantRank,
  loadCodeEval,
  meanRecallAtK,
  meanReciprocalRank,
  recallAtK,
  type CodeEvalFile,
  type CodeRecallEvalResult,
  type RankedPaths,
  type RelevantPaths,
} from "../recall-math.js";

// ── caps mirrored from the contextual endpoint contract (voyage.ts docstring) ───────
// A file = one contextual "doc"; its chunks are the inner array. Keep each request well
// under the ≤1000 docs / ≤16k chunks per-request caps. The indexer's MAX_FILE_BYTES guard
// (400k) already bounds a single file's chunk count.
export const FILES_PER_REQUEST = 100;
export const MAX_CHUNKS_PER_REQUEST = 8000;

/** File-grouped contextual embedder signature — INJECTABLE so the batching/SQL logic is
 *  unit-testable with a MOCK (no live Voyage quota). Takes nested file→chunks docs and
 *  returns nested file→chunk→vector, preserving order (the embedCodeContextual shape). */
export type ContextualFileEmbedder = (
  docs: string[][],
  inputType: "document" | "query",
) => Promise<number[][][]>;

export interface BakeoffEmbedDeps {
  pool: Pool;
  embed: ContextualFileEmbedder;
  projectId: string;
  repo?: string;
  filesPerRequest?: number;
  maxChunksPerRequest?: number;
  /** Optional progress sink (defaults to stderr). */
  log?: (msg: string) => void;
}

export interface BakeoffEmbedResult {
  files: number;
  chunks: number;
  requests: number;
}

interface ChunkRow {
  id: string;
  file_id: string;
  content: string;
}

/** How many chunks still lack a scratch (embedding_ctx) vector — the completeness probe. */
export async function countScratchPending(
  pool: Pool,
  projectId: string,
  repo?: string,
): Promise<number> {
  const params: unknown[] = [projectId];
  let filt = "";
  if (repo) {
    params.push(repo);
    filt = ` AND repository_id = $${params.length}`;
  }
  const { rows } = await pool.query<{ n: string }>(
    `SELECT count(*) AS n FROM codebase.code_chunks
     WHERE project_id = $1 AND embedding_ctx IS NULL${filt}`,
    params,
  );
  return Number(rows[0].n);
}

/**
 * Re-embed the corpus through the CONTEXTUAL endpoint into the embedding_ctx scratch
 * column, FILE-GROUPED (each file's chunks embedded together so a chunk gets its file's
 * context — the lever where context-3 might beat the code specialist). RESUMABLE: it
 * selects only chunks where embedding_ctx IS NULL, grouped by file, so a crash mid-run
 * continues where it stopped. One request at a time (no batch-level concurrency) so we
 * never hammer the endpoint; the 429/Retry-After backoff lives in voyageFetch.
 */
export async function bakeoffEmbedContextual(
  deps: BakeoffEmbedDeps,
): Promise<BakeoffEmbedResult> {
  const { pool, embed, projectId } = deps;
  const filesPerRequest = deps.filesPerRequest ?? FILES_PER_REQUEST;
  const maxChunks = deps.maxChunksPerRequest ?? MAX_CHUNKS_PER_REQUEST;
  const log = deps.log ?? ((m: string) => console.error(m));

  const repoFilt: unknown[] = [projectId];
  let filt = "";
  if (deps.repo) {
    repoFilt.push(deps.repo);
    filt = ` AND repository_id = $${repoFilt.length}`;
  }

  let files = 0;
  let chunks = 0;
  let requests = 0;

  for (;;) {
    // Pull the next batch of files (each with all its still-pending chunks) ordered by
    // file_id so contextual grouping is stable and the run is resumable. A file is
    // "pending" if ANY of its chunks lack embedding_ctx; we re-embed all of that file's
    // chunks together (contextual embedding needs the whole file's chunk set).
    const { rows: fileRows } = await pool.query<{ file_id: string }>(
      `SELECT DISTINCT file_id FROM codebase.code_chunks
       WHERE project_id = $1 AND embedding_ctx IS NULL${filt}
       ORDER BY file_id
       LIMIT $${repoFilt.length + 1}`,
      [...repoFilt, filesPerRequest],
    );
    if (fileRows.length === 0) break;

    // Load each file's chunks IN ORDER (start_line) — contextual embedding is order-aware.
    const fileIds = fileRows.map((r) => r.file_id);
    const { rows: chunkRows } = await pool.query<ChunkRow>(
      `SELECT id, file_id, content FROM codebase.code_chunks
       WHERE file_id = ANY($1::uuid[])
       ORDER BY file_id, start_line`,
      [fileIds],
    );

    // Group chunks by file, preserving order. Cap the request by chunk count: if adding a
    // file would exceed maxChunks, defer it to the next iteration (it stays pending).
    const byFile = new Map<string, ChunkRow[]>();
    for (const c of chunkRows) {
      (byFile.get(c.file_id) ?? byFile.set(c.file_id, []).get(c.file_id)!).push(
        c,
      );
    }
    const docFileIds: string[] = [];
    const docs: string[][] = [];
    const docChunkIds: string[][] = [];
    let chunkBudget = 0;
    for (const fid of fileIds) {
      const cs = byFile.get(fid);
      if (!cs || cs.length === 0) continue;
      // A single file whose chunk count alone exceeds the cap still goes in (caller caps
      // file size via MAX_FILE_BYTES, so this is the only group); otherwise stop adding.
      if (docs.length > 0 && chunkBudget + cs.length > maxChunks) break;
      docFileIds.push(fid);
      docs.push(cs.map((c) => c.content));
      docChunkIds.push(cs.map((c) => c.id));
      chunkBudget += cs.length;
    }
    if (docs.length === 0) break; // defensive — should not happen given the SELECT

    const vectors = await embed(docs, "document");
    if (vectors.length !== docs.length) {
      throw new Error(
        `embedder returned ${vectors.length} file-vectors for ${docs.length} files`,
      );
    }

    for (let f = 0; f < docs.length; f++) {
      const ids = docChunkIds[f];
      const vecs = vectors[f];
      if (vecs.length !== ids.length) {
        throw new Error(
          `file ${docFileIds[f]}: embedder returned ${vecs.length} chunk-vectors for ${ids.length} chunks`,
        );
      }
      for (let i = 0; i < ids.length; i++) {
        await pool.query(
          `UPDATE codebase.code_chunks SET embedding_ctx = $1::halfvec WHERE id = $2`,
          [toVectorLiteral(vecs[i]), ids[i]],
        );
        chunks += 1;
      }
      files += 1;
    }
    requests += 1;
    log(
      `bakeoff embed: request ${requests} (+${docs.length} files) → ${files} files, ${chunks} chunks`,
    );
  }

  return { files, chunks, requests };
}

// ── per-arm scoring (full RRF + rerank pipeline against a chosen embedding column) ──

/** Which arm to score: the incumbent (`embedding`) or the contextual scratch column. */
export type EmbeddingColumn = "embedding" | "embedding_ctx";

/** Query embedder per arm — INJECTABLE so scoring is unit-testable with a MOCK. Must
 *  embed the QUERY with the SAME family as the corpus column (query/corpus-family rule):
 *  embedding → embedCode, embedding_ctx → embedCodeContextual (single-chunk doc). */
export type QueryEmbedder = (query: string) => Promise<number[]>;

const RECALL_LIMIT = 50;

interface ScoreRow {
  file_path: string;
  content: string;
  rrf: number;
}

/**
 * Score ONE query against ONE arm with the full pipeline: BM25 ⊕ vector (RRF over the
 * chosen `column`) → rerank. Returns the ranked file paths (best-first). This mirrors
 * `searchCode` exactly except the vector column is parametrized — so both arms are
 * apples-to-apples and the live search path is untouched.
 */
export async function searchArm(
  pool: Pool,
  opts: {
    projectId: string;
    query: string;
    qvec: number[];
    column: EmbeddingColumn;
    repo?: string;
    k: number;
  },
): Promise<RankedPaths> {
  const params: unknown[] = [
    opts.query,
    toVectorLiteral(opts.qvec),
    opts.projectId,
    config.candidatePool,
  ];
  let filt = "";
  if (opts.repo) {
    params.push(opts.repo);
    filt += ` AND c.repository_id = $${params.length}`;
  }
  // `column` is a fixed enum literal (never user input) — safe to interpolate.
  const col = opts.column === "embedding_ctx" ? "embedding_ctx" : "embedding";
  const sql = `
    WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),
    bm25 AS (
      SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.search_tsv, q.tsq) DESC) AS rank
      FROM codebase.code_chunks c, q
      WHERE c.project_id = $3 AND c.search_tsv @@ q.tsq ${filt}
      LIMIT ${RECALL_LIMIT}
    ),
    vec AS (
      SELECT c.id, row_number() OVER (ORDER BY c.${col} <=> $2::halfvec) AS rank
      FROM codebase.code_chunks c
      WHERE c.project_id = $3 AND c.${col} IS NOT NULL ${filt}
      ORDER BY c.${col} <=> $2::halfvec
      LIMIT ${RECALL_LIMIT}
    )
    SELECT c.file_path, c.content,
           COALESCE(1.0/(60+bm25.rank),0) + COALESCE(1.0/(60+vec.rank),0) AS rrf
    FROM codebase.code_chunks c
    LEFT JOIN bm25 ON bm25.id = c.id
    LEFT JOIN vec  ON vec.id  = c.id
    WHERE bm25.id IS NOT NULL OR vec.id IS NOT NULL
    ORDER BY rrf DESC
    LIMIT $4`;

  const { rows } = await pool.query<ScoreRow>(sql, params);
  if (rows.length === 0) return [];

  const docs = rows.map((r) => `${r.file_path}\n${r.content}`.slice(0, 1500));
  const ranked = await rerank(opts.query, docs, rows.length);
  const order = ranked.length ? ranked : rows.map((_, i) => ({ index: i }));
  return order.slice(0, opts.k).map(({ index }) => rows[index].file_path);
}

export interface ScoreArmDeps {
  pool: Pool;
  evalFile: CodeEvalFile;
  column: EmbeddingColumn;
  embedQuery: QueryEmbedder;
  projectId: string;
  repo?: string;
  limit?: number;
}

/**
 * Score a whole eval set against one arm. Embeds each query (via the injected
 * `embedQuery` matching the arm's family), runs `searchArm`, and aggregates Recall@k +
 * MRR. The reusable measurement primitive behind both `npm run bakeoff` and the
 * regression-guard test.
 */
export async function scoreArm(
  deps: ScoreArmDeps,
): Promise<CodeRecallEvalResult> {
  const k = deps.evalFile.k;
  const limit = deps.limit ?? Math.max(k, 10);

  const perQuery: {
    hits: RankedPaths;
    relevant: RelevantPaths;
    query: string;
  }[] = [];
  for (const row of deps.evalFile.rows) {
    const qvec = await deps.embedQuery(row.query);
    const hits = await searchArm(deps.pool, {
      projectId: deps.projectId,
      query: row.query,
      qvec,
      column: deps.column,
      repo: deps.repo,
      k: limit,
    });
    perQuery.push({ query: row.query, hits, relevant: row.relevantPaths });
  }

  return {
    recallAtK: meanRecallAtK(perQuery, k),
    mrr: meanReciprocalRank(perQuery),
    perQuery: perQuery.map((q) => ({
      query: q.query,
      rank: firstRelevantRank(q.hits, q.relevant),
      recall: recallAtK(q.hits, q.relevant, k),
    })),
  };
}

// ── live query embedders (used by the CLI, NOT by the mocked tests) ─────────────────

/** Incumbent arm: embed the query with voyage-code-3 (one vector). */
const liveCodeQuery: QueryEmbedder = async (q) =>
  (await embedCode([q], "query"))[0];

/** Contextual arm: embed the query with voyage-context-3 as a single-chunk doc. */
const liveContextQuery: QueryEmbedder = async (q) => {
  const nested = await embedCodeContextual([[q]], "query");
  return nested[0][0];
};

// ── CLI entrypoint (npm run bakeoff) ────────────────────────────────────────────────
// Skipped when imported (tests import the functions directly). Spends Voyage quota.
//   1. scratch-embed the contextual arm into embedding_ctx (resumable),
//   2. score BOTH arms over the eval set with the full pipeline,
//   3. print Recall@10 + MRR@10 for each — the operator records them in test/bakeoff.md.
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  (async () => {
    const projectId = process.argv[2] ?? config.defaultProjectId;
    const repo = process.argv[3];
    const evalFile = loadCodeEval();

    const pending = await countScratchPending(defaultPool, projectId, repo);
    console.error(
      `bakeoff: ${pending} chunks missing embedding_ctx (contextual arm) for project ${projectId}${repo ? ` repo ${repo}` : ""}`,
    );
    const embedRes = await bakeoffEmbedContextual({
      pool: defaultPool,
      embed: embedCodeContextual,
      projectId,
      repo,
    });
    console.error(
      `bakeoff: contextual arm embedded — ${embedRes.files} files, ${embedRes.chunks} chunks, ${embedRes.requests} requests`,
    );

    const incumbent = await scoreArm({
      pool: defaultPool,
      evalFile,
      column: "embedding",
      embedQuery: liveCodeQuery,
      projectId,
      repo,
    });
    const contextual = await scoreArm({
      pool: defaultPool,
      evalFile,
      column: "embedding_ctx",
      embedQuery: liveContextQuery,
      projectId,
      repo,
    });

    const fmt = (r: CodeRecallEvalResult) =>
      `Recall@${evalFile.k}=${r.recallAtK.toFixed(4)}  MRR@${evalFile.k}=${r.mrr.toFixed(4)}`;
    console.error("");
    console.error("=== BAKE-OFF RESULT (record these in test/bakeoff.md) ===");
    console.error(`voyage-code-3    (embedding)     ${fmt(incumbent)}`);
    console.error(`voyage-context-3 (embedding_ctx) ${fmt(contextual)}`);
    const winner =
      contextual.recallAtK > incumbent.recallAtK
        ? "voyage-context-3"
        : "voyage-code-3 (incumbent kept — no recall gain)";
    console.error(`provisional winner: ${winner}`);
    console.error(
      "NOTE: if context-3 wins, RE-TUNE RRF + rerank on the eval set BEFORE flipping (AC-031).",
    );

    await defaultPool.end();
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
