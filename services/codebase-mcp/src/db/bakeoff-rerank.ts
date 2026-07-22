// Wave 6 bake-off: rerank-2.5-lite vs rerank-2.5 over the frozen TEST split.
//
// This measures ONLY the reranker swap. Candidate generation is shared by both arms:
// one query embedding, one live-embedding RRF pool, then two rerank calls over identical
// docs. The score is rank-based nDCG@10, so raw reranker scores are intentionally never
// compared across models.

import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { Pool } from "pg";
import { pool as defaultPool } from "./pool.js";
import { config } from "../config.js";
import { embedCode, rerank, toVectorLiteral } from "../voyage.js";
import {
  RECALL_LIMIT,
  RERANK_DOC_TRUNCATION,
  RRF_K,
  retrievalConfig,
} from "../search.js";
import {
  filterRowsBySplit,
  firstRelevantRank,
  loadCodeEval,
  ndcgAtK,
  normalizePath,
  pairedBootstrapCI,
  POOL_RECALL_K,
  recallAtK,
  type CodeEvalFile,
  type CodeEvalRow,
  type RankedPaths,
  type RelevantPaths,
} from "../recall-math.js";

export const LITE_RERANK_MODEL = "rerank-2.5-lite";
export const FULL_RERANK_MODEL = "rerank-2.5";
export const MIN_TEST_ROWS = 15;

export type RerankFn = (
  query: string,
  docs: string[],
  topK: number,
  model: string,
) => Promise<Array<{ index: number; score: number }>>;

export type QueryEmbedder = (query: string) => Promise<number[]>;

interface CandidateRow {
  file_path: string;
  content: string;
  rrf: number;
}

interface CandidateSet {
  docs: string[];
  poolPaths: RankedPaths;
}

export interface RerankPerQueryScore {
  id: string;
  query: string;
  relevantPaths: RelevantPaths;
  poolRecallAt25: number;
  ndcgAt10: number;
  rank: number | null;
  hits: RankedPaths;
}

export interface RerankArmScore {
  model: string;
  rows: number;
  aggregates: {
    recallAt25: number;
    ndcgAt10: number;
  };
  perQuery: RerankPerQueryScore[];
}

export interface ScoreRerankArmDeps {
  pool: Pool;
  evalFile: CodeEvalFile;
  model: string;
  rerankFn: RerankFn;
  embedQuery: QueryEmbedder;
  projectId: string;
  repo?: string;
  log?: (msg: string) => void;
  queryVectors?: Map<string, number[]>;
  candidates?: Map<string, CandidateSet>;
}

export interface RerankBakeoffResult {
  lite: RerankArmScore;
  full: RerankArmScore;
  deltas: number[];
  ci: { mean: number; ciLow: number; ciHigh: number };
  verdict: "SWAP" | "KEEP LITE" | "NO SIGNIFICANT DIFFERENCE — operator's call, cost immaterial";
}

export interface RunRerankBakeoffDeps
  extends Omit<ScoreRerankArmDeps, "model" | "queryVectors" | "candidates"> {
  liteModel?: string;
  fullModel?: string;
}

function testRows(evalFile: CodeEvalFile): CodeEvalRow[] {
  const rows = filterRowsBySplit(evalFile, "test");
  if (rows.length < MIN_TEST_ROWS) {
    throw new Error(
      `rerank bakeoff needs at least ${MIN_TEST_ROWS} test rows; found ${rows.length}. Check the frozen eval split before running.`,
    );
  }
  return rows;
}

function rowKey(row: CodeEvalRow, index: number): string {
  return row.id ?? `${index}:${row.query}`;
}

async function queryVector(
  row: CodeEvalRow,
  index: number,
  deps: Pick<ScoreRerankArmDeps, "embedQuery" | "queryVectors">,
): Promise<number[]> {
  const key = rowKey(row, index);
  const cached = deps.queryVectors?.get(key);
  if (cached) return cached;
  const v = await deps.embedQuery(row.query);
  deps.queryVectors?.set(key, v);
  return v;
}

async function loadCandidates(
  deps: Pick<ScoreRerankArmDeps, "pool" | "projectId" | "repo">,
  query: string,
  qvec: number[],
): Promise<CandidateSet> {
  const params: unknown[] = [
    query,
    toVectorLiteral(qvec),
    deps.projectId,
    config.candidatePool,
  ];
  let filt = "";
  if (deps.repo) {
    params.push(deps.repo);
    filt += ` AND c.repository_id = $${params.length}`;
  }

  // Fixed to the live `embedding` column: Wave 6 varies only the rerank model, and the
  // old bakeoff scratch `embedding_ctx` column was dropped by migration 005.
  const sql = `
    WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),
    bm25 AS (
      SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.search_tsv, q.tsq) DESC) AS rank
      FROM codebase.code_chunks c, q
      WHERE c.project_id = $3 AND c.search_tsv @@ q.tsq ${filt}
      LIMIT ${RECALL_LIMIT}
    ),
    vec AS (
      SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> $2::halfvec) AS rank
      FROM codebase.code_chunks c
      WHERE c.project_id = $3 AND c.embedding IS NOT NULL ${filt}
      ORDER BY c.embedding <=> $2::halfvec
      LIMIT ${RECALL_LIMIT}
    )
    SELECT c.file_path, c.content,
           (COALESCE(1.0/(${RRF_K}+bm25.rank),0) + COALESCE(1.0/(${RRF_K}+vec.rank),0))::float8 AS rrf
    FROM codebase.code_chunks c
    LEFT JOIN bm25 ON bm25.id = c.id
    LEFT JOIN vec  ON vec.id  = c.id
    WHERE bm25.id IS NOT NULL OR vec.id IS NOT NULL
    ORDER BY rrf DESC
    LIMIT $4`;

  const { rows } = await deps.pool.query<CandidateRow>(sql, params);
  return {
    docs: rows.map((r) =>
      `${r.file_path}\n${r.content}`.slice(0, RERANK_DOC_TRUNCATION),
    ),
    poolPaths: rows.map((r) => r.file_path),
  };
}

async function candidatesForRow(
  row: CodeEvalRow,
  index: number,
  deps: Pick<
    ScoreRerankArmDeps,
    "pool" | "projectId" | "repo" | "embedQuery" | "queryVectors" | "candidates"
  >,
): Promise<CandidateSet> {
  const key = rowKey(row, index);
  const cached = deps.candidates?.get(key);
  if (cached) return cached;
  const qvec = await queryVector(row, index, deps);
  const loaded = await loadCandidates(deps, row.query, qvec);
  deps.candidates?.set(key, loaded);
  return loaded;
}

function rankPaths(
  query: string,
  docs: string[],
  poolPaths: RankedPaths,
  model: string,
  rerankFn: RerankFn,
): Promise<RankedPaths> {
  if (docs.length === 0) return Promise.resolve([]);
  return rerankFn(query, docs, docs.length, model).then((ranked) => {
    const order = ranked.length
      ? ranked
      : docs.map((_, index) => ({ index, score: 0 }));
    return order.map(({ index }) => {
      const path = poolPaths[index];
      if (path === undefined) {
        throw new Error(
          `rerank bakeoff: rerank returned out-of-range index ${index} for a pool of ${poolPaths.length} paths`,
        );
      }
      return path;
    });
  });
}

/** First-occurrence dedupe by normalized path — mirrors runCodeEval's post-cut dedupe
 * (test/code-eval.helper.ts): a duplicate still burns the rank slot it occupied but
 * cannot double-credit or push a later distinct path out of the cut. */
function dedupeFirstOccurrence(paths: RankedPaths): RankedPaths {
  const seen = new Set<string>();
  return paths.filter((p) => {
    const key = normalizePath(p);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function scoreRerankArm(
  deps: ScoreRerankArmDeps,
): Promise<RerankArmScore> {
  const rows = testRows(deps.evalFile);
  deps.log?.(`scoring ${deps.model} over ${rows.length} frozen test rows`);

  const perQuery: RerankPerQueryScore[] = [];
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const candidates = await candidatesForRow(row, i, deps);
    const ranked = await rankPaths(
      row.query,
      candidates.docs,
      candidates.poolPaths,
      deps.model,
      deps.rerankFn,
    );
    // Mirror runCodeEval EXACTLY: cut to the top-10 BEFORE dedupe, so a duplicate path
    // still burns a rank slot, then dedupe first-occurrence so it can't double-credit.
    const hits = dedupeFirstOccurrence(ranked.slice(0, 10));
    perQuery.push({
      id: rowKey(row, i),
      query: row.query,
      relevantPaths: row.relevantPaths,
      poolRecallAt25: recallAtK(candidates.poolPaths, row.relevantPaths, POOL_RECALL_K),
      ndcgAt10: ndcgAtK(hits, row.relevantPaths, 10),
      rank: firstRelevantRank(hits, row.relevantPaths),
      hits,
    });
  }

  return {
    model: deps.model,
    rows: rows.length,
    aggregates: {
      recallAt25:
        perQuery.reduce((acc, q) => acc + q.poolRecallAt25, 0) /
        perQuery.length,
      ndcgAt10:
        perQuery.reduce((acc, q) => acc + q.ndcgAt10, 0) / perQuery.length,
    },
    perQuery,
  };
}

export async function runRerankBakeoff(
  deps: RunRerankBakeoffDeps,
): Promise<RerankBakeoffResult> {
  // The pool metric is fixed at Recall@25 (see runCodeEval, test/code-eval.helper.ts) —
  // refuse to score a mislabeled number if CANDIDATE_POOL drifts from that.
  if (config.candidatePool !== POOL_RECALL_K) {
    throw new Error(
      `pool-recall cutoff mismatch: config.candidatePool = ${config.candidatePool} ` +
        `but the pool metric is fixed at Recall@${POOL_RECALL_K} — unset CANDIDATE_POOL ` +
        `(or restore ${POOL_RECALL_K}) before running the bakeoff.`,
    );
  }
  const queryVectors = new Map<string, number[]>();
  const candidates = new Map<string, CandidateSet>();
  const liteModel = deps.liteModel ?? LITE_RERANK_MODEL;
  const fullModel = deps.fullModel ?? FULL_RERANK_MODEL;

  const lite = await scoreRerankArm({
    ...deps,
    model: liteModel,
    queryVectors,
    candidates,
  });
  const full = await scoreRerankArm({
    ...deps,
    model: fullModel,
    queryVectors,
    candidates,
  });

  const fullById = new Map(full.perQuery.map((q) => [q.id, q]));
  const deltas = lite.perQuery.map((q) => {
    const f = fullById.get(q.id);
    if (!f) throw new Error(`rerank bakeoff id drift: missing ${q.id}`);
    return f.ndcgAt10 - q.ndcgAt10;
  });
  const ci = pairedBootstrapCI(deltas, { seed: 42 });
  const verdict =
    ci.mean > 0 && ci.ciLow > 0
      ? "SWAP"
      : ci.mean < 0 && ci.ciHigh < 0
        ? "KEEP LITE"
        : "NO SIGNIFICANT DIFFERENCE — operator's call, cost immaterial";
  return { lite, full, deltas, ci, verdict };
}

export function bakeoffArtifact(result: RerankBakeoffResult) {
  const fullById = new Map(result.full.perQuery.map((q) => [q.id, q]));
  return {
    retrievalConfig: retrievalConfig(),
    evalVersion: 2,
    split: "test",
    rows: result.lite.rows,
    models: {
      lite: result.lite.model,
      full: result.full.model,
    },
    aggregates: {
      lite: result.lite.aggregates,
      full: result.full.aggregates,
    },
    pairedBootstrap: result.ci,
    verdict: result.verdict,
    perQuery: result.lite.perQuery.map((lite, i) => {
      const full = fullById.get(lite.id);
      if (!full) throw new Error(`rerank bakeoff id drift: missing ${lite.id}`);
      return {
        id: lite.id,
        query: lite.query,
        relevantPaths: lite.relevantPaths,
        lite: {
          poolRecallAt25: lite.poolRecallAt25,
          ndcgAt10: lite.ndcgAt10,
          rank: lite.rank,
          hits: lite.hits,
        },
        full: {
          poolRecallAt25: full.poolRecallAt25,
          ndcgAt10: full.ndcgAt10,
          rank: full.rank,
          hits: full.hits,
        },
        fullMinusLite: result.deltas[i],
      };
    }),
  };
}

export function writeBakeoffArtifact(
  result: RerankBakeoffResult,
  dir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "test", "runs"),
): string {
  mkdirSync(dir, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  const path = join(dir, `${date}-bakeoff-rerank.json`);
  writeFileSync(path, JSON.stringify(bakeoffArtifact(result), null, 2) + "\n");
  return path;
}

function printResult(
  result: RerankBakeoffResult,
  artifactPath: string,
  log: (msg: string) => void,
) {
  const fmt = (arm: RerankArmScore) =>
    `nDCG@10=${arm.aggregates.ndcgAt10.toFixed(4)}  Recall@25=${arm.aggregates.recallAt25.toFixed(4)}`;
  log("");
  log("=== RERANK BAKEOFF RESULT (record in test/bakeoff.md) ===");
  log(`${result.lite.model}  ${fmt(result.lite)}`);
  log(`${result.full.model}       ${fmt(result.full)}`);
  log(
    `full-minus-lite nDCG@10: mean=${result.ci.mean.toFixed(4)}  95% CI=[${result.ci.ciLow.toFixed(4)}, ${result.ci.ciHigh.toFixed(4)}]`,
  );
  log(`verdict: ${result.verdict}`);
  log(`artifact: ${artifactPath}`);
}

function parseArgs(argv: string[]): {
  yes: boolean;
  projectId: string;
  repo?: string;
} {
  const yes = argv.includes("--yes");
  const positional = argv.filter((a) => !a.startsWith("--"));
  return {
    yes,
    projectId: positional[0] ?? config.defaultProjectId,
    repo: positional[1],
  };
}

const liveEmbedQuery: QueryEmbedder = async (q) =>
  (await embedCode([q], "query"))[0];

const liveRerank: RerankFn = (query, docs, topK, model) =>
  rerank(query, docs, topK, model);

// ── CLI entrypoint (npm run bakeoff:rerank -- --yes) ───────────────────────────────
// Skipped when imported. Spends Voyage quota only after the explicit --yes gate.
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  (async () => {
    const { yes, projectId, repo } = parseArgs(process.argv.slice(2));
    const evalFile = loadCodeEval();
    const n = filterRowsBySplit(evalFile, "test").length;
    if (!yes) {
      console.error(
        `bakeoff:rerank is PAID and operator-only: 2 rerank calls/query × ${n} test queries (pennies). Re-run with --yes to proceed.`,
      );
      process.exit(1);
    }
    const result = await runRerankBakeoff({
      pool: defaultPool,
      evalFile,
      rerankFn: liveRerank,
      embedQuery: liveEmbedQuery,
      projectId,
      repo,
      log: (m) => console.error(m),
    });
    const artifactPath = writeBakeoffArtifact(result);
    printResult(result, artifactPath, (m) => console.error(m));
    await defaultPool.end();
  })().catch(async (e) => {
    console.error(e);
    await defaultPool.end().catch(() => {});
    process.exit(1);
  });
}
