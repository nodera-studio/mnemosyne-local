// Test-facing recall harness for the codebase-mcp embedder bake-off (AC-030/AC-031).
//
// The pure scoring math lives in ../src/recall-math.ts (so the operator-run bake-off
// module can import it under tsconfig's `rootDir: src`); this file RE-EXPORTS it plus the
// test-only bits: the eval-file loader and the DB-backed `runCodeRecallEval` runner that
// drives the live `searchCode`.
//
// Mirrors memory-mcp's recall.helper.ts. Code relevance is matched on FILE PATH (a query
// is "satisfied" if a relevant file appears in the top-k), since a query may map to any
// chunk of the right file.

import {
  filterRowsBySplit,
  firstRelevantRank,
  meanRecallAtK,
  meanReciprocalRank,
  mrr,
  ndcgAtK,
  normalizePath,
  POOL_RECALL_K,
  recallAtK,
  type CodeEvalFile,
  type CodeRecallEvalResult,
  type RankedPaths,
  type RelevantPaths,
  type RunArtifact,
  type RunPerQuery,
} from "../src/recall-math.js";

// Re-export the pure math + types + loader so test imports (and the bakeoff harness docs)
// can pull everything from one place.
export {
  recallAtK,
  mrr,
  meanRecallAtK,
  meanReciprocalRank,
  ndcgAtK,
  mulberry32,
  pairedBootstrapCI,
  regressionExcluded,
  normalizePath,
  POOL_RECALL_K,
  firstRelevantRank,
  loadCodeEval,
  filterRowsBySplit,
  writeRunArtifact,
  perQueryDeltas,
  type RankedPaths,
  type RelevantPaths,
  type CodeEvalRow,
  type CodeEvalFile,
  type CodeRecallEvalResult,
  type RunArtifact,
  type RunPerQuery,
} from "../src/recall-math.js";

/**
 * DB-backed runner: embeds + searches each seed query via the live `searchCode`, then
 * scores the ranked file paths against `relevantPaths`. Requires DATABASE_URL +
 * VOYAGE_API_KEY. `searchCode` is imported lazily so the pure-function tests do not pull
 * in the pg pool / Voyage client when no DB is configured.
 *
 * Scores the LIVE search path (whichever column searchCode reads). The bake-off's per-arm
 * scoring against the SCRATCH column lives in `src/db/bakeoff-embed.ts` (`scoreArm`),
 * which parametrizes the embedding column so both arms run the SAME pipeline.
 */
export async function runCodeRecallEval(
  evalFile: CodeEvalFile,
  opts: { projectId: string; repo?: string; limit?: number },
): Promise<CodeRecallEvalResult> {
  const { searchCode } = await import("../src/search.js");
  const k = evalFile.k;
  const limit = opts.limit ?? Math.max(k, 10);

  const perQuery: {
    hits: RankedPaths;
    relevant: RelevantPaths;
    query: string;
  }[] = [];
  for (const row of evalFile.rows) {
    const result = await searchCode({
      projectId: opts.projectId,
      query: row.query,
      repo: opts.repo,
      k: limit,
    });
    perQuery.push({
      query: row.query,
      hits: result.hits.map((h) => h.filePath),
      relevant: row.relevantPaths,
    });
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

// ── Two-layer eval runner (wave-2, AC-104) ───────────────────────────────────────────

/** Final-layer nDCG/MRR cutoff. */
export const FINAL_NDCG_K = 10;

/**
 * v2 two-layer runner, path-keyed (a path IS the stable code gold id — no supersession
 * resolution, paths do not supersede):
 *
 *  - POOL layer — `fuseCodeCandidates` directly; Recall@25 over candidate `file_path`s
 *    (the reranker cannot fix recall, so this floor is scored where recall happens).
 *  - FINAL layer — the full `searchCode` pipeline; nDCG@10 + MRR over hit paths.
 *
 * Returns the complete artifact payload (retrievalConfig() snapshot included) ready for
 * `writeRunArtifact`. Requires DATABASE_URL + a (real or mocked) Voyage; `search.js` is
 * imported lazily so pure tests never pull the pg pool / Voyage client.
 */
export async function runCodeEval(
  evalFile: CodeEvalFile,
  opts: { projectId: string; repo?: string; split?: "dev" | "test" },
): Promise<RunArtifact> {
  const { fuseCodeCandidates, searchCode, retrievalConfig } =
    await import("../src/search.js");
  const snapshot = retrievalConfig();
  // The pool-layer metric NAME is fixed (Recall@25) while config.candidatePool is
  // env-variable — refuse to record a mislabeled number rather than silently scoring a
  // different cutoff (the cutoff is deliberately NOT dynamic; the metric must stay
  // comparable across runs).
  if (snapshot.candidatePool !== POOL_RECALL_K) {
    throw new Error(
      `pool-recall cutoff mismatch: config.candidatePool = ${snapshot.candidatePool} ` +
        `but the pool metric is fixed at Recall@${POOL_RECALL_K} — unset CANDIDATE_POOL ` +
        `(or restore ${POOL_RECALL_K}) before running the eval.`,
    );
  }
  const rows = opts.split
    ? filterRowsBySplit(evalFile, opts.split)
    : evalFile.rows;

  const perQuery: RunPerQuery[] = [];
  for (const row of rows) {
    // Layer 1: candidate pool only.
    const pool = await fuseCodeCandidates({
      projectId: opts.projectId,
      query: row.query,
      repo: opts.repo,
    });
    const poolPaths = pool.map((c) => c.file_path);
    const poolRecall = recallAtK(poolPaths, row.relevantPaths, POOL_RECALL_K);

    // Layer 2: the full live pipeline. Hits are CHUNK-keyed, so the same file path can
    // appear more than once in the top-k — dedupe to the FIRST occurrence before the
    // path-keyed scoring (a duplicate is not a second relevant document; letting it
    // through both inflated nDCG past 1.0 and buried later distinct paths).
    const result = await searchCode({
      projectId: opts.projectId,
      query: row.query,
      repo: opts.repo,
      k: FINAL_NDCG_K,
    });
    const hits = result.hits;
    const seen = new Set<string>();
    const finalPaths = hits
      .map((h) => h.filePath)
      .filter((p) => {
        const key = normalizePath(p);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });

    perQuery.push({
      id: row.id ?? row.query,
      query: row.query,
      poolRecall,
      ndcg: ndcgAtK(finalPaths, row.relevantPaths, FINAL_NDCG_K),
      mrr: mrr(finalPaths, row.relevantPaths),
      rank: firstRelevantRank(finalPaths, row.relevantPaths),
    });
  }

  const n = perQuery.length;
  const mean = (f: (q: RunPerQuery) => number) =>
    n === 0 ? 0 : perQuery.reduce((a, q) => a + f(q), 0) / n;
  return {
    retrievalConfig: snapshot,
    evalVersion: evalFile.version ?? 1,
    split: opts.split ?? "all",
    rows: n,
    perQuery,
    aggregates: {
      recallAt25: mean((q) => q.poolRecall),
      ndcgAt10: mean((q) => q.ndcg),
      mrr10: mean((q) => q.mrr),
    },
  };
}
