// Recall harness for the memory-mcp embedder-flip gate (AC-023/AC-024).
//
// Two layers:
//  1. PURE math — `recallAtK`, `mrr`, and the Wave-1 eval additions `ndcgAtK` +
//     `pairedBootstrapCI` (+ `mulberry32`, `regressionExcluded`, wave-7's `signTest`).
//     These ARE the flip/regression gates' math; a bug here silently passes a bad
//     flip, so they are unit-tested directly against hand-computed values in
//     `recall.helper.test.ts`. The gate math + the v2 gold plumbing were EXTRACTED to
//     `src/eval-core.ts` in wave-7 (tsconfig rootDir="src" excludes test/, so src
//     scripts could not import them here); this file re-exports them so every
//     existing test import is untouched.
//  2. A DB-backed two-layer runner — `runRecallEval` — that scores v2 (stable-id) gold
//     at BOTH layers of the live pipeline: Recall@25 at `fuseCandidates` (the pool) and
//     nDCG@10 + MRR at the full `searchMemory`, resolving `superseded_by` forward
//     chains so consolidation can never orphan gold (AC-106). It needs loopback
//     Postgres and a (real or mocked) Voyage, so callers guard it with
//     `describe.skipIf(!process.env.DATABASE_URL)`. Run artifacts (AC-104) are written
//     by `writeRunArtifact`; the CI gate joins them per-query via `perQueryDeltas`.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  canonicalizeRanked,
  filterRowsBySplit,
  mrr,
  ndcgAtK,
  recallAtK,
  resolveGoldIds,
  type RecallEvalFileV2,
  type RankedTitles,
  type RelevantTitles,
} from "../src/eval-core.js";

// ── Extracted gate math + v2 gold plumbing (src/eval-core.ts) — re-exported verbatim ──
export {
  canonicalizeRanked,
  filterRowsBySplit,
  mrr,
  mulberry32,
  ndcgAtK,
  pairedBootstrapCI,
  recallAtK,
  regressionExcluded,
  resolveGoldIds,
  signTest,
  tryLoadRecallEvalV2,
} from "../src/eval-core.js";
export type {
  RankedTitles,
  RecallEvalFileV2,
  RecallEvalRowV2,
  RelevantTitles,
} from "../src/eval-core.js";

function normalize(title: string): string {
  return title.trim().toLowerCase();
}

/** Mean of `recallAtK` across many queries (the corpus-level Recall@k). */
export function meanRecallAtK(
  perQuery: { hits: RankedTitles; relevant: RelevantTitles }[],
  k: number,
): number {
  if (perQuery.length === 0) return 0;
  const sum = perQuery.reduce(
    (acc, q) => acc + recallAtK(q.hits, q.relevant, k),
    0,
  );
  return sum / perQuery.length;
}

/** Mean reciprocal rank across many queries. */
export function meanReciprocalRank(
  perQuery: { hits: RankedTitles; relevant: RelevantTitles }[],
): number {
  if (perQuery.length === 0) return 0;
  const sum = perQuery.reduce((acc, q) => acc + mrr(q.hits, q.relevant), 0);
  return sum / perQuery.length;
}

export interface RecallEvalRow {
  query: string;
  relevantTitles: RelevantTitles;
}

export interface RecallEvalFile {
  _seed?: boolean;
  k: number;
  rows: RecallEvalRow[];
}

const DEFAULT_EVAL_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  "fixtures",
  "recall-eval.json",
);

/** Load + shape the v1 (title-keyed) seed eval file. Kept for the seed-fixture tests
 *  and as `gold:migrate`'s input; the runner below consumes the v2 shape. */
export function loadRecallEval(path = DEFAULT_EVAL_PATH): RecallEvalFile {
  const raw = JSON.parse(readFileSync(path, "utf8")) as RecallEvalFile;
  if (!Array.isArray(raw.rows) || typeof raw.k !== "number") {
    throw new Error("recall-eval.json must have { k: number, rows: [...] }");
  }
  return raw;
}

// ── Two-layer eval runner + artifacts (wave-2, AC-104) ───────────────────────────────

/** Pool-layer Recall@k cutoff (matches `config.candidatePool`'s documented ceiling). */
export const POOL_RECALL_K = 25;
/** Final-layer nDCG/MRR cutoff. */
export const FINAL_NDCG_K = 10;

/** One query's scores across BOTH layers of an eval run. */
export interface RunPerQuery {
  id: string;
  query: string;
  /** Recall@25 at the candidate-pool layer (fuse only — did recall surface gold?). */
  poolRecall: number;
  /** nDCG@10 at the final layer (full pipeline — did rerank+blend order gold well?). */
  ndcg: number;
  /** Reciprocal rank of the first relevant final hit (0 when absent). */
  mrr: number;
  /** 1-based rank of the first relevant final hit, or null. */
  rank: number | null;
}

/**
 * The committed run-artifact payload (AC-104): the FULL retrievalConfig() snapshot plus
 * per-query scores for both layers. `test/runs/baseline-dev.json` is a committed copy
 * of one of these, chosen by the operator at a flip gate.
 */
export interface RunArtifact {
  retrievalConfig: Record<string, unknown>;
  evalVersion: number;
  split: string;
  rows: number;
  perQuery: RunPerQuery[];
  aggregates: { recallAt25: number; ndcgAt10: number; mrr10: number };
}

/**
 * v2 two-layer DB-backed runner (id-keyed gold, supersession-resolved):
 *
 *  - POOL layer — `fuseCandidates` directly; Recall@25 over candidate ids (the reranker
 *    cannot fix recall, so the floor is scored where recall happens).
 *  - FINAL layer — the full `searchMemory` pipeline; nDCG@10 + MRR over hit ids.
 *
 * Hit ids are CANONICALIZED before scoring: a hit on any member of a gold id's forward
 * chain counts as a hit on that gold id (AC-106), so IDCG stays anchored to the number
 * of gold items, not chain lengths. Returns the complete artifact payload
 * (retrievalConfig() snapshot included) ready for `writeRunArtifact`. Requires
 * DATABASE_URL + a (real or mocked) Voyage; `../src/memory.js` is imported lazily so
 * pure tests never pull the pg pool / Voyage client.
 */
export async function runRecallEval(
  evalFile: RecallEvalFileV2,
  opts: {
    projectId: string;
    split?: "dev" | "test";
    /** Pool for gold-chain resolution; defaults to the service's module pool. */
    pool?: {
      query: (
        text: string,
        params: unknown[],
      ) => Promise<{ rows: { root: string; id: string }[] }>;
    };
  },
): Promise<RunArtifact> {
  const { fuseCandidates, searchMemory, retrievalConfig } =
    await import("../src/memory.js");
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
  const pool = opts.pool ?? (await import("../src/db/pool.js")).pool;
  const rows = opts.split
    ? filterRowsBySplit(evalFile, opts.split)
    : evalFile.rows;

  const perQuery: RunPerQuery[] = [];
  for (const row of rows) {
    const chains = await resolveGoldIds(pool, row.relevantIds);
    const canonical = (ids: string[]) => canonicalizeRanked(ids, chains);

    // Layer 1: candidate pool only.
    const cands = await fuseCandidates({
      projectId: opts.projectId,
      query: row.query,
    });
    const poolIds = canonical(cands.map((c) => c.id));
    const poolRecall = recallAtK(poolIds, row.relevantIds, POOL_RECALL_K);

    // Layer 2: the full live pipeline.
    const result = await searchMemory({
      projectId: opts.projectId,
      query: row.query,
      limit: FINAL_NDCG_K,
    });
    const hits = result.hits;
    const finalIds = canonical(hits.map((h) => h.id));

    const rel = new Set(row.relevantIds.map(normalize));
    let rank: number | null = null;
    for (let i = 0; i < finalIds.length; i++) {
      if (rel.has(normalize(finalIds[i]))) {
        rank = i + 1;
        break;
      }
    }

    perQuery.push({
      id: row.id,
      query: row.query,
      poolRecall,
      ndcg: ndcgAtK(finalIds, row.relevantIds, FINAL_NDCG_K),
      mrr: mrr(finalIds, row.relevantIds),
      rank,
    });
  }

  const n = perQuery.length;
  const mean = (f: (q: RunPerQuery) => number) =>
    n === 0 ? 0 : perQuery.reduce((a, q) => a + f(q), 0) / n;
  return {
    retrievalConfig: snapshot,
    evalVersion: evalFile.version,
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

/**
 * Write an eval-run artifact as `<dir>/<ISO-date>-<label>.json` and return the path.
 * Artifacts under `test/runs/` are COMMITTED — they are the baselines the CI gate joins
 * against; tests write to a tmp dir instead so `git status` stays clean.
 */
export function writeRunArtifact(
  dir: string,
  label: string,
  payload: RunArtifact,
): string {
  mkdirSync(dir, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  const path = join(dir, `${date}-${label}.json`);
  writeFileSync(path, JSON.stringify(payload, null, 2) + "\n");
  return path;
}

/**
 * Join a baseline artifact against a fresh run BY ROW ID and return per-query nDCG@10
 * deltas (fresh − baseline, id-sorted). Fails LOUDLY when the id sets differ — a changed
 * dev split invalidates the baseline (the operator re-records it at a flip gate).
 */
export function perQueryDeltas(
  baseline: RunArtifact,
  fresh: RunArtifact,
): number[] {
  const b = new Map(baseline.perQuery.map((q) => [q.id, q]));
  const f = new Map(fresh.perQuery.map((q) => [q.id, q]));
  const missing = [...b.keys()].filter((id) => !f.has(id));
  const extra = [...f.keys()].filter((id) => !b.has(id));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `eval-split drift: baseline and fresh run disagree on row ids ` +
        `(missing from fresh: [${missing.join(", ")}]; not in baseline: [${extra.join(", ")}]). ` +
        `A changed dev split invalidates the baseline — re-record it at a flip gate.`,
    );
  }
  return [...b.keys()].sort().map((id) => f.get(id)!.ndcg - b.get(id)!.ndcg);
}
