// Pure recall/MRR scoring math for the Wave 5 code embedder bake-off (AC-030/AC-031).
//
// Lives in src/ (not test/) because BOTH the operator-run bake-off module
// (src/db/bakeoff-embed.ts) and the test harness (test/code-eval.helper.ts) need it, and
// tsconfig.json's `rootDir: src` forbids src importing from test. There is NO shared
// package in this repo, so this duplicates memory-mcp's recall.helper.ts math (keyed on
// file PATH here, on memory TITLE there). A bug here silently picks the wrong embedder,
// so it is unit-tested directly against hand-computed values (test/code-eval.helper.test.ts).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

/** Pool-layer Recall@k cutoff (matches `config.candidatePool`'s documented ceiling); shared by the eval runner (test/code-eval.helper.ts) and the rerank bakeoff (src/db/bakeoff-rerank.ts). */
export const POOL_RECALL_K = 25;

/** A ranked list of candidate file paths (best-first) returned for one query. */
export type RankedPaths = string[];

/** The set of file paths considered relevant for one query. */
export type RelevantPaths = string[];

/** Normalize a path for comparison: trim, lowercase, strip a leading "./". */
export function normalizePath(p: string): string {
  return p.trim().toLowerCase().replace(/^\.\//, "");
}

function relevantSet(relevant: RelevantPaths): Set<string> {
  return new Set(relevant.map(normalizePath));
}

/**
 * Recall@k for a single query: 1 if any relevant path appears in the top-k hits, else 0.
 * Edge cases: empty `relevant` -> 0 (conservative); k <= 0 -> 0.
 */
export function recallAtK(
  hits: RankedPaths,
  relevant: RelevantPaths,
  k: number,
): number {
  if (k <= 0 || relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  return hits.slice(0, k).some((h) => rel.has(normalizePath(h))) ? 1 : 0;
}

/**
 * Reciprocal rank for a single query: 1 / (1-based rank of the first relevant hit), or 0
 * when no relevant path appears (or inputs are empty). MRR is the mean over a query set.
 */
export function mrr(hits: RankedPaths, relevant: RelevantPaths): number {
  if (relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  for (let i = 0; i < hits.length; i++) {
    if (rel.has(normalizePath(hits[i]))) return 1 / (i + 1);
  }
  return 0;
}

/** Mean of `recallAtK` across many queries (the corpus-level Recall@k). */
export function meanRecallAtK(
  perQuery: { hits: RankedPaths; relevant: RelevantPaths }[],
  k: number,
): number {
  if (perQuery.length === 0) return 0;
  return (
    perQuery.reduce((acc, q) => acc + recallAtK(q.hits, q.relevant, k), 0) /
    perQuery.length
  );
}

/** Mean reciprocal rank across many queries. */
export function meanReciprocalRank(
  perQuery: { hits: RankedPaths; relevant: RelevantPaths }[],
): number {
  if (perQuery.length === 0) return 0;
  return (
    perQuery.reduce((acc, q) => acc + mrr(q.hits, q.relevant), 0) /
    perQuery.length
  );
}

/**
 * nDCG@k for a single query with BINARY gains (a hit is relevant or it is not — binary
 * suffices for this program's gold sets). DCG = Σ 1/log2(i+1) over relevant hits in the
 * top-k (1-based rank i); IDCG = the ideal DCG for min(|relevant|, k) items; nDCG =
 * DCG/IDCG ∈ [0,1]. Keyed on normalized paths here (stable ids after the wave-2 key
 * switch — the function is key-agnostic strings). This IS the regression-gate metric;
 * a bug here silently green-lights a bad flip. Edge cases mirror `recallAtK`: empty
 * `relevant` -> 0, k <= 0 -> 0.
 *
 * ONE-CREDIT rule: each relevant key gains at its FIRST occurrence only — chunk-keyed
 * hit lists routinely carry the same file path twice, and crediting every occurrence
 * lets DCG exceed the IDCG anchor (nDCG > 1), silently biasing the regression gate.
 * This matches `mrr`/`recallAtK` first-occurrence semantics.
 */
export function ndcgAtK(
  hits: RankedPaths,
  relevant: RelevantPaths,
  k: number,
): number {
  if (k <= 0 || relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  const credited = new Set<string>();
  let dcg = 0;
  const topK = hits.slice(0, k);
  for (let i = 0; i < topK.length; i++) {
    // 1-based rank = i+1 → gain discount 1/log2(rank+1) = 1/log2(i+2)
    const key = normalizePath(topK[i]);
    if (rel.has(key) && !credited.has(key)) {
      credited.add(key);
      dcg += 1 / Math.log2(i + 2);
    }
  }
  let idcg = 0;
  const idealCount = Math.min(rel.size, k);
  for (let i = 0; i < idealCount; i++) idcg += 1 / Math.log2(i + 2);
  return idcg === 0 ? 0 : dcg / idcg;
}

/**
 * Tiny seeded PRNG (mulberry32) so the bootstrap below is DETERMINISTIC under a fixed
 * seed (AC-103) — a gate that flickers across runs is not a gate.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Linear-interpolation percentile of a SORTED ascending array (numpy type-7). */
function percentileSorted(sorted: number[], q: number): number {
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * Percentile paired bootstrap over PER-QUERY metric deltas. Sign convention:
 * delta = candidate − baseline (positive = candidate wins). Resamples
 * `deltas.length` values with replacement `iterations` times using the seeded PRNG,
 * records each resample mean, and returns the observed mean plus the percentile CI
 * (default 95% → 2.5th/97.5th). Deterministic for a fixed seed. Empty input → zeros.
 */
export function pairedBootstrapCI(
  deltas: number[],
  opts: { iterations?: number; seed?: number; level?: number } = {},
): { mean: number; ciLow: number; ciHigh: number } {
  if (deltas.length === 0) return { mean: 0, ciLow: 0, ciHigh: 0 };
  const iterations = opts.iterations ?? 10_000;
  const level = opts.level ?? 0.95;
  const rnd = mulberry32(opts.seed ?? 42);
  const n = deltas.length;
  const mean = deltas.reduce((a, b) => a + b, 0) / n;
  const means = new Array<number>(iterations);
  for (let it = 0; it < iterations; it++) {
    let sum = 0;
    for (let j = 0; j < n; j++) sum += deltas[Math.floor(rnd() * n)];
    means[it] = sum / n;
  }
  means.sort((a, b) => a - b);
  const alpha = (1 - level) / 2;
  return {
    mean,
    ciLow: percentileSorted(means, alpha),
    ciHigh: percentileSorted(means, 1 - alpha),
  };
}

/**
 * Gate semantics (AC-105): the regression side is excluded-zero when the WHOLE CI sits
 * below zero — i.e. the candidate is statistically-significantly worse. With
 * delta = candidate − baseline, that is `ciHigh < 0`.
 */
export function regressionExcluded(ci: {
  ciLow: number;
  ciHigh: number;
}): boolean {
  return ci.ciHigh < 0;
}

/** First relevant rank (1-based) in a ranked list, or null. Shared by both runners. */
export function firstRelevantRank(
  hits: RankedPaths,
  relevant: RelevantPaths,
): number | null {
  const rel = relevantSet(relevant);
  for (let i = 0; i < hits.length; i++) {
    if (rel.has(normalizePath(hits[i]))) return i + 1;
  }
  return null;
}

export interface CodeEvalRow {
  query: string;
  /** Relevant file path(s). A hit on ANY of these in the top-k counts as recall. */
  relevantPaths: RelevantPaths;
  /** Optional human-readable label / archetype for the query (ignored by the loader). */
  archetype?: string;
  // ── v2 fields (wave-2). For code the STABLE ID is the file path (chunk ids churn on
  // every reindex) — so v2 keeps `relevantPaths` and only adds bookkeeping. All three
  // are optional so the loader tolerates v1 files (bakeoff back-compat).
  /** Stable row id ("c-001"…) — the join key for baseline↔fresh gate comparisons. */
  id?: string;
  /** Gold split: "dev" (CI gate recomputes it) or "test" (spent only by flip gates). */
  split?: "dev" | "test";
  /** How the row entered gold: seed-v1 | distilled | log-harvest. */
  provenance?: string;
  /** Who human-approved the row (AC-109); absent on unapproved seeds. */
  approvedBy?: string;
}

export interface CodeEvalFile {
  /** 2 for split/provenance-aware files; absent on v1 (tolerated for back-compat). */
  version?: number;
  changelog?: string[];
  _seed?: boolean;
  k: number;
  rows: CodeEvalRow[];
}

/**
 * Rows of one split. v1 rows carry no `split`, so they default to "dev" — the CI gate
 * only ever recomputes dev, and a v1 file predates the frozen test split anyway.
 */
export function filterRowsBySplit(
  file: CodeEvalFile,
  split: "dev" | "test",
): CodeEvalRow[] {
  return file.rows.filter((r) => (r.split ?? "dev") === split);
}

export interface CodeRecallEvalResult {
  recallAtK: number;
  mrr: number;
  perQuery: { query: string; rank: number | null; recall: number }[];
}

/**
 * Load + shape the eval fixture (v2 with `version`/`split`/`provenance`/row ids, or a
 * bare v1 file — tolerated for bakeoff back-compat). The fixture lives in
 * test/fixtures/code-eval.json; `test/` is a sibling of both `src/` (under tsx) and
 * `dist/` (compiled), so the default "../test/fixtures/..." resolves in both. Callers
 * may pass an explicit path.
 */
export function loadCodeEval(
  path = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test",
    "fixtures",
    "code-eval.json",
  ),
): CodeEvalFile {
  const raw = JSON.parse(readFileSync(path, "utf8")) as CodeEvalFile;
  if (!Array.isArray(raw.rows) || typeof raw.k !== "number") {
    throw new Error("code-eval.json must have { k: number, rows: [...] }");
  }
  if (raw.version !== undefined && raw.version !== 2) {
    throw new Error(`code-eval.json version ${raw.version} is not supported`);
  }
  if (raw.version === 2) {
    for (const r of raw.rows) {
      if (!r.id || (r.split !== "dev" && r.split !== "test")) {
        throw new Error(
          `code-eval.json v2 rows need an id and a dev|test split (bad row: ${JSON.stringify(r.query)})`,
        );
      }
    }
  }
  return raw;
}

// ── Eval-run artifacts + gate join (wave-2, AC-104/AC-105) ───────────────────────────

/** One query's scores across BOTH layers of an eval run. */
export interface RunPerQuery {
  /** Gold row id (v2) — the baseline↔fresh join key. */
  id: string;
  query: string;
  /** Recall@25 at the candidate-pool layer (fuse only — did recall surface gold?). */
  poolRecall: number;
  /** nDCG@10 at the final layer (full pipeline — did rerank order gold well?). */
  ndcg: number;
  /** Reciprocal rank of the first relevant final hit (0 when absent). */
  mrr: number;
  /** 1-based rank of the first relevant final hit, or null. */
  rank: number | null;
}

/**
 * The committed run-artifact payload (AC-104): the FULL retrievalConfig() snapshot plus
 * per-query scores for both layers. `test/runs/baseline-dev.json` is a committed copy of
 * one of these, chosen by the operator at a flip gate.
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
