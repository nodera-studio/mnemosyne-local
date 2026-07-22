// Eval-core: the pure gate/eval math + v2 gold plumbing, extracted VERBATIM from
// test/recall.helper.ts (wave-7 Step 1). It lives under src/ because tsconfig
// rootDir="src" excludes test/ — src files (src/db/compare-runs.ts, wave-7's
// src/db/bakeoff-blend.ts) cannot import test/recall.helper.ts, and duplicating the
// CI math per script is how gates silently diverge. test/recall.helper.ts re-exports
// everything here, so every existing test import is untouched.
//
// Contents: `ndcgAtK`, `mulberry32`, `pairedBootstrapCI` (+ private
// `percentileSorted`), `regressionExcluded`, the NEW `signTest`, the v2 eval row
// types + `tryLoadRecallEvalV2` + `filterRowsBySplit`, `resolveGoldIds`,
// `canonicalizeRanked`. The DB-backed runner (`runRecallEval`) and the v1 loader stay
// in test/recall.helper.ts — they compose the live pipeline and belong to the harness.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

/** A ranked list of candidate keys (best-first) returned for one query. */
export type RankedTitles = string[];

/** The set of keys considered relevant for one query. */
export type RelevantTitles = string[];

function normalize(title: string): string {
  return title.trim().toLowerCase();
}

function relevantSet(relevant: RelevantTitles): Set<string> {
  return new Set(relevant.map(normalize));
}

/**
 * Recall@k for a single query: 1 if any relevant title appears in the top-k hits,
 * else 0. (Binary "did we surface a relevant result in the top-k" — the gate we care
 * about for a small held-out set; average across queries for the corpus number.)
 *
 * Edge cases: empty `relevant` -> 0 (nothing to recall, conservative); k <= 0 -> 0.
 */
export function recallAtK(
  hits: RankedTitles,
  relevant: RelevantTitles,
  k: number,
): number {
  if (k <= 0 || relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  const topK = hits.slice(0, k);
  return topK.some((h) => rel.has(normalize(h))) ? 1 : 0;
}

/**
 * Reciprocal rank for a single query: 1 / (rank of the first relevant hit), ranks
 * 1-based. Returns 0 when no relevant title appears anywhere in `hits` (or inputs are
 * empty). MRR over a query set is the mean of these per-query reciprocal ranks.
 */
export function mrr(hits: RankedTitles, relevant: RelevantTitles): number {
  if (relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  for (let i = 0; i < hits.length; i++) {
    if (rel.has(normalize(hits[i]))) return 1 / (i + 1);
  }
  return 0;
}

/**
 * nDCG@k for a single query with BINARY gains (a hit is relevant or it is not — binary
 * suffices for this program's gold sets). DCG = Σ 1/log2(i+1) over relevant hits in the
 * top-k (1-based rank i); IDCG = the ideal DCG for min(|relevant|, k) items; nDCG =
 * DCG/IDCG ∈ [0,1]. Key-agnostic: compares normalized strings, so it works on titles
 * today and stable ids after the wave-2 key switch. This IS the regression-gate metric;
 * a bug here silently passes a bad flip. Edge cases mirror `recallAtK`: empty
 * `relevant` -> 0, k <= 0 -> 0.
 *
 * ONE-CREDIT rule: each relevant key gains at its FIRST occurrence only — canonicalized
 * hit lists can carry the same key twice (e.g. two supersession-chain members mapping to
 * one gold id), and crediting every occurrence lets DCG exceed the IDCG anchor
 * (nDCG > 1), silently biasing the regression gate. This matches `mrr`/`recallAtK`
 * first-occurrence semantics.
 */
export function ndcgAtK(
  hits: RankedTitles,
  relevant: RelevantTitles,
  k: number,
): number {
  if (k <= 0 || relevant.length === 0) return 0;
  const rel = relevantSet(relevant);
  const credited = new Set<string>();
  let dcg = 0;
  const topK = hits.slice(0, k);
  for (let i = 0; i < topK.length; i++) {
    // 1-based rank = i+1 → gain discount 1/log2(rank+1) = 1/log2(i+2)
    const key = normalize(topK[i]);
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

/** Standard-normal CDF via the Abramowitz–Stegun 7.1.26 erf approximation
 *  (|error| < 1.5e-7 — far below any decision threshold the sign test feeds). */
function normalCdf(z: number): number {
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  const p = 0.5 * (1 + erf);
  return z < 0 ? 1 - p : p;
}

/**
 * Two-sided SIGN TEST over per-query deltas (wave-7): counts wins (> 0), losses (< 0),
 * ties (= 0), then tests H0 "wins and losses are equally likely" on the NON-ZERO deltas
 * only (standard sign-test practice — ties carry no direction). Normal approximation
 * with continuity correction: z = (|wins − n/2| − 0.5)/√(n/4), p = 2·(1 − Φ(z)),
 * clamped to [0, 1]. Zero non-zero deltas → p = 1 (no evidence either way). The
 * bakeoff's verdict uses the CI as the gate and this as corroboration.
 */
export function signTest(deltas: number[]): {
  wins: number;
  losses: number;
  ties: number;
  p: number;
} {
  let wins = 0;
  let losses = 0;
  let ties = 0;
  for (const d of deltas) {
    if (d > 0) wins++;
    else if (d < 0) losses++;
    else ties++;
  }
  const n = wins + losses;
  if (n === 0) return { wins, losses, ties, p: 1 };
  const z = Math.max(0, Math.abs(wins - n / 2) - 0.5) / Math.sqrt(n / 4);
  // z = 0 (a balanced-to-within-the-correction split) is EXACTLY p = 1 — bypass the
  // erf approximation so its ~1e-9 residual never renders as p = 0.999999999.
  const p = z === 0 ? 1 : Math.min(1, 2 * (1 - normalCdf(z)));
  return { wins, losses, ties, p };
}

// ── v2 gold (stable memory ids + splits, wave-2) ─────────────────────────────────────

export interface RecallEvalRowV2 {
  /** Stable row id ("m-001"…) — the baseline↔fresh gate join key. */
  id: string;
  query: string;
  /** Gold MEMORY ids. Scoring resolves `superseded_by` forward chains (AC-106). */
  relevantIds: string[];
  archetype?: string;
  /** Optional facet tag (e.g. "temporal") — sliced separately by the wave-7 bakeoff. */
  facet?: string;
  /** "dev" (CI gate recomputes it) or "test" (spent only by flip gates, AC-109). */
  split: "dev" | "test";
  provenance: "seed-v1" | "distilled" | "log-harvest";
  approvedBy?: string;
}

export interface RecallEvalFileV2 {
  version: 2;
  k: number;
  changelog: string[];
  rows: RecallEvalRowV2[];
}

// Resolves to <service-root>/test/fixtures/recall-eval.json from BOTH src/ (tsx) and
// dist/ (compiled) — src and dist sit directly under the service root.
const DEFAULT_EVAL_PATH = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "test",
  "fixtures",
  "recall-eval.json",
);

/**
 * Load the eval file WHEN it has been migrated to v2 (stable-id gold); returns null on
 * a v1 file so callers (the gate, the live runner test) can skip gracefully until the
 * operator approves + renames the `gold:migrate` proposal (AC-109).
 */
export function tryLoadRecallEvalV2(
  path = DEFAULT_EVAL_PATH,
): RecallEvalFileV2 | null {
  const raw = JSON.parse(
    readFileSync(path, "utf8"),
  ) as Partial<RecallEvalFileV2>;
  if (raw.version !== 2) return null;
  if (!Array.isArray(raw.rows) || typeof raw.k !== "number") {
    throw new Error(
      "recall-eval.json v2 must have { version: 2, k, rows: [...] }",
    );
  }
  for (const r of raw.rows) {
    if (
      !r.id ||
      !Array.isArray(r.relevantIds) ||
      (r.split !== "dev" && r.split !== "test")
    ) {
      throw new Error(
        `recall-eval.json v2 rows need id/relevantIds/split (bad row: ${JSON.stringify(r.query)})`,
      );
    }
  }
  return raw as RecallEvalFileV2;
}

/** Rows of one split. */
export function filterRowsBySplit(
  file: RecallEvalFileV2,
  split: "dev" | "test",
): RecallEvalRowV2[] {
  return file.rows.filter((r) => r.split === split);
}

/**
 * Resolve each gold id's supersession FORWARD chain (AC-106): gold id → the set of ids
 * a hit may carry and still count as that gold item (itself + every successor reached
 * via `superseded_by`). Consolidation marks losers superseded — search then returns the
 * WINNER — so without this walk, consolidating a gold-referenced memory would orphan
 * the gold row. Recursive CTE with the path-array cycle guard (decisionChain pattern);
 * ids absent from the DB still resolve to themselves.
 */
export async function resolveGoldIds(
  pool: {
    query: (
      text: string,
      params: unknown[],
    ) => Promise<{ rows: { root: string; id: string }[] }>;
  },
  ids: string[],
): Promise<Map<string, Set<string>>> {
  const chains = new Map<string, Set<string>>();
  for (const id of ids) chains.set(id, new Set([id]));
  if (ids.length === 0) return chains;

  // Forward walk: each chain row carries its own `superseded_by` pointer; the recursive
  // member joins the NEXT row (the successor) on m.id = c.superseded_by.
  const { rows } = await pool.query(
    `WITH RECURSIVE chain AS (
       SELECT m.id, m.superseded_by, m.id AS root, ARRAY[m.id] AS path
       FROM memory.memories m
       WHERE m.id = ANY($1::uuid[])
       UNION ALL
       SELECT m.id, m.superseded_by, c.root, c.path || m.id
       FROM memory.memories m
       JOIN chain c ON m.id = c.superseded_by
       WHERE NOT (m.id = ANY(c.path))
     )
     SELECT root, id FROM chain`,
    [ids],
  );
  for (const r of rows) chains.get(r.root)?.add(r.id);
  return chains;
}

/**
 * Canonicalize a ranked id list against the gold chains (AC-106): walk the hits in rank
 * order and let each hit CONSUME one not-yet-credited gold id whose chain contains it.
 * Overlapping chains can converge on one winner (consolidating two gold-referenced
 * memories into the same survivor), so a member may belong to SEVERAL gold ids — a
 * first-wins member→gold map would let a hit on the shared member credit only one fixed
 * gold, orphaning the other even when a separate hit could have stood for it. Hits whose
 * golds are all consumed (or that match no chain) pass through as their raw id, so a
 * duplicate can never credit the same gold twice.
 */
export function canonicalizeRanked(
  ids: string[],
  chains: Map<string, Set<string>>,
): string[] {
  // member -> gold ids whose chain contains it, in `chains` insertion order
  // (= relevantIds order) for determinism.
  const goldsByMember = new Map<string, string[]>();
  for (const [gold, members] of chains) {
    for (const m of members) {
      const list = goldsByMember.get(m);
      if (list) list.push(gold);
      else goldsByMember.set(m, [gold]);
    }
  }
  const credited = new Set<string>();
  return ids.map((id) => {
    for (const gold of goldsByMember.get(id) ?? []) {
      if (!credited.has(gold)) {
        credited.add(gold);
        return gold;
      }
    }
    return id;
  });
}
