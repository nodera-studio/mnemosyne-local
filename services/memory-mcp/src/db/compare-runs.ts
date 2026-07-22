// FREE eval-artifact comparator — `npm run eval:compare -- <runA.json> <runB.json>`
// (wave-5 Step 5; also the wave-7 bakeoff's pairwise verdict tool).
//
// Reads two run artifacts (the writeRunArtifact JSON shape), joins per-query scores
// BY ROW ID (loud failure on id drift — a changed split invalidates the comparison),
// and prints the mean nDCG@10 delta with a seed-42 paired-bootstrap 95% CI plus
// aggregate deltas and sign counts. Sign convention: delta = B − A (positive = B wins),
// matching test/recall.helper.ts.
//
// The CI math (mulberry32 + percentile bootstrap) comes from src/eval-core.ts — the
// wave-7 extraction that ended this file's temporary inline copies (tsconfig
// rootDir="src" excludes test/, so test/recall.helper.ts was unimportable from here).
// Both are re-exported so test/compare-runs.test.ts's equivalence pins stay put.
//
// No DB, no network, no quota — reads two local JSON files (AC-108 does not bind).

import { readFileSync } from "node:fs";
import { pairedBootstrapCI } from "../eval-core.js";

export { mulberry32, pairedBootstrapCI } from "../eval-core.js";

// ── Artifact comparison ───────────────────────────────────────────────────────────────

/** The per-query slice of a run artifact this tool needs (RunArtifact superset-safe). */
export interface ComparePerQuery {
  id: string;
  ndcg: number;
  poolRecall: number;
  mrr: number;
}

export interface CompareArtifact {
  split?: string;
  rows?: number;
  perQuery: ComparePerQuery[];
}

export interface MetricDelta {
  meanA: number;
  meanB: number;
  meanDelta: number;
}

export interface CompareResult {
  rows: number;
  ndcg: MetricDelta & { ci: { mean: number; ciLow: number; ciHigh: number } };
  poolRecall: MetricDelta;
  mrr: MetricDelta;
  /** Sign counts over per-query nDCG deltas. */
  wins: number;
  losses: number;
  ties: number;
}

export function loadArtifact(path: string): CompareArtifact {
  const raw = JSON.parse(readFileSync(path, "utf8")) as CompareArtifact;
  if (!Array.isArray(raw.perQuery)) {
    throw new Error(`${path} is not a run artifact (missing perQuery[])`);
  }
  return raw;
}

/**
 * Join two artifacts' per-query scores by row id (loud failure on drift) and compute
 * B − A deltas: seed-42 paired-bootstrap CI on nDCG@10 (the gated metric) plus mean
 * deltas for pool Recall@25 and MRR, and per-query sign counts.
 */
export function compareRuns(
  a: CompareArtifact,
  b: CompareArtifact,
  opts: { seed?: number; iterations?: number } = {},
): CompareResult {
  const byIdA = new Map(a.perQuery.map((q) => [q.id, q]));
  const byIdB = new Map(b.perQuery.map((q) => [q.id, q]));
  const missing = [...byIdA.keys()].filter((id) => !byIdB.has(id));
  const extra = [...byIdB.keys()].filter((id) => !byIdA.has(id));
  if (missing.length > 0 || extra.length > 0) {
    throw new Error(
      `run-id drift: the artifacts disagree on row ids ` +
        `(missing from B: [${missing.join(", ")}]; not in A: [${extra.join(", ")}]). ` +
        `Compare runs recorded on the SAME split only.`,
    );
  }
  const ids = [...byIdA.keys()].sort();
  const rowsA = ids.map((id) => byIdA.get(id)!);
  const rowsB = ids.map((id) => byIdB.get(id)!);
  const mean = (xs: number[]) =>
    xs.length === 0 ? 0 : xs.reduce((s, x) => s + x, 0) / xs.length;
  const metric = (f: (q: ComparePerQuery) => number): MetricDelta => {
    const meanA = mean(rowsA.map(f));
    const meanB = mean(rowsB.map(f));
    return { meanA, meanB, meanDelta: meanB - meanA };
  };

  const ndcgDeltas = ids.map((id) => byIdB.get(id)!.ndcg - byIdA.get(id)!.ndcg);
  const eps = 1e-12;
  return {
    rows: ids.length,
    ndcg: {
      ...metric((q) => q.ndcg),
      ci: pairedBootstrapCI(ndcgDeltas, {
        seed: opts.seed ?? 42,
        iterations: opts.iterations,
      }),
    },
    poolRecall: metric((q) => q.poolRecall),
    mrr: metric((q) => q.mrr),
    wins: ndcgDeltas.filter((d) => d > eps).length,
    losses: ndcgDeltas.filter((d) => d < -eps).length,
    ties: ndcgDeltas.filter((d) => Math.abs(d) <= eps).length,
  };
}

const f4 = (x: number) => x.toFixed(4);

export function formatCompare(
  r: CompareResult,
  labelA = "A",
  labelB = "B",
): string {
  const sig =
    r.ndcg.ci.ciLow > 0
      ? `${labelB} significantly BETTER (CI excludes zero)`
      : r.ndcg.ci.ciHigh < 0
        ? `${labelB} significantly WORSE (CI excludes zero)`
        : "no significant difference (CI includes zero)";
  return [
    `eval:compare — ${r.rows} paired rows (delta = ${labelB} − ${labelA})`,
    `  nDCG@10     ${f4(r.ndcg.meanA)} → ${f4(r.ndcg.meanB)}   Δ ${f4(r.ndcg.meanDelta)}   95% CI [${f4(r.ndcg.ci.ciLow)}, ${f4(r.ndcg.ci.ciHigh)}] (seed 42)`,
    `  Recall@25   ${f4(r.poolRecall.meanA)} → ${f4(r.poolRecall.meanB)}   Δ ${f4(r.poolRecall.meanDelta)}`,
    `  MRR@10      ${f4(r.mrr.meanA)} → ${f4(r.mrr.meanB)}   Δ ${f4(r.mrr.meanDelta)}`,
    `  per-query nDCG: ${r.wins} win(s), ${r.losses} loss(es), ${r.ties} tie(s)`,
    `  verdict: ${sig}`,
  ].join("\n");
}

// ── CLI entrypoint (npm run eval:compare -- <runA.json> <runB.json>) ─────────────────
// Skipped when imported (tests import the exported functions directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const [pathA, pathB] = process.argv
    .slice(2)
    .filter((a) => !a.startsWith("-"));
  if (!pathA || !pathB) {
    console.error(
      "usage: npm run eval:compare -- <runA.json> <runB.json>\n" +
        "  A = baseline/pre artifact, B = candidate/post artifact (delta = B − A).",
    );
    process.exit(1);
  }
  try {
    const result = compareRuns(loadArtifact(pathA), loadArtifact(pathB));
    console.log(formatCompare(result, pathA, pathB));
  } catch (e) {
    console.error((e as Error).message);
    process.exit(1);
  }
}
