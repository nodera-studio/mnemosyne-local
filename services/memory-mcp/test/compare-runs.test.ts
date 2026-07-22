// Wave 5, Step 5 — `npm run eval:compare` (src/db/compare-runs.ts): pure unit tests on
// fixture artifacts with hand-computed values, plus a pin that the INLINED CI math is
// byte-equivalent to the canonical test/recall.helper.ts implementation (the inline
// copy exists because tsconfig rootDir="src" excludes test/; wave-7 extracts
// src/eval-core.ts and re-points both). No DB, no network.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  compareRuns,
  formatCompare,
  mulberry32 as inlineMulberry32,
  pairedBootstrapCI as inlinePairedBootstrapCI,
  type CompareArtifact,
} from "../src/db/compare-runs.js";
import {
  mulberry32 as canonicalMulberry32,
  pairedBootstrapCI as canonicalPairedBootstrapCI,
} from "./recall.helper.js";
import { EVAL_RECORD_COST_NOTE, parseRunEvalArgs } from "./run-eval.js";

const here = dirname(fileURLToPath(import.meta.url));

function artifact(
  rows: Array<[string, number, number, number]>,
): CompareArtifact {
  return {
    split: "test",
    rows: rows.length,
    perQuery: rows.map(([id, ndcg, poolRecall, mrr]) => ({
      id,
      ndcg,
      poolRecall,
      mrr,
    })),
  };
}

describe("compareRuns (hand-computed fixtures)", () => {
  it("constant deltas: mean = every resample mean = CI bounds (degenerate bootstrap)", () => {
    const a = artifact([
      ["m-001", 0.5, 1, 0.5],
      ["m-002", 0.5, 1, 0.5],
      ["m-003", 0.5, 0, 0.25],
    ]);
    const b = artifact([
      ["m-001", 0.6, 1, 0.5],
      ["m-002", 0.6, 1, 0.5],
      ["m-003", 0.6, 1, 0.25],
    ]);
    const r = compareRuns(a, b);
    expect(r.rows).toBe(3);
    // Every per-query delta is exactly 0.1, so every bootstrap resample mean is 0.1.
    expect(r.ndcg.meanDelta).toBeCloseTo(0.1, 12);
    expect(r.ndcg.ci.mean).toBeCloseTo(0.1, 12);
    expect(r.ndcg.ci.ciLow).toBeCloseTo(0.1, 12);
    expect(r.ndcg.ci.ciHigh).toBeCloseTo(0.1, 12);
    // Aggregate deltas: recall 2/3 → 1 (Δ = 1/3); mrr unchanged.
    expect(r.poolRecall.meanDelta).toBeCloseTo(1 / 3, 12);
    expect(r.mrr.meanDelta).toBeCloseTo(0, 12);
    // Sign counts: all three ndcg deltas positive.
    expect(r.wins).toBe(3);
    expect(r.losses).toBe(0);
    expect(r.ties).toBe(0);
  });

  it("mixed deltas: sign counts + means match hand computation", () => {
    const a = artifact([
      ["m-001", 0.2, 1, 0.5],
      ["m-002", 0.8, 1, 1.0],
      ["m-003", 0.5, 1, 0.5],
      ["m-004", 0.4, 1, 0.25],
    ]);
    const b = artifact([
      ["m-001", 0.4, 1, 0.5], // +0.2
      ["m-002", 0.7, 1, 1.0], // -0.1
      ["m-003", 0.5, 1, 0.5], //  0
      ["m-004", 0.5, 1, 0.25], // +0.1
    ]);
    const r = compareRuns(a, b);
    expect(r.ndcg.meanA).toBeCloseTo(0.475, 12);
    expect(r.ndcg.meanB).toBeCloseTo(0.525, 12);
    expect(r.ndcg.meanDelta).toBeCloseTo(0.05, 12);
    expect(r.wins).toBe(2);
    expect(r.losses).toBe(1);
    expect(r.ties).toBe(1);
    // The CI is a real interval around the observed mean, deterministic under seed 42.
    const again = compareRuns(a, b);
    expect(again.ndcg.ci).toEqual(r.ndcg.ci);
    expect(r.ndcg.ci.ciLow).toBeLessThanOrEqual(r.ndcg.ci.mean);
    expect(r.ndcg.ci.ciHigh).toBeGreaterThanOrEqual(r.ndcg.ci.mean);
  });

  it("row-id drift between artifacts fails LOUDLY", () => {
    const a = artifact([["m-001", 0.5, 1, 0.5]]);
    const b = artifact([["m-002", 0.5, 1, 0.5]]);
    expect(() => compareRuns(a, b)).toThrow(/run-id drift/);
  });

  it("formatCompare prints the metrics, CI, sign counts, and a verdict line", () => {
    const a = artifact([
      ["m-001", 0.5, 1, 0.5],
      ["m-002", 0.5, 1, 0.5],
    ]);
    const b = artifact([
      ["m-001", 0.6, 1, 0.5],
      ["m-002", 0.6, 1, 0.5],
    ]);
    const out = formatCompare(compareRuns(a, b), "pre.json", "post.json");
    expect(out).toContain("nDCG@10");
    expect(out).toContain("Recall@25");
    expect(out).toContain("MRR@10");
    expect(out).toContain("95% CI");
    expect(out).toContain("2 win(s), 0 loss(es), 0 tie(s)");
    expect(out).toContain("delta = post.json − pre.json");
    expect(out).toContain("significantly BETTER");
  });
});

describe("inline CI math ≡ canonical test/recall.helper.ts math (until src/eval-core.ts)", () => {
  it("mulberry32 sequences are identical for the same seed", () => {
    const a = inlineMulberry32(42);
    const b = canonicalMulberry32(42);
    for (let i = 0; i < 100; i++) expect(a()).toBe(b());
  });

  it("pairedBootstrapCI is byte-identical on a mixed delta vector (seed 42 default)", () => {
    const deltas = [0.2, -0.1, 0, 0.1, 0.05, -0.3, 0.4, 0.02];
    expect(inlinePairedBootstrapCI(deltas)).toEqual(
      canonicalPairedBootstrapCI(deltas),
    );
    expect(
      inlinePairedBootstrapCI(deltas, { seed: 7, iterations: 500 }),
    ).toEqual(canonicalPairedBootstrapCI(deltas, { seed: 7, iterations: 500 }));
    expect(inlinePairedBootstrapCI([])).toEqual({
      mean: 0,
      ciLow: 0,
      ciHigh: 0,
    });
  });
});

describe("eval:record recorder (pure guards, AC-108/AC-109)", () => {
  it("refuses without --yes; parses split/label; rejects unknown splits", () => {
    expect(parseRunEvalArgs([])).toBe(EVAL_RECORD_COST_NOTE);
    expect(parseRunEvalArgs(["--split", "test"])).toBe(EVAL_RECORD_COST_NOTE);
    expect(parseRunEvalArgs(["--yes"])).toEqual({
      yes: true,
      split: "test",
      label: "test",
    });
    expect(
      parseRunEvalArgs([
        "--yes",
        "--split",
        "dev",
        "--label",
        "pre-consolidation",
      ]),
    ).toEqual({ yes: true, split: "dev", label: "pre-consolidation" });
    expect(parseRunEvalArgs(["--yes", "--split", "prod"])).toMatch(
      /invalid --split/,
    );
    // A flag-shaped "value" is an omitted value, never swallowed as the value
    // (the label flows into the artifact filename).
    expect(parseRunEvalArgs(["--yes", "--label", "--split", "test"])).toEqual({
      yes: true,
      split: "test",
      label: "test",
    });
    expect(parseRunEvalArgs(["--yes", "--split"])).toEqual({
      yes: true,
      split: "test",
      label: "test",
    });
  });

  it("both new operator scripts are npm-run gated with import.meta.url main-guards (AC-108)", () => {
    const pkg = JSON.parse(
      readFileSync(join(here, "..", "package.json"), "utf8"),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts["eval:compare"]).toBeTruthy();
    expect(pkg.scripts["eval:record"]).toBeTruthy();
    for (const f of ["src/db/compare-runs.ts", "test/run-eval.ts"]) {
      const src = readFileSync(join(here, "..", f), "utf8");
      expect(src, `${f} must main-guard on import.meta.url`).toMatch(
        /import\.meta\.url/,
      );
    }
  });
});
