// Hand-computed unit tests for the bake-off scoring math (AC-030). A silent bug here
// (off-by-one rank, wrong empty-set default, path-normalization slip) would pick the
// wrong embedder with a green-looking number, so the math is tested directly — no DB,
// no Voyage. Also guards the seed eval fixture's shape (≥50 rows, seed flag).

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import {
  filterRowsBySplit,
  firstRelevantRank,
  loadCodeEval,
  meanRecallAtK,
  meanReciprocalRank,
  mrr,
  mulberry32,
  ndcgAtK,
  normalizePath,
  pairedBootstrapCI,
  recallAtK,
  regressionExcluded,
} from "./code-eval.helper.js";

describe("normalizePath", () => {
  it("lowercases, trims, and strips a leading ./", () => {
    expect(normalizePath("  ./Src/Auth.TS ")).toBe("src/auth.ts");
  });
  it("leaves an already-normal path unchanged", () => {
    expect(normalizePath("services/codebase-mcp/src/search.ts")).toBe(
      "services/codebase-mcp/src/search.ts",
    );
  });
});

describe("recallAtK (gate math)", () => {
  const hits = ["a/one.ts", "b/two.ts", "c/three.ts", "d/four.ts"];

  it("relevant at rank 1 -> 1", () => {
    expect(recallAtK(hits, ["a/one.ts"], 10)).toBe(1);
  });
  it("relevant at rank 3, k=3 -> 1", () => {
    expect(recallAtK(hits, ["c/three.ts"], 3)).toBe(1);
  });
  it("relevant at rank 3, k=2 -> 0 (outside top-k)", () => {
    expect(recallAtK(hits, ["c/three.ts"], 2)).toBe(0);
  });
  it("not in hits -> 0", () => {
    expect(recallAtK(hits, ["z/none.ts"], 10)).toBe(0);
  });
  it("any of several relevant paths in top-k -> 1", () => {
    expect(recallAtK(hits, ["z/none.ts", "b/two.ts"], 10)).toBe(1);
  });
  it("empty relevant set -> 0 (conservative)", () => {
    expect(recallAtK(hits, [], 10)).toBe(0);
  });
  it("k <= 0 -> 0", () => {
    expect(recallAtK(hits, ["a/one.ts"], 0)).toBe(0);
  });
  it("matches paths case-insensitively and ignores leading ./", () => {
    expect(recallAtK(["  ./A/One.TS  "], ["a/one.ts"], 1)).toBe(1);
  });
});

describe("mrr (gate math)", () => {
  const hits = ["a/one.ts", "b/two.ts", "c/three.ts", "d/four.ts"];

  it("relevant at rank 1 -> 1.0", () => {
    expect(mrr(hits, ["a/one.ts"])).toBe(1.0);
  });
  it("relevant at rank 2 -> 0.5", () => {
    expect(mrr(hits, ["b/two.ts"])).toBe(0.5);
  });
  it("relevant at rank 3 -> 1/3", () => {
    expect(mrr(hits, ["c/three.ts"])).toBeCloseTo(1 / 3, 10);
  });
  it("takes the FIRST relevant hit when several are relevant", () => {
    expect(mrr(hits, ["c/three.ts", "b/two.ts"])).toBe(0.5);
  });
  it("absent relevant path -> 0", () => {
    expect(mrr(hits, ["z/none.ts"])).toBe(0);
  });
  it("empty relevant set -> 0", () => {
    expect(mrr(hits, [])).toBe(0);
  });
});

describe("firstRelevantRank", () => {
  const hits = ["a.ts", "b.ts", "c.ts"];
  it("returns the 1-based rank of the first relevant path", () => {
    expect(firstRelevantRank(hits, ["b.ts"])).toBe(2);
  });
  it("returns null when no relevant path is present", () => {
    expect(firstRelevantRank(hits, ["z.ts"])).toBeNull();
  });
});

describe("corpus-level aggregates", () => {
  it("meanRecallAtK averages per-query recall", () => {
    const perQuery = [
      { hits: ["a", "b"], relevant: ["a"] }, // 1
      { hits: ["c", "d"], relevant: ["x"] }, // 0
    ];
    expect(meanRecallAtK(perQuery, 10)).toBe(0.5);
  });
  it("meanReciprocalRank averages per-query reciprocal ranks", () => {
    const perQuery = [
      { hits: ["a", "b"], relevant: ["a"] }, // 1
      { hits: ["c", "d"], relevant: ["d"] }, // 0.5
    ];
    expect(meanReciprocalRank(perQuery)).toBeCloseTo(0.75, 10);
  });
  it("empty query set -> 0 for both", () => {
    expect(meanRecallAtK([], 10)).toBe(0);
    expect(meanReciprocalRank([])).toBe(0);
  });
});

// ── Wave-1 eval math (AC-103): every expected value is hand-computed in a comment. ──

describe("ndcgAtK (gate math)", () => {
  const hits = ["a/one.ts", "b/two.ts", "c/three.ts", "d/four.ts", "e/five.ts"];

  it("single relevant at rank 1 -> 1.0", () => {
    // DCG = 1/log2(2) = 1; IDCG (1 item) = 1 → 1.0
    expect(ndcgAtK(hits, ["a/one.ts"], 10)).toBe(1);
  });

  it("single relevant at rank 3, k=10 -> 1/log2(4) = 0.5", () => {
    // DCG = 1/log2(3+1) = 0.5; IDCG (1 item) = 1 → 0.5
    expect(ndcgAtK(hits, ["c/three.ts"], 10)).toBeCloseTo(0.5, 10);
  });

  it("two relevant at ranks 2+4, k=10, |relevant|=2", () => {
    // DCG  = 1/log2(3) + 1/log2(5)
    // IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    const dcg = 1 / Math.log2(3) + 1 / Math.log2(5);
    const idcg = 1 + 1 / Math.log2(3);
    expect(ndcgAtK(hits, ["b/two.ts", "d/four.ts"], 10)).toBeCloseTo(
      dcg / idcg,
      10,
    );
  });

  it("relevant absent from hits -> 0", () => {
    expect(ndcgAtK(hits, ["z/none.ts"], 10)).toBe(0);
  });

  it("k = 0 -> 0", () => {
    expect(ndcgAtK(hits, ["a/one.ts"], 0)).toBe(0);
  });

  it("empty relevant set -> 0", () => {
    expect(ndcgAtK(hits, [], 10)).toBe(0);
  });

  it("k < |relevant| truncates the IDCG (3 relevant, k=2, hits at ranks 1+2 -> 1.0)", () => {
    // DCG  = 1/log2(2) + 1/log2(3)
    // IDCG = ideal for min(3, 2) = 2 items = the same sum → 1.0
    expect(ndcgAtK(hits, ["a/one.ts", "b/two.ts", "z/none.ts"], 2)).toBeCloseTo(
      1,
      10,
    );
  });

  it("relevant outside top-k does not count (relevant at rank 3, k=2 -> 0)", () => {
    expect(ndcgAtK(hits, ["c/three.ts"], 2)).toBe(0);
  });

  it("normalizes paths (case, whitespace, leading ./)", () => {
    expect(ndcgAtK(["  ./A/One.TS  "], ["a/one.ts"], 1)).toBe(1);
  });

  it("credits a duplicated relevant path ONCE (same file at ranks 1+2 -> exactly 1.0)", () => {
    // one-credit rule: DCG = 1/log2(2) = 1 — the rank-2 duplicate adds NOTHING.
    // IDCG (1 item) = 1 → 1.0. Per-occurrence crediting would give
    // 1 + 1/log2(3) ≈ 1.63 > 1, biasing the regression gate (chunk-keyed hit lists
    // routinely carry the same file twice).
    const v = ndcgAtK(["a/one.ts", "a/one.ts", "b/two.ts"], ["a/one.ts"], 10);
    expect(v).toBe(1);
    expect(v).toBeLessThanOrEqual(1);
  });

  it("a duplicate occupies a rank slot (masks a later second relevant) but never double-credits", () => {
    // hits = [one, one, two], relevant = {one, two}:
    // DCG  = 1/log2(2) [one@1] + 0 [duplicate] + 1/log2(4) [two pushed to rank 3]
    // IDCG = 1/log2(2) + 1/log2(3)
    const dcg = 1 + 1 / Math.log2(4);
    const idcg = 1 + 1 / Math.log2(3);
    expect(
      ndcgAtK(
        ["a/one.ts", "a/one.ts", "b/two.ts"],
        ["a/one.ts", "b/two.ts"],
        10,
      ),
    ).toBeCloseTo(dcg / idcg, 10);
  });
});

describe("mulberry32 (deterministic PRNG)", () => {
  it("same seed -> same first 5 values", () => {
    const a = mulberry32(1234);
    const b = mulberry32(1234);
    expect([a(), a(), a(), a(), a()]).toEqual([b(), b(), b(), b(), b()]);
  });

  it("values lie in [0, 1)", () => {
    const r = mulberry32(7);
    for (let i = 0; i < 100; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });
});

describe("pairedBootstrapCI (gate math)", () => {
  it("constant deltas [0.1 x20] -> mean 0.1, ciLow = ciHigh = 0.1", () => {
    // Every resample mean of a constant array is that constant.
    const ci = pairedBootstrapCI(Array(20).fill(0.1), { iterations: 500 });
    expect(ci.mean).toBeCloseTo(0.1, 10);
    expect(ci.ciLow).toBeCloseTo(0.1, 10);
    expect(ci.ciHigh).toBeCloseTo(0.1, 10);
  });

  it("all-negative deltas -> CI below zero -> regressionExcluded true", () => {
    const ci = pairedBootstrapCI(Array(20).fill(-0.05), { iterations: 500 });
    expect(ci.ciHigh).toBeLessThan(0);
    expect(regressionExcluded(ci)).toBe(true);
  });

  it("mixed +/- deltas straddling zero -> regressionExcluded false", () => {
    // Symmetric ±0.1: resample means straddle 0, so the 95% CI contains it.
    const deltas = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1];
    const ci = pairedBootstrapCI(deltas, { iterations: 2000, seed: 7 });
    expect(ci.ciLow).toBeLessThan(0);
    expect(ci.ciHigh).toBeGreaterThan(0);
    expect(regressionExcluded(ci)).toBe(false);
  });

  it("is deterministic: same seed twice -> identical output", () => {
    const deltas = [0.02, -0.01, 0.05, 0.0, -0.03, 0.04, 0.01];
    const a = pairedBootstrapCI(deltas, { iterations: 1000, seed: 99 });
    const b = pairedBootstrapCI(deltas, { iterations: 1000, seed: 99 });
    expect(a).toEqual(b);
  });

  it("empty input -> zeros", () => {
    expect(pairedBootstrapCI([])).toEqual({ mean: 0, ciLow: 0, ciHigh: 0 });
  });

  it("CI brackets the observed mean", () => {
    const deltas = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25, 0.12, 0.18];
    const ci = pairedBootstrapCI(deltas, { iterations: 2000, seed: 3 });
    expect(ci.ciLow).toBeLessThanOrEqual(ci.mean);
    expect(ci.ciHigh).toBeGreaterThanOrEqual(ci.mean);
  });
});

describe("regressionExcluded (gate semantics)", () => {
  it("whole CI below zero -> true (delta = candidate - baseline)", () => {
    expect(regressionExcluded({ ciLow: -0.2, ciHigh: -0.01 })).toBe(true);
  });
  it("CI containing zero -> false", () => {
    expect(regressionExcluded({ ciLow: -0.05, ciHigh: 0.05 })).toBe(false);
  });
  it("whole CI above zero (a win) -> false", () => {
    expect(regressionExcluded({ ciLow: 0.01, ciHigh: 0.2 })).toBe(false);
  });
});

// The fixture graduated at gate G0 (2026-07-03): paths re-verified against the live
// index, cross-repo disambiguation applied, splits assigned and the test split FROZEN
// (AC-109). Sanitized for the public template repo (2026-07-21): the original 15
// rows scoring a second, private downstream repo were removed — the remaining 52 seed
// rows score only this repo's own services/ tree. The pre-freeze seed-state assertions
// (`_seed === true`, all-dev, all seed-v1) are retired for the frozen-gold contract.
describe("frozen code-eval fixture (AC-030, v2, approved at G0)", () => {
  it("loads and is shaped { k, rows } with ≥50 rows", () => {
    const ev = loadCodeEval();
    expect(typeof ev.k).toBe("number");
    expect(ev.k).toBe(10); // Recall@10 / MRR@10
    expect(Array.isArray(ev.rows)).toBe(true);
    expect(ev.rows.length).toBeGreaterThanOrEqual(50);
    for (const row of ev.rows) {
      expect(typeof row.query).toBe("string");
      expect(row.query.length).toBeGreaterThan(0);
      expect(Array.isArray(row.relevantPaths)).toBe(true);
      expect(row.relevantPaths.length).toBeGreaterThan(0);
      for (const p of row.relevantPaths) expect(typeof p).toBe("string");
    }
  });

  it("is v2: version header + changelog, unique c-NNN ids, split + provenance on every row", () => {
    const ev = loadCodeEval();
    expect(ev.version).toBe(2);
    expect(Array.isArray(ev.changelog)).toBe(true);
    expect(ev.changelog!.length).toBeGreaterThan(0);
    const ids = new Set<string>();
    for (const row of ev.rows) {
      expect(row.id).toMatch(/^c-\d{3}$/);
      ids.add(row.id!);
      expect(["dev", "test"]).toContain(row.split);
      expect(["seed-v1", "distilled"]).toContain(row.provenance);
    }
    expect(ids.size).toBe(ev.rows.length); // ids are unique (they key the gate join)
  });

  it("has a frozen non-empty test split (~1/3, promoted at G0) and is no longer a seed", () => {
    const ev = loadCodeEval();
    const test = filterRowsBySplit(ev, "test");
    expect(test.length).toBeGreaterThanOrEqual(15);
    expect(test.length).toBeLessThan(ev.rows.length / 2);
    expect(ev._seed).toBe(false);
  });

  it("covers the documented query archetypes", () => {
    const ev = loadCodeEval();
    const archetypes = new Set(ev.rows.map((r) => r.archetype).filter(Boolean));
    for (const a of [
      "where-is-X",
      "how-do-we-X",
      "symbol-lookup",
      "route-to-handler",
      "cross-file-usage",
    ]) {
      expect(archetypes.has(a)).toBe(true);
    }
  });
});

describe("loadCodeEval version tolerance", () => {
  const tmp = mkdtempSync(join(tmpdir(), "code-eval-load-"));
  afterAll(() => rmSync(tmp, { recursive: true, force: true }));

  it("tolerates a bare v1 file (bakeoff back-compat) — rows default to the dev split", () => {
    const p = join(tmp, "v1.json");
    writeFileSync(
      p,
      JSON.stringify({
        k: 10,
        rows: [{ query: "q", relevantPaths: ["a.ts"] }],
      }),
    );
    const ev = loadCodeEval(p);
    expect(ev.version).toBeUndefined();
    expect(ev.rows).toHaveLength(1);
    expect(filterRowsBySplit(ev, "dev")).toHaveLength(1);
    expect(filterRowsBySplit(ev, "test")).toHaveLength(0);
  });

  it("rejects v2 rows missing id or split", () => {
    const p = join(tmp, "bad-v2.json");
    writeFileSync(
      p,
      JSON.stringify({
        version: 2,
        k: 10,
        rows: [{ query: "q", relevantPaths: ["a.ts"] }],
      }),
    );
    expect(() => loadCodeEval(p)).toThrow(/v2 rows need an id/);
  });

  it("rejects unknown versions", () => {
    const p = join(tmp, "v3.json");
    writeFileSync(p, JSON.stringify({ version: 3, k: 10, rows: [] }));
    expect(() => loadCodeEval(p)).toThrow(/not supported/);
  });
});
