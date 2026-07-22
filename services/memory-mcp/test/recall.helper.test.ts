import { describe, expect, it } from "vitest";
import {
  canonicalizeRanked,
  meanRecallAtK,
  meanReciprocalRank,
  mrr,
  mulberry32,
  ndcgAtK,
  pairedBootstrapCI,
  recallAtK,
  regressionExcluded,
  runRecallEval,
  signTest,
  tryLoadRecallEvalV2,
} from "./recall.helper.js";
import { config } from "../src/config.js";

describe("recallAtK (gate math)", () => {
  const hits = ["alpha", "beta", "gamma", "delta"];

  it("relevant at rank 1 -> 1", () => {
    expect(recallAtK(hits, ["alpha"], 10)).toBe(1);
  });

  it("relevant at rank 3, k=3 -> 1", () => {
    expect(recallAtK(hits, ["gamma"], 3)).toBe(1);
  });

  it("relevant at rank 3, k=2 -> 0 (outside top-k)", () => {
    expect(recallAtK(hits, ["gamma"], 2)).toBe(0);
  });

  it("not in hits -> 0", () => {
    expect(recallAtK(hits, ["zeta"], 10)).toBe(0);
  });

  it("empty relevant set -> 0", () => {
    expect(recallAtK(hits, [], 10)).toBe(0);
  });

  it("k <= 0 -> 0", () => {
    expect(recallAtK(hits, ["alpha"], 0)).toBe(0);
  });

  it("matches titles case-insensitively and ignores surrounding whitespace", () => {
    expect(recallAtK(["  Alpha  "], ["alpha"], 1)).toBe(1);
  });
});

describe("mrr (gate math)", () => {
  const hits = ["alpha", "beta", "gamma", "delta"];

  it("relevant at rank 1 -> 1.0", () => {
    expect(mrr(hits, ["alpha"])).toBe(1.0);
  });

  it("relevant at rank 2 -> 0.5", () => {
    expect(mrr(hits, ["beta"])).toBe(0.5);
  });

  it("relevant at rank 3 -> 1/3", () => {
    expect(mrr(hits, ["gamma"])).toBeCloseTo(1 / 3, 10);
  });

  it("takes the FIRST relevant hit when several are relevant", () => {
    expect(mrr(hits, ["gamma", "beta"])).toBe(0.5);
  });

  it("absent relevant title -> 0", () => {
    expect(mrr(hits, ["zeta"])).toBe(0);
  });

  it("empty relevant set -> 0", () => {
    expect(mrr(hits, [])).toBe(0);
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
  const hits = ["alpha", "beta", "gamma", "delta", "epsilon"];

  it("single relevant at rank 1 -> 1.0", () => {
    // DCG = 1/log2(2) = 1; IDCG (1 item) = 1 → 1.0
    expect(ndcgAtK(hits, ["alpha"], 10)).toBe(1);
  });

  it("single relevant at rank 3, k=10 -> 1/log2(4) = 0.5", () => {
    // DCG = 1/log2(3+1) = 0.5; IDCG (1 item) = 1 → 0.5
    expect(ndcgAtK(hits, ["gamma"], 10)).toBeCloseTo(0.5, 10);
  });

  it("two relevant at ranks 2+4, k=10, |relevant|=2", () => {
    // DCG  = 1/log2(3) + 1/log2(5)
    // IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    const dcg = 1 / Math.log2(3) + 1 / Math.log2(5);
    const idcg = 1 + 1 / Math.log2(3);
    expect(ndcgAtK(hits, ["beta", "delta"], 10)).toBeCloseTo(dcg / idcg, 10);
  });

  it("relevant absent from hits -> 0", () => {
    expect(ndcgAtK(hits, ["zeta"], 10)).toBe(0);
  });

  it("k = 0 -> 0", () => {
    expect(ndcgAtK(hits, ["alpha"], 0)).toBe(0);
  });

  it("empty relevant set -> 0", () => {
    expect(ndcgAtK(hits, [], 10)).toBe(0);
  });

  it("k < |relevant| truncates the IDCG (3 relevant, k=2, hits at ranks 1+2 -> 1.0)", () => {
    // DCG  = 1/log2(2) + 1/log2(3)
    // IDCG = ideal for min(3, 2) = 2 items = the same sum → 1.0
    expect(ndcgAtK(hits, ["alpha", "beta", "zeta"], 2)).toBeCloseTo(1, 10);
  });

  it("relevant outside top-k does not count (relevant at rank 3, k=2 -> 0)", () => {
    expect(ndcgAtK(hits, ["gamma"], 2)).toBe(0);
  });

  it("matches case-insensitively with surrounding whitespace (normalize)", () => {
    expect(ndcgAtK(["  Alpha  "], ["alpha"], 1)).toBe(1);
  });

  it("credits a duplicated relevant key ONCE (same key at ranks 1+2 -> exactly 1.0)", () => {
    // one-credit rule: DCG = 1/log2(2) = 1 — the rank-2 duplicate adds NOTHING.
    // IDCG (1 item) = 1 → 1.0. Per-occurrence crediting would give
    // 1 + 1/log2(3) ≈ 1.63 > 1, biasing the regression gate.
    const v = ndcgAtK(["alpha", "alpha", "beta"], ["alpha"], 10);
    expect(v).toBe(1);
    expect(v).toBeLessThanOrEqual(1);
  });

  it("a duplicate occupies a rank slot (masks a later second relevant) but never double-credits", () => {
    // hits = [alpha, alpha, beta], relevant = {alpha, beta}:
    // DCG  = 1/log2(2) [alpha@1] + 0 [duplicate] + 1/log2(4) [beta pushed to rank 3]
    // IDCG = 1/log2(2) + 1/log2(3)
    const dcg = 1 + 1 / Math.log2(4);
    const idcg = 1 + 1 / Math.log2(3);
    expect(
      ndcgAtK(["alpha", "alpha", "beta"], ["alpha", "beta"], 10),
    ).toBeCloseTo(dcg / idcg, 10);
  });
});

describe("canonicalizeRanked (AC-106 gold-chain canonicalization)", () => {
  // Two gold ids whose supersession chains CONVERGE on one winner `w` (consolidation
  // merged both gold-referenced memories into the same survivor).
  const converging = () =>
    new Map<string, Set<string>>([
      ["g1", new Set(["g1", "w"])],
      ["g2", new Set(["g2", "w"])],
    ]);

  it("two golds converging on one winner: a shared-member hit credits any not-yet-credited gold", () => {
    // g1 is consumed by its OWN member at rank 1; the shared winner at rank 2 must then
    // stand for g2 (a first-wins member→gold map would map w→g1 and orphan g2).
    expect(canonicalizeRanked(["g1", "w"], converging())).toEqual(["g1", "g2"]);
  });

  it("a single shared-member hit credits exactly one gold (chains insertion order)", () => {
    expect(canonicalizeRanked(["w"], converging())).toEqual(["g1"]);
  });

  it("exhausted golds pass through as the raw id — no gold is ever credited twice", () => {
    expect(canonicalizeRanked(["w", "w", "x"], converging())).toEqual([
      "g1",
      "g2",
      "x",
    ]);
  });

  it("single chain: a second member of an already-credited gold passes through raw", () => {
    const chains = new Map<string, Set<string>>([
      ["g", new Set(["g", "mid", "w"])],
    ]);
    expect(canonicalizeRanked(["mid", "w"], chains)).toEqual(["g", "w"]);
  });

  it("ids matching no chain pass through unchanged", () => {
    expect(canonicalizeRanked(["x", "y"], new Map())).toEqual(["x", "y"]);
  });
});

describe("mulberry32 (deterministic PRNG)", () => {
  it("same seed -> same first 5 values", () => {
    const a = mulberry32(1234);
    const b = mulberry32(1234);
    const seqA = [a(), a(), a(), a(), a()];
    const seqB = [b(), b(), b(), b(), b()];
    expect(seqA).toEqual(seqB);
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

// ── Wave-7 sign test (bakeoff verdict corroboration) — hand-computed values ─────────

describe("signTest (two-sided, continuity-corrected normal approximation)", () => {
  it("counts wins/losses/ties and excludes ties from the test", () => {
    const r = signTest([0.1, 0.2, -0.05, 0, 0, 0.3]);
    expect(r.wins).toBe(3);
    expect(r.losses).toBe(1);
    expect(r.ties).toBe(2);
    // n = 4 non-zero: z = (|3 − 2| − 0.5)/√1 = 0.5 → p = 2·(1 − Φ(0.5)) ≈ 0.6171
    expect(r.p).toBeCloseTo(0.6171, 3);
  });

  it("8 wins / 2 losses → z = 2.5/√2.5, p ≈ 0.1138 (hand-computed)", () => {
    const deltas = [...Array(8).fill(0.01), ...Array(2).fill(-0.01)];
    const r = signTest(deltas);
    expect(r.wins).toBe(8);
    expect(r.losses).toBe(2);
    expect(r.ties).toBe(0);
    // z = (|8 − 5| − 0.5)/√(10/4) = 2.5/1.58114 = 1.58114 → p = 2·(1 − Φ(1.58114))
    expect(r.p).toBeCloseTo(0.1138, 3);
  });

  it("9 wins / 1 loss → p ≈ 0.0269 (hand-computed)", () => {
    const r = signTest([...Array(9).fill(1), -1]);
    // z = (|9 − 5| − 0.5)/√(10/4) = 3.5/1.58114 = 2.21359 → p = 2·(1 − Φ(2.21359))
    expect(r.p).toBeCloseTo(0.0269, 3);
  });

  it("balanced 5/5 → the continuity correction floors z at 0 → p = 1", () => {
    const r = signTest([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]);
    expect(r.p).toBe(1);
  });

  it("all ties (or empty) → p = 1, no evidence either way", () => {
    expect(signTest([0, 0, 0])).toEqual({ wins: 0, losses: 0, ties: 3, p: 1 });
    expect(signTest([])).toEqual({ wins: 0, losses: 0, ties: 0, p: 1 });
  });

  it("more lopsided splits give smaller p (monotone in |wins − losses|)", () => {
    const p6 = signTest([...Array(6).fill(1), ...Array(4).fill(-1)]).p;
    const p8 = signTest([...Array(8).fill(1), ...Array(2).fill(-1)]).p;
    const p10 = signTest(Array(10).fill(1)).p;
    expect(p8).toBeLessThan(p6);
    expect(p10).toBeLessThan(p8);
    expect(p10).toBeGreaterThan(0);
  });

  it("p is clamped to [0, 1]", () => {
    for (const deltas of [[1], [1, -1], Array(200).fill(1)]) {
      const { p } = signTest(deltas);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
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

// The fixture graduated at gate G0 (2026-07-03): the operator approved the v2
// stable-id gold and renamed it over recall-eval.json (AC-109), so the v1
// seed-shape assertions are retired and the frozen-gold contract is pinned instead.
// `loadRecallEval` stays exported as gold:migrate's v1-input reader.
describe("frozen gold fixture (v2, approved at G0)", () => {
  it("loads as v2 with ≥30 rows, every row carrying non-empty gold + approval", () => {
    const ev = tryLoadRecallEvalV2();
    expect(ev).not.toBeNull();
    expect(typeof ev!.k).toBe("number");
    expect(ev!.rows.length).toBeGreaterThanOrEqual(30);
    for (const row of ev!.rows) {
      expect(typeof row.query).toBe("string");
      expect(row.relevantIds.length).toBeGreaterThan(0);
      expect(row.approvedBy).toBeTruthy();
    }
  });

  it("has a frozen non-empty test split and a changelog recording approval", () => {
    const ev = tryLoadRecallEvalV2()!;
    const test = ev.rows.filter((r) => r.split === "test");
    expect(test.length).toBeGreaterThanOrEqual(10);
    expect(ev.changelog.some((line) => /APPROVED/.test(line))).toBe(true);
  });
});

// DB-backed runner — needs loopback Postgres AND a LIVE Voyage key (it embeds each
// query through the real /embeddings|/contextualizedembeddings endpoint, burning quota).
// Per recall-eval.md it is "run deliberately, not on every CI push", so it is gated on
// an explicit opt-in (`VOYAGE_LIVE_TESTS=1`) in addition to DATABASE_URL — otherwise a
// DB-only run (e.g. the backfill/recall-gate suites) would 401 on a non-live key.
// It also requires the gold file to be MIGRATED to v2 (stable ids) — until the operator
// approves + renames the `gold:migrate` proposal, it skips with a note.
const LIVE =
  !!process.env.DATABASE_URL && process.env.VOYAGE_LIVE_TESTS === "1";
describe.skipIf(!LIVE)("runRecallEval (DB-backed, two layers)", () => {
  it("runs both layers over the v2 gold set and returns the artifact payload", async () => {
    const ev = tryLoadRecallEvalV2();
    if (!ev) {
      console.error(
        "runRecallEval: gold file still v1 — run `npm run gold:migrate`, approve, rename (AC-109)",
      );
      expect(ev).toBeNull(); // structural assertion so the test is not empty
      return;
    }
    const result = await runRecallEval(ev, {
      projectId: config.defaultProjectId,
    });
    expect(result.perQuery.length).toBe(ev.rows.length);
    for (const agg of [
      result.aggregates.recallAt25,
      result.aggregates.ndcgAt10,
      result.aggregates.mrr10,
    ]) {
      expect(agg).toBeGreaterThanOrEqual(0);
      expect(agg).toBeLessThanOrEqual(1);
    }
    expect(result.retrievalConfig).toMatchObject({ service: "memory-mcp" });
  });
});
