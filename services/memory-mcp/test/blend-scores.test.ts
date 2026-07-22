// Pure unit tests for `blendScores` (wave-7 Step 3, AC-701/AC-706). No DB, no network
// (importing memory.js creates the pg pool object but never connects). Every expected
// value is hand-computed in a comment.

import { describe, expect, it } from "vitest";
import { blendScores, type BlendConfig } from "../src/memory.js";

// Frozen scoring instant — all ages below are exact day counts from here.
const NOW = Date.UTC(2026, 6, 1, 12, 0, 0); // 2026-07-01T12:00:00Z
const DAY = 86_400_000;
const daysAgo = (d: number) => new Date(NOW - d * DAY);

/** The live default (A0): additive 0.7/0.2/0.1, exp decay, τ30, nothing exempt. */
function a0(): BlendConfig {
  return {
    form: "additive",
    weights: { relevance: 0.7, recency: 0.2, importance: 0.1 },
    decay: {
      shape: "exp",
      tauDays: 30,
      tauDaysByType: {},
      powerExponent: 0.5,
      exempt: { types: [], sourceKinds: [] },
    },
  };
}

/** Recency-isolating config: final = recency exactly (weights 0/1/0). */
function recencyOnly(decay: Partial<BlendConfig["decay"]> = {}): BlendConfig {
  const base = a0();
  return {
    ...base,
    weights: { relevance: 0, recency: 1, importance: 0 },
    decay: { ...base.decay, ...decay },
  };
}

interface TestCand {
  id: string;
  type: "episodic" | "semantic" | "procedural" | "entity";
  importance: number;
  created_at: Date;
  event_date?: Date | null;
  source_kind?: string | null;
  relevance: number;
}

function cand(over: Partial<TestCand> & { id: string }): TestCand {
  return {
    type: "semantic",
    importance: 0.5,
    created_at: daysAgo(10),
    event_date: null,
    source_kind: null,
    relevance: 0.5,
    ...over,
  };
}

describe("blendScores — A0 reproduces the legacy formula (AC-701)", () => {
  it("hand-computed legacy final at age 30d: 0.7·rel + 0.2·e⁻¹ + 0.1·imp", () => {
    const [r] = blendScores(
      [
        cand({
          id: "x",
          relevance: 0.6,
          importance: 0.8,
          created_at: daysAgo(30),
        }),
      ],
      a0(),
      NOW,
    );
    // 0.7·0.6 + 0.2·e^(−30/30) + 0.1·0.8 = 0.42 + 0.2·0.36787944117 + 0.08
    expect(r.final).toBeCloseTo(0.42 + 0.2 * Math.exp(-1) + 0.08, 12);
  });

  it("relevance-0 fallback (candidate missing from rerank): final = 0.2·recency + 0.1·imp", () => {
    const [r] = blendScores(
      [
        cand({
          id: "x",
          relevance: 0,
          importance: 0.4,
          created_at: daysAgo(60),
        }),
      ],
      a0(),
      NOW,
    );
    // 0 + 0.2·e^(−2) + 0.04
    expect(r.final).toBeCloseTo(0.2 * Math.exp(-2) + 0.04, 12);
  });

  it("age 0 (created now): recency term is exactly the full weight", () => {
    const [r] = blendScores(
      [
        cand({
          id: "x",
          relevance: 0.5,
          importance: 0.5,
          created_at: new Date(NOW),
        }),
      ],
      a0(),
      NOW,
    );
    // 0.35 + 0.2·e⁰ + 0.05 = 0.6
    expect(r.final).toBeCloseTo(0.6, 12);
  });
});

describe("blendScores — decay shapes (AC-706)", () => {
  it("exp at age = τ gives e⁻¹ ≈ 0.3679; power gives 2^−0.5 ≈ 0.7071", () => {
    const row = [cand({ id: "x", created_at: daysAgo(30) })];
    const [exp] = blendScores(row, recencyOnly({ shape: "exp" }), NOW);
    const [pow] = blendScores(
      row,
      recencyOnly({ shape: "power", powerExponent: 0.5 }),
      NOW,
    );
    expect(exp.final).toBeCloseTo(Math.exp(-1), 12); // ≈ 0.36788
    expect(pow.final).toBeCloseTo(1 / Math.sqrt(2), 12); // ≈ 0.70711
    // Power decays SLOWER than exp at age = τ — the whole point of the A2 arm.
    expect(pow.final).toBeGreaterThan(exp.final);
  });

  it("power uses the configured exponent: (1 + age/τ)^−2 at age = τ → 0.25", () => {
    const [r] = blendScores(
      [cand({ id: "x", created_at: daysAgo(30) })],
      recencyOnly({ shape: "power", powerExponent: 2 }),
      NOW,
    );
    expect(r.final).toBeCloseTo(0.25, 12);
  });

  it("per-type τ routing: tauDaysByType[type] ?? tauDays", () => {
    const cfg = recencyOnly({ tauDaysByType: { semantic: 90 } });
    const [sem] = blendScores(
      [cand({ id: "s", type: "semantic", created_at: daysAgo(90) })],
      cfg,
      NOW,
    );
    const [epi] = blendScores(
      [cand({ id: "e", type: "episodic", created_at: daysAgo(90) })],
      cfg,
      NOW,
    );
    expect(sem.final).toBeCloseTo(Math.exp(-1), 12); // τ90 routed
    expect(epi.final).toBeCloseTo(Math.exp(-3), 12); // falls back to τ30
  });
});

describe("blendScores — exemptions (AC-706)", () => {
  const cfg = recencyOnly({
    exempt: { types: ["entity"], sourceKinds: ["decision"] },
  });

  it("type:entity gets recency exactly 1.0 regardless of age", () => {
    const [r] = blendScores(
      [cand({ id: "x", type: "entity", created_at: daysAgo(300) })],
      cfg,
      NOW,
    );
    expect(r.final).toBe(1);
  });

  it("source_kind:decision gets recency exactly 1.0 regardless of age", () => {
    const [r] = blendScores(
      [
        cand({
          id: "x",
          type: "semantic",
          source_kind: "decision",
          created_at: daysAgo(300),
        }),
      ],
      cfg,
      NOW,
    );
    expect(r.final).toBe(1);
  });

  it("non-exempt rows still decay", () => {
    const [r] = blendScores(
      [cand({ id: "x", type: "semantic", created_at: daysAgo(300) })],
      cfg,
      NOW,
    );
    expect(r.final).toBeCloseTo(Math.exp(-10), 12);
  });
});

describe("blendScores — multiplicative form", () => {
  it("rel × (1 + w.recency·recency + w.importance·importance), hand-computed", () => {
    const cfg: BlendConfig = {
      ...a0(),
      form: "multiplicative",
      weights: { relevance: 1, recency: 0.2, importance: 0.05 },
    };
    const [r] = blendScores(
      [
        cand({
          id: "x",
          relevance: 0.6,
          importance: 0.8,
          created_at: daysAgo(30),
        }),
      ],
      cfg,
      NOW,
    );
    // 0.6 · (1 + 0.2·e⁻¹ + 0.05·0.8)
    expect(r.final).toBeCloseTo(
      0.6 * (1 + 0.2 * Math.exp(-1) + 0.05 * 0.8),
      12,
    );
  });

  it("relevance 0 zeroes the multiplicative final (no additive floor)", () => {
    const cfg: BlendConfig = {
      ...a0(),
      form: "multiplicative",
      weights: { relevance: 1, recency: 0.2, importance: 0.05 },
    };
    const [r] = blendScores(
      [
        cand({
          id: "x",
          relevance: 0,
          importance: 1,
          created_at: new Date(NOW),
        }),
      ],
      cfg,
      NOW,
    );
    expect(r.final).toBe(0);
  });
});

describe("blendScores — ordering contract (AC-701)", () => {
  it("A4 (weights 1/0/0) ordering ≡ rerank (relevance) ordering", () => {
    const cfg: BlendConfig = {
      ...a0(),
      weights: { relevance: 1, recency: 0, importance: 0 },
    };
    const rows = [
      cand({ id: "mid", relevance: 0.5, created_at: daysAgo(1) }),
      cand({ id: "hi", relevance: 0.9, created_at: daysAgo(300) }),
      cand({ id: "lo", relevance: 0.2, created_at: new Date(NOW) }),
    ];
    const out = blendScores(rows, cfg, NOW);
    expect(out.map((r) => r.id)).toEqual(["hi", "mid", "lo"]);
    expect(out.map((r) => r.final)).toEqual([0.9, 0.5, 0.2]);
  });

  it("legacy comparator preserved: equal finals keep INPUT order (stable sort, no tie-breakers)", () => {
    const twin = (id: string) =>
      cand({ id, relevance: 0.5, importance: 0.5, created_at: daysAgo(5) });
    const forward = blendScores([twin("a"), twin("b"), twin("c")], a0(), NOW);
    expect(forward.map((r) => r.id)).toEqual(["a", "b", "c"]);
    // Reversed input keeps reversed output — proof there is NO hidden id tie-breaker.
    const reversed = blendScores([twin("c"), twin("b"), twin("a")], a0(), NOW);
    expect(reversed.map((r) => r.id)).toEqual(["c", "b", "a"]);
  });

  it("sorts DESC by final across mixed rows", () => {
    const rows = [
      cand({
        id: "old-important",
        relevance: 0.5,
        importance: 0.9,
        created_at: daysAgo(90),
      }),
      cand({
        id: "fresh",
        relevance: 0.5,
        importance: 0.1,
        created_at: new Date(NOW),
      }),
    ];
    const out = blendScores(rows, a0(), NOW);
    // fresh: 0.35 + 0.2 + 0.01 = 0.56; old-important: 0.35 + 0.2·e⁻³ + 0.09 ≈ 0.44996
    expect(out.map((r) => r.id)).toEqual(["fresh", "old-important"]);
    for (let i = 1; i < out.length; i++) {
      expect(out[i - 1].final).toBeGreaterThanOrEqual(out[i].final);
    }
  });
});

describe("blendScores — temporal source (AC-706)", () => {
  it("NULL event_date falls back to created_at", () => {
    const [withNull] = blendScores(
      [cand({ id: "x", created_at: daysAgo(30), event_date: null })],
      recencyOnly(),
      NOW,
    );
    expect(withNull.final).toBeCloseTo(Math.exp(-1), 12);
  });

  it("event_date wins over created_at when present", () => {
    const [r] = blendScores(
      [
        cand({
          id: "x",
          created_at: daysAgo(300), // would be e⁻¹⁰ if used
          event_date: daysAgo(30),
        }),
      ],
      recencyOnly(),
      NOW,
    );
    expect(r.final).toBeCloseTo(Math.exp(-1), 12);
  });

  it("negative age (future event_date) is NOT clamped — recency exceeds 1 under exp", () => {
    const [r] = blendScores(
      [cand({ id: "x", created_at: daysAgo(1), event_date: daysAgo(-30) })],
      recencyOnly(),
      NOW,
    );
    // age = −30d → e^(+1) ≈ 2.71828 — current behavior preserved, no clamp.
    expect(r.final).toBeCloseTo(Math.E, 12);
    expect(r.final).toBeGreaterThan(1);
  });
});
