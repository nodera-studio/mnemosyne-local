// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Wave 7 (blend/decay bakeoff)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/
// wave-7-blend-decay-bakeoff.md, AC-701…AC-706 + the 2026-07-04 review-cycle
// guarantees). Every assertion below was drafted from the plan text and the
// exported type surface only — never from the implementation bodies. This suite
// is the independent anti-reward-hacking gate, additive to the Implementer's own
// blend-scores/bakeoff-blend/config-default tests.
//
// AC map:
//   AC-701 — blendScores under the default (A0) config reproduces the LEGACY
//            final by hand-computation (0.7·rel + 0.2·e^(−age/30) + 0.1·imp),
//            DESC by `b.final − a.final` with STABLE tie order; a candidate the
//            reranker did not score blends with relevance 0 (via rerankAndBlend).
//   AC-702 — retrievalConfig() exposes nested blend {form, weights, decay{shape,
//            tauDays, tauDaysByType, powerExponent, exempt}} + top-level
//            scoringVersion; the scalar recencyHalfLifeDays is GONE; JSON
//            round-trips verbatim; defaults = the live A0 knobs.
//   AC-703 — the bakeoff calls fuse + rerank exactly ONCE per query for the
//            single-k blend arms while ALL arms score; with the k axis the
//            fuse/rerank counts scale by |k grid| = {20,60,120} but the query
//            embed stays ONE per query (qvec reuse).
//   AC-704 — parse layer refuses without --yes BEFORE touching deps and defaults
//            to the dev split; the run refuses <15-row splits AND any
//            empty-relevantIds row before spend; the artifact carries the full
//            retrievalConfig() snapshot, every arm's config, per-query scores,
//            seed 42, and rejectedForNow: ["access-based decay"].
//   AC-705 — per-arm-vs-control verdict fields (meanDelta, ci, signTest,
//            affected, sliceByType + sliceTemporal); winner rule: CI excluding
//            zero on the win side + wins > losses + affected > 0, highest mean
//            delta wins; no qualifier ⇒ KEEP A0. Both outcomes are forced with
//            synthetic captured relevances.
//   AC-706 — effectiveDate = event_date ?? created_at; per-type τ resolves
//            tauDaysByType[type] ?? tauDays; exemption by type AND by
//            source_kind pins recency to exactly 1.0; a future event_date
//            (negative age) is NOT clamped (recency > 1 under exp).
//   Review-cycle (binding, plan-level):
//     • fuseCandidates(rrfK): non-finite or ≤ 0 throws BEFORE any embed or SQL;
//       explicit rrfK=60 is ordering-identical to the default on a seeded corpus;
//       the fused candidates carry source_kind (the decay-exemption prerequisite).
//     • env pins fail loudly: RECENCY_TAU_DAYS=-30 and DECAY_EXEMPT=type:bogus
//       reject at config import.
//
// Deterministic throughout: Voyage is module-mocked; the bakeoff runs on
// injected fuse/rerank/embed deps with a frozen `now`; zero network anywhere
// (asserted). DB tests are self-contained under conf-w7-* project ids.
//
// Run:
//   npx tsc -p tsconfig.test.json --noEmit
//   DATABASE_URL=postgresql://postgres:postgres@localhost:5544/postgres \
//     npx vitest run conformance-w7
// ─────────────────────────────────────────────────────────────────────────────

import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic helpers, hoisted so the vi.mock factory can use them.
const H = vi.hoisted(() => {
  function fnv1a(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }
  function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function fakeVec(key: string, dim = 1024): number[] {
    const rnd = mulberry32(fnv1a(key));
    return Array.from({ length: dim }, () =>
      Number((rnd() * 2 - 1).toFixed(3)),
    );
  }
  // Doc contents carry a `relscore=<x>` marker; docs WITHOUT the marker are
  // omitted from the rerank result — that is exactly the "missing rerank score"
  // path AC-701 pins to relevance 0.
  function markerRerank(
    docs: string[],
  ): Array<{ index: number; score: number }> {
    return docs
      .map((d, index) => ({ index, m: /relscore=([0-9.]+)/.exec(d) }))
      .filter((x) => x.m !== null)
      .map((x) => ({ index: x.index, score: Number(x.m![1]) }))
      .sort((a, b) => b.score - a.score);
  }
  const counters = { embed: 0, embedContextual: 0, embedContextualSingle: 0 };
  return { fnv1a, mulberry32, fakeVec, markerRerank, counters };
});

// HARD RULE: never call live Voyage — module-mock the boundary (with counters).
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => {
    H.counters.embed++;
    return texts.map((t) => H.fakeVec(`t:${t}`));
  },
  embedContextual: async (docs: unknown[]) => {
    H.counters.embedContextual++;
    return docs.map((d) =>
      Array.isArray(d)
        ? (d as string[]).map((t) => H.fakeVec(`t:${t}`))
        : H.fakeVec(`t:${String(d)}`),
    );
  },
  embedContextualSingle: async (texts: string[]) => {
    H.counters.embedContextualSingle++;
    return texts.map((t) => H.fakeVec(`t:${t}`));
  },
  rerank: async (_query: string, docs: string[], _topK: number) =>
    H.markerRerank(docs),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";
process.env.ANTHROPIC_API_KEY ??= "test-key";

import {
  blendScores,
  fuseCandidates,
  rerankAndBlend,
  retrievalConfig,
  type BlendableCandidate,
  type BlendConfig,
  type DecayConfig,
  type FusedCandidate,
  type MemoryType,
} from "../src/memory.js";
import { config } from "../src/config.js";
import {
  formatVerdict,
  parseBakeoffBlendArgs,
  predeclaredArms,
  runBakeoffBlend,
  writeBakeoffBlendArtifact,
  type BakeoffBlendArtifact,
  type BakeoffBlendDeps,
} from "../src/db/bakeoff-blend.js";
import type { RecallEvalFileV2, RecallEvalRowV2 } from "../src/eval-core.js";

const here = dirname(fileURLToPath(import.meta.url));
const serviceRoot = join(here, "..");

const totalEmbedCalls = () =>
  H.counters.embed +
  H.counters.embedContextual +
  H.counters.embedContextualSingle;

const DAY = 86_400_000;
/** Frozen scoring instant for all hand-computed age math. */
const NOW = Date.UTC(2026, 5, 1);
const d = (ageDays: number) => new Date(NOW - ageDays * DAY);

/** The plan's A0/live default, written out from the spec — NOT read from code. */
const a0Config = (): BlendConfig => ({
  form: "additive",
  weights: { relevance: 0.7, recency: 0.2, importance: 0.1 },
  decay: {
    shape: "exp",
    tauDays: 30,
    tauDaysByType: {},
    powerExponent: 0.5,
    exempt: { types: [], sourceKinds: [] },
  },
});

/** The legacy final formula, verbatim from the plan (AC-701). */
const legacyFinal = (rel: number, ageDays: number, imp: number): number =>
  0.7 * rel + 0.2 * Math.exp(-ageDays / 30) + 0.1 * imp;

interface TestCand extends BlendableCandidate {
  id: string;
}
function tc(id: string, over: Partial<BlendableCandidate> = {}): TestCand {
  return {
    id,
    type: "semantic",
    importance: 0.5,
    created_at: new Date(NOW),
    event_date: null,
    source_kind: null,
    relevance: 0.5,
    ...over,
  };
}

const tauFor = (decay: DecayConfig, t: MemoryType): number =>
  decay.tauDaysByType[t] ?? decay.tauDays;

// nDCG@10 for a single gold item found at rank 1 vs rank 2 (binary gains).
const D1 = 1 - 1 / Math.log2(3);

// ═════════════════════════════ PURE (no DB) ══════════════════════════════════

describe("W7 AC-701 — pure blendScores reproduces the legacy score", () => {
  it("hand-computed finals under the default config; DESC order; ties keep input order", () => {
    const rows = [
      tc("low", { relevance: 0.4, created_at: d(60), importance: 0.2 }),
      tc("tie-first", { relevance: 0.5, created_at: d(30) }),
      tc("top", { relevance: 0.9, created_at: d(3), importance: 0.8 }),
      tc("tie-second", { relevance: 0.5, created_at: d(30) }),
      tc("mid", { relevance: 0.7, created_at: d(0) }),
    ];
    const out = blendScores(rows, a0Config(), NOW);

    // Legacy comparator b.final − a.final, stable on ties (no new tie-breakers):
    // the two identical candidates stay in input order.
    expect(out.map((r) => r.id)).toEqual([
      "top",
      "mid",
      "tie-first",
      "tie-second",
      "low",
    ]);
    for (let i = 1; i < out.length; i++) {
      expect(out[i - 1].final).toBeGreaterThanOrEqual(out[i].final);
    }

    const expected: Record<string, number> = {
      top: legacyFinal(0.9, 3, 0.8),
      mid: legacyFinal(0.7, 0, 0.5),
      "tie-first": legacyFinal(0.5, 30, 0.5),
      "tie-second": legacyFinal(0.5, 30, 0.5),
      low: legacyFinal(0.4, 60, 0.2),
    };
    for (const r of out) expect(r.final).toBeCloseTo(expected[r.id], 10);
  });

  it("a candidate the reranker did not score blends with relevance 0 (rerankAndBlend, default config)", async () => {
    const nowMs = Date.now();
    const fused = (
      id: string,
      content: string,
      ageDays: number,
      importance: number,
    ): FusedCandidate => ({
      id,
      title: "w7 conformance candidate",
      content,
      summary: null,
      type: "semantic",
      importance,
      created_at: new Date(nowMs - ageDays * DAY),
      event_date: null,
      source_kind: null,
      status: "active",
      rrf: 0.03,
      bm25_rank: 1,
      vec_rank: 2,
    });
    // The module-mocked reranker scores only docs carrying a relscore marker —
    // "unranked" has none, so its rerank score is MISSING ⇒ relevance 0.
    const rows = [
      fused("unranked", "no marker here at all", 15, 0.6),
      fused("strong", "claim relscore=0.90 end", 3, 0.8),
      fused("weak", "claim relscore=0.60 end", 45, 0.4),
    ];
    const hits = await rerankAndBlend("w7 conformance probe", rows, 3);

    expect(hits.map((h) => h.id)).toEqual(["strong", "weak", "unranked"]);
    const expected: Record<string, number> = {
      strong: legacyFinal(0.9, 3, 0.8),
      weak: legacyFinal(0.6, 45, 0.4),
      unranked: legacyFinal(0, 15, 0.6), // relevance 0 fallback
    };
    // Hit scores are 4-decimal-rounded finals; allow the rounding half-step
    // plus the sub-second age drift of the un-injectable Date.now().
    for (const h of hits) {
      expect(Math.abs(h.score - expected[h.id])).toBeLessThan(6e-5);
    }
  });
});

describe("W7 AC-702 — retrievalConfig() nested blend + scoringVersion", () => {
  it("exposes the nested blend config, drops recencyHalfLifeDays, and JSON round-trips verbatim", () => {
    const c = retrievalConfig();

    expect(c.scoringVersion).toBe("blend-2");
    expect(Object.keys(c.blend).sort()).toEqual(["decay", "form", "weights"]);
    expect(Object.keys(c.blend.weights).sort()).toEqual([
      "importance",
      "recency",
      "relevance",
    ]);
    expect(Object.keys(c.blend.decay).sort()).toEqual([
      "exempt",
      "powerExponent",
      "shape",
      "tauDays",
      "tauDaysByType",
    ]);
    expect(Object.keys(c.blend.decay.exempt).sort()).toEqual([
      "sourceKinds",
      "types",
    ]);

    // Env keys are ALWAYS materialized to their defaults (stable key set):
    expect(c.blend.form).toBe("additive");
    expect(c.blend.weights).toEqual({
      relevance: 0.7,
      recency: 0.2,
      importance: 0.1,
    });
    expect(c.blend.decay.shape).toBe("exp");
    expect(c.blend.decay.tauDays).toBe(30);
    expect(c.blend.decay.tauDaysByType).toEqual({});
    expect(c.blend.decay.powerExponent).toBe(0.5);
    expect(c.blend.decay.exempt).toEqual({ types: [], sourceKinds: [] });
    expect(c.rrfK).toBe(60); // golden-pinned default (2026-07-04 amendment)

    // The old scalar is GONE — everywhere in the serialized shape.
    expect("recencyHalfLifeDays" in c).toBe(false);
    const json = JSON.stringify(c);
    expect(json).not.toContain("recencyHalfLifeDays");

    // JSON-safe by construction: round-trips verbatim.
    expect(JSON.parse(json)).toEqual(c);
  });
});

describe("W7 AC-706 — decay semantics of pure blendScores", () => {
  it("effectiveDate = event_date ?? created_at (event_date wins; null falls back)", () => {
    const rows = [
      // created today but the EVENT was 90 days ago → age 90 must be used
      tc("evented", { created_at: d(0), event_date: d(90) }),
      // no event_date → created_at is the fallback (age 30)
      tc("created-only", { created_at: d(30), event_date: null }),
    ];
    const out = blendScores(rows, a0Config(), NOW);
    const byId = Object.fromEntries(out.map((r) => [r.id, r.final]));
    expect(byId["evented"]).toBeCloseTo(legacyFinal(0.5, 90, 0.5), 10);
    expect(byId["created-only"]).toBeCloseTo(legacyFinal(0.5, 30, 0.5), 10);
  });

  it("per-type τ resolves tauDaysByType[type] ?? tauDays", () => {
    const cfg = a0Config();
    cfg.decay.tauDaysByType = { semantic: 90 };
    const rows = [
      tc("sem", { type: "semantic", created_at: d(90) }), // τ90 → e^−1
      tc("epi", { type: "episodic", created_at: d(90) }), // fallback τ30 → e^−3
    ];
    const out = blendScores(rows, cfg, NOW);
    const byId = Object.fromEntries(out.map((r) => [r.id, r.final]));
    expect(byId["sem"]).toBeCloseTo(
      0.7 * 0.5 + 0.2 * Math.exp(-1) + 0.1 * 0.5,
      10,
    );
    expect(byId["epi"]).toBeCloseTo(
      0.7 * 0.5 + 0.2 * Math.exp(-3) + 0.1 * 0.5,
      10,
    );
  });

  it("exemption by type AND by source_kind pins recency to exactly 1.0", () => {
    const cfg = a0Config();
    cfg.decay.exempt = { types: ["entity"], sourceKinds: ["decision"] };
    const rows = [
      tc("by-type", { type: "entity", created_at: d(3650) }),
      tc("by-source", {
        type: "semantic",
        source_kind: "decision",
        created_at: d(3650),
      }),
      tc("not-exempt", { type: "semantic", created_at: d(3650) }),
    ];
    const out = blendScores(rows, cfg, NOW);
    const byId = Object.fromEntries(out.map((r) => [r.id, r.final]));
    const noAgePenalty = 0.7 * 0.5 + 0.2 * 1.0 + 0.1 * 0.5; // recency = 1.0
    expect(byId["by-type"]).toBeCloseTo(noAgePenalty, 12);
    expect(byId["by-source"]).toBeCloseTo(noAgePenalty, 12);
    // The 10-year-old non-exempt sibling decays to ~zero recency.
    expect(byId["not-exempt"]).toBeCloseTo(0.7 * 0.5 + 0.1 * 0.5, 9);
  });

  it("a future event_date (negative age) is NOT clamped — recency > 1 under exp", () => {
    const rows = [tc("future", { created_at: d(0), event_date: d(-30) })];
    const [out] = blendScores(rows, a0Config(), NOW);
    // age = −30 days, τ = 30 → recency = e^{+1} ≈ 2.718 (unclamped).
    expect(out.final).toBeCloseTo(0.7 * 0.5 + 0.2 * Math.E + 0.1 * 0.5, 10);
    expect(out.final).toBeGreaterThan(0.7 * 0.5 + 0.2 * 1.0 + 0.1 * 0.5);
  });
});

describe("W7 — fuseCandidates rrfK validation (review-cycle, binding)", () => {
  it("rrfK 0 / -5 / NaN / Infinity throw BEFORE any embed call", async () => {
    for (const bad of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
      const embedsBefore = totalEmbedCalls();
      await expect(
        fuseCandidates({
          projectId: "conf-w7-never",
          query: "w7 invalid k probe",
          rrfK: bad,
        }),
      ).rejects.toThrow();
      expect(totalEmbedCalls()).toBe(embedsBefore); // zero spend pre-throw
    }
  });
});

describe("W7 AC-704 — bakeoff parse layer + operator script", () => {
  it("refuses without --yes (a refusal string, before any dep is touched) and defaults to the dev split", () => {
    const refusal = parseBakeoffBlendArgs([]);
    expect(typeof refusal).toBe("string");
    expect(refusal as string).toMatch(/--yes/);

    const args = parseBakeoffBlendArgs(["--yes"]);
    expect(typeof args).not.toBe("string");
    if (typeof args !== "string") {
      expect(args.yes).toBe(true);
      expect(args.split).toBe("dev"); // the frozen test split is never the default
      expect(args.projectId).toBe(config.defaultProjectId);
    }
  });

  it("package.json exposes the bakeoff:blend operator script; the script main-guards", () => {
    const pkg = JSON.parse(
      readFileSync(join(serviceRoot, "package.json"), "utf8"),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts["bakeoff:blend"]).toBeTruthy();
    const src = readFileSync(
      join(serviceRoot, "src", "db", "bakeoff-blend.ts"),
      "utf8",
    );
    expect(src).toMatch(/import\.meta\.url/);
  });
});

describe("W7 — the predeclared arm table is FROZEN to the plan", () => {
  it("5 blend arms (A0…A4, k=60) + 2 k arms (k=20/120 over the A0 blend); A0 equals the live config", () => {
    const arms = predeclaredArms();
    expect(arms).toHaveLength(7);
    const byName = Object.fromEntries(arms.map((a) => [a.name, a]));
    for (const n of ["A0", "A1", "A2", "A3", "A4"]) {
      expect(byName[n], `arm ${n} must be predeclared`).toBeTruthy();
      expect(byName[n].rrfK).toBe(60);
    }

    // A0 control ≡ live default (the drift-assert baseline) ≡ the plan's A0.
    expect(byName.A0.blend).toEqual(config.blendConfig);
    expect(byName.A0.blend).toEqual(a0Config());

    // A1 per-type exp: episodic τ30, semantic/procedural τ90; entity+decision exempt.
    const a1 = byName.A1.blend;
    expect(a1.form).toBe("additive");
    expect(a1.weights).toEqual({
      relevance: 0.7,
      recency: 0.2,
      importance: 0.1,
    });
    expect(a1.decay.shape).toBe("exp");
    expect(tauFor(a1.decay, "episodic")).toBe(30);
    expect(tauFor(a1.decay, "semantic")).toBe(90);
    expect(tauFor(a1.decay, "procedural")).toBe(90);
    expect(a1.decay.exempt.types).toContain("entity");
    expect(a1.decay.exempt.sourceKinds).toContain("decision");

    // A2 = A1's τ/exemptions with power decay 1/(1+age/τ)^0.5.
    const a2 = byName.A2.blend;
    expect(a2.form).toBe("additive");
    expect(a2.decay.shape).toBe("power");
    expect(a2.decay.powerExponent).toBe(0.5);
    expect(tauFor(a2.decay, "episodic")).toBe(30);
    expect(tauFor(a2.decay, "semantic")).toBe(90);
    expect(tauFor(a2.decay, "procedural")).toBe(90);
    expect(a2.decay.exempt.types).toContain("entity");
    expect(a2.decay.exempt.sourceKinds).toContain("decision");

    // A3 multiplicative rel·(1 + 0.2·recency + 0.05·importance), exp τ30, none exempt.
    const a3 = byName.A3.blend;
    expect(a3.form).toBe("multiplicative");
    expect(a3.weights.recency).toBe(0.2);
    expect(a3.weights.importance).toBe(0.05);
    expect(a3.decay.shape).toBe("exp");
    expect(tauFor(a3.decay, "semantic")).toBe(30);
    expect(a3.decay.exempt.types).toHaveLength(0);
    expect(a3.decay.exempt.sourceKinds).toHaveLength(0);

    // A4 relevance-only.
    const a4 = byName.A4.blend;
    expect(a4.form).toBe("additive");
    expect(a4.weights).toEqual({ relevance: 1, recency: 0, importance: 0 });

    // k axis: exactly the predeclared grid {20, 120} around the k=60 control,
    // each scoring the A0 blend so the k change is isolated.
    const kArms = arms.filter((a) => a.rrfK !== 60);
    expect(kArms.map((a) => a.rrfK).sort((x, y) => x - y)).toEqual([20, 120]);
    for (const ka of kArms) expect(ka.blend).toEqual(byName.A0.blend);
  });
});

describe("W7 — env pins fail loudly (review-cycle, binding)", () => {
  it("RECENCY_TAU_DAYS=-30 rejects at config import", async () => {
    vi.resetModules();
    vi.stubEnv("RECENCY_TAU_DAYS", "-30");
    try {
      await expect(import("../src/config.js")).rejects.toThrow(
        /RECENCY_TAU_DAYS/i,
      );
    } finally {
      vi.unstubAllEnvs();
      vi.resetModules();
    }
  });

  it("DECAY_EXEMPT=type:bogus (unknown memory type) rejects at config import", async () => {
    vi.resetModules();
    vi.stubEnv("DECAY_EXEMPT", "type:bogus");
    try {
      await expect(import("../src/config.js")).rejects.toThrow(/DECAY_EXEMPT/i);
    } finally {
      vi.unstubAllEnvs();
      vi.resetModules();
    }
  });
});

// ═════════════════════════════ DB-BACKED ═════════════════════════════════════

// Unique fixture namespace — never collides with other suites' project ids.
const P_BAKE = "conf-w7-bakeoff";
const P_RRF = "conf-w7-rrf";
const ALL_PROJECTS = [P_BAKE, P_RRF];

const wid7 = (n: number) =>
  `70000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// ── Synthetic bakeoff corpus (15 dev rows), engineered from the arm table ─────
//
// Q1 rows (10×): gold is SEMANTIC, 90 days old; decoy is EPISODIC, fresh.
//   winner-scenario relevances gold 0.9 / decoy 0.77 put the gold FIRST only
//   under A2 (power τ90: 0.63+0.2/√2+0.05 ≈ 0.821 > 0.789) and A4 (0.9 > 0.77);
//   A0 (0.690), A1 (0.754) and A3 (0.931 vs 0.943) keep the decoy first.
// Q2 rows (5×, facet "temporal"): gold is EPISODIC fresh, decoy EPISODIC 10y old;
//   relevances gold 0.8 / decoy 0.95 → every decay arm puts the gold first, but
//   A4 (relevance-only) puts the DECOY first — punishing A4's mean delta below
//   A2's, so the predeclared highest-mean-delta rule picks A2 uniquely.
// KEEP scenario: gold 0.9 / decoy 0.1 everywhere → all arms produce identical
//   permutations → zero deltas, zero affected → verdict KEEPs A0.
interface Fx {
  rowId: string;
  query: string;
  facet?: string;
  gold: { id: string; type: MemoryType; createdMs: number };
  decoy: { id: string; type: MemoryType; createdMs: number };
  rel: { win: { gold: number; decoy: number } };
}
const FX: Fx[] = [];
for (let i = 1; i <= 10; i++) {
  FX.push({
    rowId: `w7-q${String(i).padStart(2, "0")}`,
    query: `conf w7 probe ${String(i).padStart(2, "0")}`,
    gold: { id: wid7(i), type: "semantic", createdMs: NOW - 90 * DAY },
    decoy: { id: wid7(100 + i), type: "episodic", createdMs: NOW },
    rel: { win: { gold: 0.9, decoy: 0.77 } },
  });
}
for (let i = 11; i <= 15; i++) {
  FX.push({
    rowId: `w7-q${String(i).padStart(2, "0")}`,
    query: `conf w7 probe ${String(i).padStart(2, "0")}`,
    facet: "temporal",
    gold: { id: wid7(i), type: "episodic", createdMs: NOW },
    decoy: { id: wid7(100 + i), type: "episodic", createdMs: NOW - 3650 * DAY },
    rel: { win: { gold: 0.8, decoy: 0.95 } },
  });
}

const devRows: RecallEvalRowV2[] = FX.map((f) => ({
  id: f.rowId,
  query: f.query,
  relevantIds: [f.gold.id],
  split: "dev",
  provenance: "seed-v1",
  ...(f.facet ? { facet: f.facet } : {}),
}));
const evalFile: RecallEvalFileV2 = {
  version: 2,
  k: 10,
  changelog: [],
  rows: [
    ...devRows,
    {
      // A frozen-TEST-split row: a dev-split run must never score (or spend on) it.
      id: "w7-t16",
      query: "conf w7 held-out test row",
      relevantIds: [FX[0].gold.id],
      split: "test",
      provenance: "seed-v1",
    },
  ],
};

describe.skipIf(skip)("W7 conformance — DB-backed", () => {
  let db: pg.Pool;
  let tmp: string;
  const fetchCalls: unknown[][] = [];
  let origFetch: typeof fetch;

  // Structural pool dep (dodges pg overload variance in strict mode).
  const poolDep = {
    query: (text: string, params: unknown[]) =>
      db.query(text, params as unknown[] | undefined),
  };

  function makeDeps(scenario: "win" | "keep") {
    const counters = { fuse: 0, rerank: 0, embed: 0 };
    const rrfKs: number[] = [];
    const relOf = (fx: Fx, side: "gold" | "decoy"): number =>
      scenario === "win" ? fx.rel.win[side] : side === "gold" ? 0.9 : 0.1;
    const cand = (
      spec: { id: string; type: MemoryType; createdMs: number },
      rel: number,
    ): FusedCandidate => ({
      id: spec.id,
      title: "w7 bakeoff candidate",
      content: `w7 bakeoff candidate relscore=${rel.toFixed(2)} end`,
      summary: null,
      type: spec.type,
      importance: 0.5,
      created_at: new Date(spec.createdMs),
      event_date: null,
      source_kind: null,
      status: "active",
      rrf: 0.03,
      bm25_rank: 1,
      vec_rank: 2,
    });
    const fuse: typeof fuseCandidates = async (input) => {
      counters.fuse++;
      rrfKs.push(input.rrfK ?? 60);
      const fx = FX.find((f) => f.query === input.query);
      if (!fx) {
        throw new Error(
          `conformance-w7: fuse called for an unexpected query: ${input.query}`,
        );
      }
      // Decoy first in pool order — arms must EARN the gold's rank via scoring.
      return [
        cand(fx.decoy, relOf(fx, "decoy")),
        cand(fx.gold, relOf(fx, "gold")),
      ];
    };
    const deps: BakeoffBlendDeps = {
      pool: poolDep,
      fuse,
      rerankFn: async (_q, docs, _topK) => {
        counters.rerank++;
        return H.markerRerank(docs);
      },
      embedQuery: async (q) => {
        counters.embed++;
        return H.fakeVec(`w7:${q}`);
      },
      log: () => {},
      now: NOW,
    };
    return { deps, counters, rrfKs };
  }

  async function seedRow(r: {
    id: string;
    projectId: string;
    type: MemoryType;
    title: string;
    content: string;
    createdMs: number;
    sourceKind?: string | null;
    vec?: number[] | null;
  }): Promise<void> {
    await db.query(
      `INSERT INTO memory.memories
         (id, project_id, type, title, content, importance, created_at,
          source_kind, status, embedding_v2)
       VALUES ($1,$2,$3,$4,$5,0.5,$6,$7,'active',$8::halfvec)`,
      [
        r.id,
        r.projectId,
        r.type,
        r.title,
        r.content,
        new Date(r.createdMs).toISOString(),
        r.sourceKind ?? null,
        r.vec ? `[${r.vec.join(",")}]` : null,
      ],
    );
  }

  async function cleanup(): Promise<void> {
    await db.query(`DELETE FROM memory.memories WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Idempotent migration replay (HOLD files skipped) — self-sufficient on a fresh DB.
    const sqlDir = join(serviceRoot, "sql");
    const { readdirSync } = await import("node:fs");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();
    tmp = mkdtempSync(join(tmpdir(), "conf-w7-"));

    // Gold + decoy rows exist as REAL memories: gold-id resolution and the
    // per-type slice SELECT run against the actual DB, not a shaped stub.
    for (const f of FX) {
      await seedRow({
        id: f.gold.id,
        projectId: P_BAKE,
        type: f.gold.type,
        title: `w7 gold ${f.rowId}`,
        content: `w7 bakeoff gold row for ${f.query}`,
        createdMs: f.gold.createdMs,
      });
      await seedRow({
        id: f.decoy.id,
        projectId: P_BAKE,
        type: f.decoy.type,
        title: `w7 decoy ${f.rowId}`,
        content: `w7 bakeoff decoy row for ${f.query}`,
        createdMs: f.decoy.createdMs,
      });
    }

    // Small live-fusion corpus for the rrfK-default equivalence check.
    await seedRow({
      id: wid7(0x201),
      projectId: P_RRF,
      type: "semantic",
      title: "rrf corpus full match",
      content: "conformance rrf corpus probe alpha",
      createdMs: NOW - 1 * DAY,
      vec: H.fakeVec("rrf:1"),
    });
    await seedRow({
      id: wid7(0x202),
      projectId: P_RRF,
      type: "semantic",
      title: "rrf corpus decision row",
      content: "conformance rrf corpus beta",
      createdMs: NOW - 2 * DAY,
      sourceKind: "decision",
      vec: H.fakeVec("rrf:2"),
    });
    await seedRow({
      id: wid7(0x203),
      projectId: P_RRF,
      type: "semantic",
      title: "rrf corpus gamma",
      content: "conformance rrf gamma",
      createdMs: NOW - 3 * DAY,
      vec: H.fakeVec("rrf:3"),
    });
    await seedRow({
      id: wid7(0x204),
      projectId: P_RRF,
      type: "semantic",
      title: "rrf corpus delta",
      content: "conformance delta",
      createdMs: NOW - 4 * DAY,
      vec: H.fakeVec("rrf:4"),
    });

    // Zero-network tripwire for the whole DB block (Voyage mocked, deps injected).
    origFetch = globalThis.fetch;
    globalThis.fetch = (async (...args: unknown[]) => {
      fetchCalls.push(args);
      throw new Error("conformance-w7: network call escaped the mocks");
    }) as typeof fetch;
  }, 120_000);

  afterAll(async () => {
    globalThis.fetch = origFetch;
    await cleanup();
    await db.end();
    rmSync(tmp, { recursive: true, force: true });
  });

  // ── AC-703 / AC-704 / AC-705: the bakeoff on injected deps ─────────────────

  describe("AC-703/704/705 — capture-once bakeoff, artifact, winner rule", () => {
    let winner: ReturnType<typeof makeDeps>;
    let winnerArtifact: BakeoffBlendArtifact;

    beforeAll(async () => {
      winner = makeDeps("win");
      winnerArtifact = await runBakeoffBlend(winner.deps, {
        evalFile,
        projectId: P_BAKE,
        split: "dev",
        arms: ["A0", "A1", "A2", "A3", "A4"], // single-k: the 5 blend arms
      });
    }, 60_000);

    it("AC-703: fuse + rerank run exactly ONCE per query (single k) while ALL 5 arms score; embed once per query", () => {
      expect(winner.counters.fuse).toBe(15);
      expect(winner.counters.rerank).toBe(15);
      expect(winner.counters.embed).toBe(15);
      expect(new Set(winner.rrfKs)).toEqual(new Set([60])); // blend arms capture at the control k

      expect(Object.keys(winnerArtifact.arms).sort()).toEqual([
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
      ]);
      for (const arm of Object.values(winnerArtifact.arms)) {
        expect(arm.perQuery).toHaveLength(15);
        for (const q of arm.perQuery) {
          expect(Number.isFinite(q.ndcg)).toBe(true);
          expect(q.poolRecall).toBe(1); // gold always survives the 2-row pool
        }
      }
    });

    it("AC-704: dev-split default surface — the frozen test-split row is never scored", () => {
      expect(winnerArtifact.split).toBe("dev");
      expect(winnerArtifact.rows).toBe(15);
      for (const arm of Object.values(winnerArtifact.arms)) {
        expect(arm.perQuery.some((q) => q.id === "w7-t16")).toBe(false);
      }
    });

    it("AC-704: the artifact carries the config snapshot, per-arm config, per-query scores, seed 42, and the access-decay rejection record", () => {
      expect(winnerArtifact.seed).toBe(42);
      expect(winnerArtifact.config).toEqual(retrievalConfig());
      expect(winnerArtifact.rejectedForNow).toEqual(["access-based decay"]);
      expect(winnerArtifact.arms.A0.armConfig.rrfK).toBe(60);
      expect(winnerArtifact.arms.A0.armConfig.blend).toEqual(
        config.blendConfig,
      );
      for (const arm of Object.values(winnerArtifact.arms)) {
        expect(arm.armConfig.blend).toBeTruthy();
        expect(typeof arm.ndcgMean).toBe("number");
      }

      const path = writeBakeoffBlendArtifact(winnerArtifact, tmp);
      expect(path.startsWith(tmp)).toBe(true); // artifacts only in the dir we chose
      expect(path).toMatch(/bakeoff-blend\.json$/);
      expect(existsSync(path)).toBe(true);
      expect(JSON.parse(readFileSync(path, "utf8"))).toEqual(
        JSON.parse(JSON.stringify(winnerArtifact)),
      );
    });

    it("AC-705: verdict fields per arm; the engineered A2 qualifies (win-side CI, wins>losses, affected>0) and wins on mean delta", () => {
      expect(Object.keys(winnerArtifact.comparisons).sort()).toEqual([
        "A1",
        "A2",
        "A3",
        "A4",
      ]);
      for (const c of Object.values(winnerArtifact.comparisons)) {
        expect(typeof c.meanDelta).toBe("number");
        expect(typeof c.ci.ciLow).toBe("number");
        expect(typeof c.ci.ciHigh).toBe("number");
        expect(typeof c.signTest.wins).toBe("number");
        expect(typeof c.signTest.losses).toBe("number");
        expect(typeof c.signTest.ties).toBe("number");
        expect(typeof c.signTest.p).toBe("number");
        expect(typeof c.affected).toBe("number");
        expect(c.sliceByType).toBeTruthy();
      }

      const a2 = winnerArtifact.comparisons.A2;
      // 10 Q1 wins of +D1 each, 5 Q2 ties → mean = 10·D1/15.
      expect(a2.meanDelta).toBeCloseTo((10 * D1) / 15, 10);
      expect(a2.ci.ciLow).toBeGreaterThan(0); // CI excludes zero on the WIN side
      expect(a2.signTest).toMatchObject({ wins: 10, losses: 0, ties: 5 });
      expect(a2.affected).toBe(10);
      expect(a2.label).toBe("WIN");
      // Slices: semantic golds carry the win; temporal(episodic) golds are flat.
      expect(a2.sliceByType.semantic.n).toBe(10);
      expect(a2.sliceByType.semantic.meanDelta).toBeCloseTo(D1, 10);
      expect(a2.sliceByType.episodic.n).toBe(5);
      expect(a2.sliceByType.episodic.meanDelta).toBeCloseTo(0, 12);
      expect(a2.sliceTemporal).not.toBeNull();
      expect(a2.sliceTemporal!.n).toBe(5);
      expect(a2.sliceTemporal!.meanDelta).toBeCloseTo(0, 12);

      // A1/A3 reproduce A0's permutations exactly → zero affected queries.
      expect(winnerArtifact.comparisons.A1.affected).toBe(0);
      expect(winnerArtifact.comparisons.A3.affected).toBe(0);
      // A4 wins Q1 but loses every temporal row → strictly below A2's mean.
      const a4 = winnerArtifact.comparisons.A4;
      expect(a4.signTest).toMatchObject({ wins: 10, losses: 5 });
      expect(a4.meanDelta).toBeCloseTo((5 * D1) / 15, 10);
      expect(a4.meanDelta).toBeLessThan(a2.meanDelta);

      expect(winnerArtifact.verdict.winner).toBe("A2");
      expect(formatVerdict(winnerArtifact)).toContain("A2");
      // A0 sanity anchors: A2 ranks every gold first; A0 misses the 10 Q1 golds.
      expect(winnerArtifact.arms.A2.ndcgMean).toBeCloseTo(1, 12);
      expect(winnerArtifact.arms.A0.ndcgMean).toBeCloseTo(
        (10 * (1 - D1) + 5) / 15,
        10,
      );
    });

    it("AC-703 (k axis): fuse + rerank scale by the predeclared k grid {20,60,120}; the query embed stays ONE per query", async () => {
      const kAxis = makeDeps("win");
      const artifact = await runBakeoffBlend(kAxis.deps, {
        evalFile,
        projectId: P_BAKE,
        split: "dev",
        // no arms subset → ALL predeclared arms, including the two k arms
      });
      expect(kAxis.counters.embed).toBe(15); // qvec reused across k
      expect(kAxis.counters.fuse).toBe(45); // 15 queries × |{20,60,120}|
      expect(kAxis.counters.rerank).toBe(45); // one rerank per query per k
      const perK = new Map<number, number>();
      for (const k of kAxis.rrfKs) perK.set(k, (perK.get(k) ?? 0) + 1);
      expect(new Set(perK.keys())).toEqual(new Set([20, 60, 120]));
      for (const [, n] of perK) expect(n).toBe(15);

      expect(Object.keys(artifact.arms)).toHaveLength(7);
      // The k arms saw identical candidates here → zero delta → the blend
      // winner is unchanged by the k axis.
      expect(artifact.verdict.winner).toBe("A2");
    }, 60_000);

    it("AC-705: when NO arm qualifies, the verdict KEEPs A0", async () => {
      const keep = makeDeps("keep");
      const artifact = await runBakeoffBlend(keep.deps, {
        evalFile,
        projectId: P_BAKE,
        split: "dev",
        arms: ["A0", "A1", "A2", "A3", "A4"],
      });
      for (const c of Object.values(artifact.comparisons)) {
        expect(c.affected).toBe(0);
        expect(c.meanDelta).toBeCloseTo(0, 12);
      }
      expect(artifact.verdict.winner).toBe("A0");
      expect(formatVerdict(artifact)).toMatch(/KEEP|A0/);
    }, 60_000);
  });

  // ── AC-704: pre-spend guards ───────────────────────────────────────────────

  describe("AC-704 — pre-spend guards refuse before any capture", () => {
    it("a <15-row split throws with ZERO fuse/rerank/embed calls", async () => {
      const g = makeDeps("win");
      const small: RecallEvalFileV2 = {
        version: 2,
        k: 10,
        changelog: [],
        rows: devRows.slice(0, 14),
      };
      await expect(
        runBakeoffBlend(g.deps, {
          evalFile: small,
          projectId: P_BAKE,
          split: "dev",
        }),
      ).rejects.toThrow();
      expect(g.counters).toEqual({ fuse: 0, rerank: 0, embed: 0 });
    }, 30_000);

    it("an empty-relevantIds row throws with ZERO fuse/rerank/embed calls", async () => {
      const g = makeDeps("win");
      const holed: RecallEvalFileV2 = {
        version: 2,
        k: 10,
        changelog: [],
        rows: [...devRows.slice(0, 14), { ...devRows[14], relevantIds: [] }],
      };
      await expect(
        runBakeoffBlend(g.deps, {
          evalFile: holed,
          projectId: P_BAKE,
          split: "dev",
        }),
      ).rejects.toThrow();
      expect(g.counters).toEqual({ fuse: 0, rerank: 0, embed: 0 });
    }, 30_000);
  });

  // ── rrfK default equivalence + source_kind carriage (review-cycle) ─────────

  describe("fuseCandidates — rrfK=60 ≡ default; candidates carry source_kind", () => {
    it("explicit rrfK 60 produces the default's exact ordering on a seeded corpus", async () => {
      const qvec = H.fakeVec("rrf:2"); // pre-computed → no embed variance
      const base = await fuseCandidates({
        projectId: P_RRF,
        query: "conformance rrf corpus probe",
        qvec,
        limit: 10,
      });
      const explicit = await fuseCandidates({
        projectId: P_RRF,
        query: "conformance rrf corpus probe",
        qvec,
        rrfK: 60,
      });
      expect(base.length).toBeGreaterThan(0);
      expect(explicit.map((c) => c.id)).toEqual(base.map((c) => c.id));

      // AC-706 prerequisite: the fused SELECT carries source_kind so the
      // decision exemption is implementable downstream.
      for (const c of base) expect("source_kind" in c).toBe(true);
      const decision = base.find((c) => c.id === wid7(0x202));
      expect(decision).toBeTruthy();
      expect(decision!.source_kind).toBe("decision");
    }, 30_000);
  });

  it("zero network calls escaped the mocks across the whole DB suite", () => {
    expect(fetchCalls).toHaveLength(0);
  });
});
