// AC-104 groundwork: retrievalConfig() is the snapshot serialized into every eval-run
// artifact, so it must (a) carry every knob of the retrieval pipeline, (b) equal the
// constants the pipeline ACTUALLY runs with (single source — no drift), and (c) be
// JSON-safe (wave-2's artifact writer and wave-6/7's bakeoffs serialize it verbatim).
// Wave-7 (AC-702): the scalar recencyHalfLifeDays is REPLACED by the nested `blend`
// config + a top-level `scoringVersion` code constant.
// Pure unit test — no DB, no network (importing memory.js creates the pg pool object
// but never connects).

import { describe, expect, it } from "vitest";
import {
  RERANK_DOC_TRUNCATION,
  SCORING_VERSION,
  SNIPPET_CHARS,
  retrievalConfig,
} from "../src/memory.js";
import { config } from "../src/config.js";

describe("retrievalConfig() snapshot (memory-mcp)", () => {
  const snap = retrievalConfig();

  it("contains every retrieval knob", () => {
    expect(Object.keys(snap).sort()).toEqual(
      [
        "service",
        "scoringVersion",
        "rrfK",
        "blend",
        "candidatePool",
        "recallLimit",
        "embedModel",
        "contextModel",
        "rerankModel",
        "rerankDocTruncation",
        "rerankDocIncludesSummary",
        "snippetChars",
      ].sort(),
    );
    expect(Object.keys(snap.blend).sort()).toEqual([
      "decay",
      "form",
      "weights",
    ]);
    expect(Object.keys(snap.blend.weights).sort()).toEqual([
      "importance",
      "recency",
      "relevance",
    ]);
    expect(Object.keys(snap.blend.decay).sort()).toEqual([
      "exempt",
      "powerExponent",
      "shape",
      "tauDays",
      "tauDaysByType",
    ]);
    expect(Object.keys(snap.blend.decay.exempt).sort()).toEqual([
      "sourceKinds",
      "types",
    ]);
  });

  it("matches the tuned constants and live config values", () => {
    expect(snap.service).toBe("memory-mcp");
    expect(snap.scoringVersion).toBe(SCORING_VERSION);
    expect(snap.scoringVersion).toBe("blend-2");
    expect(snap.rrfK).toBe(60);
    // Defaults = the pre-wave-7 live behavior (A0): additive 0.7/0.2/0.1, exp decay,
    // τ30 for every type, nothing exempt. A bakeoff winner ships as env pins only.
    expect(snap.blend).toEqual({
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
    // The snapshot serializes the SAME object the pipeline runs with (single source).
    expect(snap.blend).toEqual(config.blendConfig);
    expect(snap.recallLimit).toBe(50);
    expect(snap.candidatePool).toBe(config.candidatePool);
    expect(snap.embedModel).toBe(config.embedModel);
    expect(snap.contextModel).toBe(config.contextModel);
    expect(snap.rerankModel).toBe(config.rerankModel);
  });

  it("drops the retired scalar recencyHalfLifeDays (wave-7 rename to blend.decay.tauDays)", () => {
    expect("recencyHalfLifeDays" in snap).toBe(false);
  });

  it("serializes the SAME truncation constants rerankAndBlend runs with", () => {
    expect(snap.rerankDocTruncation).toBe(RERANK_DOC_TRUNCATION);
    expect(snap.snippetChars).toBe(SNIPPET_CHARS);
    expect(RERANK_DOC_TRUNCATION).toBe(1200);
    expect(SNIPPET_CHARS).toBe(180);
    // Wave-2 (AC-810): reranker docs carry the stored summary as a prefix line.
    expect(snap.rerankDocIncludesSummary).toBe(true);
  });

  it("is JSON-safe (round-trips losslessly)", () => {
    expect(JSON.parse(JSON.stringify(snap))).toEqual(snap);
  });

  it("returns fresh literals — mutating a snapshot never reaches live config", () => {
    const a = retrievalConfig();
    a.blend.weights.relevance = 999;
    a.blend.decay.exempt.types.push("entity");
    expect(retrievalConfig().blend.weights.relevance).toBe(0.7);
    expect(retrievalConfig().blend.decay.exempt.types).toEqual([]);
    expect(config.blendConfig.weights.relevance).toBe(0.7);
  });
});
