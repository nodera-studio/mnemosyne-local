// AC-104 groundwork: retrievalConfig() is the snapshot serialized into every eval-run
// artifact, so it must (a) carry every knob of the retrieval pipeline, (b) equal the
// constants the pipeline ACTUALLY runs with (single source — no drift), and (c) be
// JSON-safe (wave-2's artifact writer and wave-6's bakeoff serialize it verbatim).
// Pure unit test — no DB, no network (importing search.js creates the pg pool object
// but never connects).

import { describe, expect, it } from "vitest";
import {
  RERANK_DOC_TRUNCATION,
  SNIPPET_CHARS,
  SNIPPET_LINES,
  retrievalConfig,
} from "../src/search.js";
import { config } from "../src/config.js";

describe("retrievalConfig() snapshot (codebase-mcp)", () => {
  const snap = retrievalConfig();

  it("contains every retrieval knob", () => {
    expect(Object.keys(snap).sort()).toEqual(
      [
        "service",
        "rrfK",
        "candidatePool",
        "recallLimit",
        "codeEmbedModel",
        "codeContextModel",
        "rerankModel",
        "maxMergedLines",
        "rerankDocTruncation",
        "snippetLines",
        "snippetChars",
      ].sort(),
    );
  });

  it("matches the tuned constants and live config values", () => {
    expect(snap.service).toBe("codebase-mcp");
    expect(snap.rrfK).toBe(60);
    expect(snap.recallLimit).toBe(50);
    expect(snap.candidatePool).toBe(config.candidatePool);
    expect(snap.codeEmbedModel).toBe(config.codeEmbedModel);
    expect(snap.codeContextModel).toBe(config.codeContextModel);
    expect(snap.rerankModel).toBe(config.rerankModel);
    expect(snap.maxMergedLines).toBe(config.maxMergedLines);
  });

  it("serializes the SAME shaping constants rerankCodeHits runs with", () => {
    expect(snap.rerankDocTruncation).toBe(RERANK_DOC_TRUNCATION);
    expect(snap.snippetLines).toBe(SNIPPET_LINES);
    expect(snap.snippetChars).toBe(SNIPPET_CHARS);
    expect(RERANK_DOC_TRUNCATION).toBe(1500);
    expect(SNIPPET_LINES).toBe(4);
    expect(SNIPPET_CHARS).toBe(240);
  });

  it("is JSON-safe (round-trips losslessly)", () => {
    expect(JSON.parse(JSON.stringify(snap))).toEqual(snap);
  });
});
