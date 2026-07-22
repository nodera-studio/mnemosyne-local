// AC-022: an UNSET contextual model env (VOYAGE_CONTEXT_MODEL) must fall back to the
// documented default and the service must start cleanly — i.e. assertServerConfig must
// NOT require it. config is read at import time, so we reset modules between cases.

import { afterEach, describe, expect, it, vi } from "vitest";

const ORIGINAL = { ...process.env };

afterEach(() => {
  process.env = { ...ORIGINAL };
  vi.resetModules();
});

describe("contextModel config default (AC-022)", () => {
  it("falls back to voyage-context-4 when VOYAGE_CONTEXT_MODEL is unset", async () => {
    vi.resetModules();
    delete process.env.VOYAGE_CONTEXT_MODEL;
    const { config } = await import("../src/config.js");
    expect(config.contextModel).toBe("voyage-context-4");
  });

  it("honors VOYAGE_CONTEXT_MODEL when set", async () => {
    vi.resetModules();
    process.env.VOYAGE_CONTEXT_MODEL = "voyage-context-3-custom";
    const { config } = await import("../src/config.js");
    expect(config.contextModel).toBe("voyage-context-3-custom");
  });

  it("legacy embedModel still defaults to voyage-4-large", async () => {
    vi.resetModules();
    delete process.env.VOYAGE_EMBED_MODEL;
    const { config } = await import("../src/config.js");
    expect(config.embedModel).toBe("voyage-4-large");
  });

  it("consolidateModel treats an EMPTY env string as unset (compose `${VAR:-}` materializes empty)", async () => {
    vi.resetModules();
    process.env.CONSOLIDATE_MODEL = "";
    const { config } = await import("../src/config.js");
    expect(config.consolidateModel).toBe("claude-haiku-4-5");
  });

  it("distillModel falls through empty DISTILL_MODEL and empty CONSOLIDATE_MODEL", async () => {
    vi.resetModules();
    process.env.DISTILL_MODEL = "";
    process.env.CONSOLIDATE_MODEL = "";
    const { config } = await import("../src/config.js");
    expect(config.distillModel).toBe("claude-haiku-4-5");
  });

  it("assertServerConfig does NOT require the context model (starts cleanly)", async () => {
    vi.resetModules();
    delete process.env.VOYAGE_CONTEXT_MODEL;
    process.env.DATABASE_URL = "postgres://x/y";
    process.env.VOYAGE_API_KEY = "test-key";
    const { assertServerConfig } = await import("../src/config.js");
    expect(() => assertServerConfig()).not.toThrow();
  });
});

// Wave-7 blend/decay pins fail LOUDLY on malformed values — a typo'd compose pin
// silently reverting to defaults would pass every post-swap gate while shipping the
// wrong config (review findings: both engines).
describe("blend/decay env pin validation (wave-7)", () => {
  const loadConfig = async (env: Record<string, string>) => {
    vi.resetModules();
    for (const [k, v] of Object.entries(env)) process.env[k] = v;
    return import("../src/config.js");
  };

  it("valid pins load and materialize", async () => {
    const { config } = await loadConfig({
      BLEND_FORM: "multiplicative",
      RECENCY_TAU_DAYS: "90",
      RECENCY_TAU_DAYS_BY_TYPE: '{"semantic":90}',
      DECAY_EXEMPT: "type:entity,source_kind:decision",
    });
    expect(config.blendConfig.form).toBe("multiplicative");
    expect(config.blendConfig.decay.tauDays).toBe(90);
    expect(config.blendConfig.decay.exempt).toEqual({
      types: ["entity"],
      sourceKinds: ["decision"],
    });
  });

  it.each([
    ["BLEND_FORM", "multiplicativ"], // typo'd enum
    ["DECAY_SHAPE", "linear"],
    ["RECENCY_TAU_DAYS", "0"], // zero τ: recency 0 for the past, Infinity for the future
    ["RECENCY_TAU_DAYS", "-30"], // negative τ INVERTS decay
    ["RECENCY_TAU_DAYS", "abc"],
    ["DECAY_POWER_EXPONENT", "0"],
    ["DECAY_POWER_EXPONENT", "-0.5"],
    ["RECENCY_TAU_DAYS_BY_TYPE", "not-json"],
    ["RECENCY_TAU_DAYS_BY_TYPE", '{"semantic":0}'],
    ["RECENCY_TAU_DAYS_BY_TYPE", '["semantic"]'],
    ["DECAY_EXEMPT", "type:entty"], // typo'd memory type
    ["DECAY_EXEMPT", "type:"], // empty payload
    ["DECAY_EXEMPT", "source_kind:"],
    ["DECAY_EXEMPT", "entity"], // missing prefix
  ])("%s=%s throws at load", async (key, value) => {
    await expect(loadConfig({ [key]: value })).rejects.toThrow();
  });

  it("blend weights accept 0 (a valid arm config, e.g. A4 relevance-only)", async () => {
    const { config } = await loadConfig({
      BLEND_W_RECENCY: "0",
      BLEND_W_IMPORTANCE: "0",
    });
    expect(config.blendConfig.weights.recency).toBe(0);
    expect(config.blendConfig.weights.importance).toBe(0);
  });
});
