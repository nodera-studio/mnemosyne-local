// AC-030: regression guard for the CHOSEN code embedder. Runs the labeled eval set
// against the LIVE searchCode path and asserts Recall@10 / MRR@10 stay within sane bounds
// (and, once the operator freezes a baseline, at/above the recorded chosen-arm numbers in
// test/bakeoff.md). It embeds each query through the real Voyage endpoint and hits the
// live search path, so it BURNS QUOTA and needs a populated corpus — it is gated on an
// explicit opt-in (`VOYAGE_LIVE_TESTS=1`) on top of DATABASE_URL, exactly like memory-mcp's
// recall runner. Off by default so the normal DB suite (which uses a non-live key) never
// 401s here.

import { describe, expect, it } from "vitest";
import { loadCodeEval, runCodeRecallEval } from "./code-eval.helper.js";
import { config } from "../src/config.js";

// Once the operator freezes the eval set and records the chosen arm's numbers in
// test/bakeoff.md, raise these to the recorded baseline so a future regression fails CI.
// Until then they are loose floors (the corpus + held-out set may be small).
const RECALL_FLOOR = Number(process.env.CODE_RECALL_FLOOR ?? "0");
const MRR_FLOOR = Number(process.env.CODE_MRR_FLOOR ?? "0");

const LIVE =
  !!process.env.DATABASE_URL && process.env.VOYAGE_LIVE_TESTS === "1";

describe.skipIf(!LIVE)("code recall on the live search path (AC-030)", () => {
  it("eval set scores within bounds and meets the recorded floor", async () => {
    const ev = loadCodeEval();
    const repo = process.env.CODE_RECALL_REPO; // optional repo filter
    const result = await runCodeRecallEval(ev, {
      projectId: config.defaultProjectId,
      repo,
    });

    expect(result.recallAtK).toBeGreaterThanOrEqual(0);
    expect(result.recallAtK).toBeLessThanOrEqual(1);
    expect(result.mrr).toBeGreaterThanOrEqual(0);
    expect(result.mrr).toBeLessThanOrEqual(1);
    expect(result.perQuery.length).toBe(ev.rows.length);

    // Regression floors (default 0 until a baseline is frozen post-bake-off).
    expect(result.recallAtK).toBeGreaterThanOrEqual(RECALL_FLOOR);
    expect(result.mrr).toBeGreaterThanOrEqual(MRR_FLOOR);
  });
});

// Infra-less guard so the file always has a running assertion: the eval set is loadable
// and large enough to be a meaningful instrument (≥50 rows).
describe("code-eval set is a usable instrument", () => {
  it("loads ≥50 rows at k=10", () => {
    const ev = loadCodeEval();
    expect(ev.rows.length).toBeGreaterThanOrEqual(50);
    expect(ev.k).toBe(10);
  });
});
