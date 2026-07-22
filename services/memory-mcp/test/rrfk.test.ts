// Wave 4 (retrieval token efficiency) — injectable RRF_K (AC-815). Covers:
//   AC-815 default   — fuseCandidates without rrfK is indistinguishable from an
//                      explicit rrfK: 60 call: identical SQL text AND identical bound
//                      parameters (pool ordering itself is pinned by search-golden
//                      on :5544).
//   AC-815 override  — a finite rrfK > 0 reaches BOTH RRF terms through ONE bound
//                      parameter ($5::float8) and changes nothing else: the SQL text
//                      is k-invariant; only the bound value differs.
//   AC-815 guard     — non-finite or ≤ 0 rrfK rejects BEFORE issuing any SQL and
//                      BEFORE spending an embed call.
//
// (Wave-7 merge note: k was originally template-interpolated into the SQL text; the
// wave-7 review parameterized it — same contract, stronger hygiene. These assertions
// pin the parameterized mechanism.)
//
// Fully mocked (voyage module mock + pool.query spy) — no DATABASE_URL needed; the
// suite runs identically with and without a DB. NO live quota.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const H = vi.hoisted(() => ({ embedCalls: 0 }));

vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map(() => [0.1, 0.2, 0.3]),
  embedContextual: async (docs: string[][]) =>
    docs.map((d) => d.map(() => [0.1, 0.2, 0.3])),
  embedContextualSingle: async (texts: string[]) => {
    H.embedCalls += 1;
    return texts.map(() => [0.1, 0.2, 0.3]);
  },
  rerank: async (_query: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_, i) => ({ index: i, score: 0.5 })),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import { fuseCandidates } from "../src/memory.js";
import { pool as livePool } from "../src/db/pool.js";

type QueryFn = (...args: unknown[]) => unknown;

describe("injectable rrfK (AC-815)", () => {
  const captured: Array<{ sql: string; params: unknown[] }> = [];
  let spy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    captured.length = 0;
    H.embedCalls = 0;
    spy = vi
      .spyOn(livePool as unknown as Record<"query", QueryFn>, "query")
      .mockImplementation((...args: unknown[]) => {
        if (typeof args[0] === "string") {
          captured.push({
            sql: args[0],
            params: (args[1] as unknown[]) ?? [],
          });
        }
        return Promise.resolve({ rows: [] });
      });
  });

  afterEach(() => {
    spy.mockRestore();
  });

  async function capture(
    rrfK?: number,
  ): Promise<{ sql: string; params: unknown[] }> {
    captured.length = 0;
    await fuseCandidates({ projectId: "rrfk-test", query: "alpha beta", rrfK });
    expect(captured.length).toBe(1);
    return captured[0];
  }

  it("default (no rrfK) — identical SQL AND identical bound params to explicit rrfK: 60; k is $5 in both terms", async () => {
    const def = await capture(undefined);
    const sixty = await capture(60);
    expect(def.sql).toBe(sixty.sql);
    expect(def.params).toEqual(sixty.params);
    expect(def.params[4]).toBe(60);
    expect(def.sql).toContain("COALESCE(1.0/($5::float8+bm25.rank),0)");
    expect(def.sql).toContain("COALESCE(1.0/($5::float8+vec.rank),0)");
    // Exactly the two RRF terms read the k parameter — nowhere else.
    expect(def.sql.split("$5::float8").length - 1).toBe(2);
  });

  it.each([20, 120])(
    "override rrfK=%d — reaches both RRF terms via the bound param; SQL text is k-invariant",
    async (k) => {
      const def = await capture(undefined);
      const kd = await capture(k);
      expect(kd.sql).toBe(def.sql); // parameterized: text never changes with k
      expect(kd.params[4]).toBe(k);
      // Everything else bound identically.
      expect(kd.params.filter((_, i) => i !== 4)).toEqual(
        def.params.filter((_, i) => i !== 4),
      );
    },
  );

  it.each([Number.NaN, Infinity, -Infinity, 0, -5])(
    "invalid rrfK=%s — rejects BEFORE issuing SQL and BEFORE the embed call",
    async (k) => {
      await expect(
        fuseCandidates({ projectId: "rrfk-test", query: "alpha", rrfK: k }),
      ).rejects.toThrow(/invalid rrfK/);
      expect(captured.length).toBe(0);
      expect(H.embedCalls).toBe(0);
    },
  );

  it("valid rrfK still spends exactly one embed when qvec is absent", async () => {
    await capture(120);
    expect(H.embedCalls).toBe(1);
  });
});
