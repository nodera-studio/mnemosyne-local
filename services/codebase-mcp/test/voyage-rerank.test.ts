// Wave 6: rerank() accepts an explicit model override for the lite-vs-full bakeoff.
// Fetch is fully mocked here — no live Voyage quota is burned.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { rerank } from "../src/voyage.js";

interface Captured {
  url: string;
  body: any;
}

let captured: Captured[];

beforeEach(() => {
  process.env.VOYAGE_API_KEY ??= "test-key";
  captured = [];
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function mockFetch(response: unknown, ok = true, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: any) => {
      captured.push({ url, body: JSON.parse(init.body) });
      return {
        ok,
        status,
        headers: { get: () => null },
        json: async () => response,
        text: async () => JSON.stringify(response),
      } as unknown as Response;
    }),
  );
}

describe("rerank model override (wave-6)", () => {
  it("POSTs the passed model when provided", async () => {
    mockFetch({
      data: [{ index: 0, relevance_score: 0.95 }],
    });

    const out = await rerank(
      "find the auth handler",
      ["src/auth.ts\nexport function auth() {}", "src/db.ts"],
      10,
      "rerank-2.5",
    );

    expect(captured).toHaveLength(1);
    expect(captured[0].url).toBe("https://api.voyageai.com/v1/rerank");
    expect(captured[0].body).toMatchObject({
      query: "find the auth handler",
      documents: ["src/auth.ts\nexport function auth() {}", "src/db.ts"],
      model: "rerank-2.5",
      top_k: 2,
    });
    expect(out).toEqual([{ index: 0, score: 0.95 }]);
  });
});
