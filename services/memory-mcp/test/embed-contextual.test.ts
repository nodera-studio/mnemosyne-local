// AC-020: embedContextual() POSTs to /v1/contextualizedembeddings with nested inputs,
// output_dimension 1024, output_dtype int8, and returns 1024-length vectors. Fully
// MOCKED fetch — NO live Voyage quota is burned here.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  embedContextual,
  embedContextualSingle,
  rerank,
  toVectorLiteral,
} from "../src/voyage.js";

// A captured request so each test can assert on the exact body/url Voyage was called with.
interface Captured {
  url: string;
  body: any;
}

/** Build a fake /contextualizedembeddings response: int8 vectors of `dim` length, one
 *  per chunk, nested per doc to mirror the real { data: [{ data: [{embedding}] }] }. */
function fakeContextualResponse(docChunkCounts: number[], dim = 1024) {
  let seed = 1;
  return {
    data: docChunkCounts.map((nChunks, docIdx) => ({
      index: docIdx,
      data: Array.from({ length: nChunks }, (_, chunkIdx) => ({
        index: chunkIdx,
        text: `chunk ${docIdx}.${chunkIdx}`,
        // small 8-bit integers — what output_dtype:int8 returns
        embedding: Array.from(
          { length: dim },
          () => (seed = (seed * 7 + 3) % 251) - 125, // deterministic ints in [-125,125]
        ),
      })),
    })),
    model: "voyage-context-4",
    usage: { total_tokens: 42 },
  };
}

let captured: Captured[];

beforeEach(() => {
  process.env.VOYAGE_API_KEY ??= "test-key";
  captured = [];
});

afterEach(() => {
  vi.restoreAllMocks();
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

describe("embedContextual (AC-020)", () => {
  it("POSTs to /v1/contextualizedembeddings with nested inputs + int8 + dim 1024", async () => {
    mockFetch(fakeContextualResponse([2, 1]));
    const docs = [["doc1 chunk1", "doc1 chunk2"], ["doc2 chunk1"]];

    const out = await embedContextual(docs, "document");

    expect(captured).toHaveLength(1);
    const { url, body } = captured[0];
    // endpoint
    expect(url).toBe("https://api.voyageai.com/v1/contextualizedembeddings");
    // nested inputs (one inner array per doc, ordered chunks)
    expect(body.inputs).toEqual(docs);
    expect(Array.isArray(body.inputs[0])).toBe(true);
    // model + flexible-dim + int8 quantization
    expect(body.model).toBe("voyage-context-4");
    expect(body.input_type).toBe("document");
    expect(body.output_dimension).toBe(1024);
    expect(body.output_dtype).toBe("int8");

    // shape: doc → chunk → vector, preserving structure
    expect(out).toHaveLength(2);
    expect(out[0]).toHaveLength(2); // doc1: 2 chunks
    expect(out[1]).toHaveLength(1); // doc2: 1 chunk
    // each vector is 1024-dim (dimension-safe for halfvec(1024))
    expect(out[0][0]).toHaveLength(1024);
    expect(out[1][0]).toHaveLength(1024);
  });

  it("returns [] for empty input without calling fetch", async () => {
    mockFetch(fakeContextualResponse([]));
    const out = await embedContextual([], "document");
    expect(out).toEqual([]);
    expect(captured).toHaveLength(0);
  });

  it("query embedding uses input_type=query", async () => {
    mockFetch(fakeContextualResponse([1]));
    await embedContextual([["what did we decide"]], "query");
    expect(captured[0].body.input_type).toBe("query");
  });
});

describe("rerank model override (wave-6)", () => {
  it("POSTs the passed model when provided", async () => {
    mockFetch({
      data: [{ index: 1, relevance_score: 0.9 }],
    });

    const out = await rerank(
      "where is the rollout note",
      ["doc a", "doc b"],
      5,
      "rerank-2.5",
    );

    expect(captured).toHaveLength(1);
    expect(captured[0].url).toBe("https://api.voyageai.com/v1/rerank");
    expect(captured[0].body).toMatchObject({
      query: "where is the rollout note",
      documents: ["doc a", "doc b"],
      model: "rerank-2.5",
      top_k: 2,
    });
    expect(out).toEqual([{ index: 1, score: 0.9 }]);
  });
});

describe("embedContextualSingle (one-chunk-per-doc helper, AC-020)", () => {
  it("wraps each text as [[text]] and returns one 1024-len vector per text", async () => {
    mockFetch(fakeContextualResponse([1, 1, 1]));
    const out = await embedContextualSingle(["a", "b", "c"], "document");

    // wrapped as nested single-chunk docs
    expect(captured[0].body.inputs).toEqual([["a"], ["b"], ["c"]]);
    // one vector per text, each 1024-dim
    expect(out).toHaveLength(3);
    for (const v of out) expect(v).toHaveLength(1024);
  });

  it("int8 vectors serialize into a halfvec literal as plain integers", async () => {
    mockFetch(fakeContextualResponse([1], 4)); // tiny dim for a readable literal
    const [vec] = await embedContextualSingle(["x"], "document");
    const literal = toVectorLiteral(vec);
    // halfvec literal is `[i1,i2,...]` of integers — no decimals, no cast needed
    expect(literal).toMatch(/^\[-?\d+(,-?\d+)*\]$/);
  });

  it("returns [] for empty input", async () => {
    mockFetch(fakeContextualResponse([]));
    expect(await embedContextualSingle([], "document")).toEqual([]);
    expect(captured).toHaveLength(0);
  });
});
