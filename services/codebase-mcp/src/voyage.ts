import { config } from "./config.js";

const BASE = "https://api.voyageai.com/v1";
const MAX_ATTEMPTS = 4;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function voyageFetch(path: string, body: unknown): Promise<Response> {
  let lastText = "";
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.voyageApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (res.ok) return res;
    const retryable = res.status === 429 || res.status >= 500;
    if (retryable && attempt < MAX_ATTEMPTS - 1) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const waitMs =
        retryAfter > 0
          ? retryAfter * 1000
          : Math.min(8000, 500 * 2 ** attempt) +
            Math.floor(Math.random() * 250);
      await sleep(waitMs);
      continue;
    }
    lastText = await res.text();
    throw new Error(`voyage ${path} ${res.status}: ${lastText}`);
  }
  throw new Error(
    `voyage ${path} failed after ${MAX_ATTEMPTS} attempts: ${lastText}`,
  );
}

/** Embed code with voyage-code-3 (1024-dim). Batches of up to 64 are recommended by caller. */
export async function embedCode(
  texts: string[],
  inputType: "document" | "query",
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const res = await voyageFetch("/embeddings", {
    input: texts,
    model: config.codeEmbedModel,
    input_type: inputType,
  });
  const json = (await res.json()) as { data: Array<{ embedding: number[] }> };
  return json.data.map((d) => d.embedding);
}

/**
 * Contextualized CODE embeddings (voyage-context-3) via POST /v1/contextualizedembeddings,
 * for the Wave 5 bake-off arm. `docs` is NESTED: one inner array per FILE, holding that
 * file's ordered chunks (`[[file1_chunk1, file1_chunk2], [file2_chunk1], ...]`). Each chunk
 * is encoded in the context of the other chunks from the SAME file — this file-grouping is
 * where context-3 might beat the code specialist, since a chunk gets its file's context.
 * Returns `number[][][]` — file → chunk → vector, preserving input order.
 *
 * Matryoshka 1024 + int8 (`output_dimension:1024`, `output_dtype:int8`) for apples-to-apples
 * with the incumbent halfvec(1024). int8 vectors serialize into a halfvec literal as plain
 * integers via `toVectorLiteral` (no separate cast).
 *
 * Per-request caps (enforced by the CALLER, not here): ≤1000 docs, ≤16k chunks, ≤120k total
 * tokens, ≤32k tokens per doc. A file whose chunks exceed 32k tokens/doc must be split across
 * requests by the caller (reuse the indexer's MAX_FILE_BYTES guard).
 */
export async function embedCodeContextual(
  docs: string[][],
  inputType: "document" | "query",
): Promise<number[][][]> {
  if (docs.length === 0) return [];
  const res = await voyageFetch("/contextualizedembeddings", {
    inputs: docs,
    model: config.codeContextModel,
    input_type: inputType,
    output_dimension: 1024,
    output_dtype: "int8",
  });
  // Response: { data: [ { data: [ { embedding, index, text }, ... ], index }, ... ] }
  const json = (await res.json()) as {
    data: Array<{ data: Array<{ embedding: number[] }> }>;
  };
  return json.data.map((doc) => doc.data.map((chunk) => chunk.embedding));
}

export async function rerank(
  query: string,
  documents: string[],
  topK: number,
  model: string = config.rerankModel,
): Promise<Array<{ index: number; score: number }>> {
  if (documents.length === 0) return [];
  const res = await voyageFetch("/rerank", {
    query,
    documents,
    model,
    top_k: Math.min(topK, documents.length),
  });
  const json = (await res.json()) as {
    data?: Array<{ index: number; relevance_score: number }>;
    results?: Array<{ index: number; relevance_score: number }>;
  };
  const arr = json.data ?? json.results ?? [];
  return arr.map((d) => ({ index: d.index, score: d.relevance_score }));
}

export function toVectorLiteral(v: number[]): string {
  return `[${v.join(",")}]`;
}
