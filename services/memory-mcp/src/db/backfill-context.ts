// One-shot, RESUMABLE operator script: re-embed every active memory through the
// contextual endpoint (voyage-context-4) into memory.memories.embedding_v2, in batches
// of ≤500. This is a DELIBERATE ops action that spends Voyage quota ($0.12/M for
// context-4, first 200M free) — it is NEVER run on container start. Gate: `npm run backfill:context`.
//
// Resumability: it selects only active rows where embedding_v2 IS NULL, so re-running
// continues where it stopped. The Retry-After/429 backoff is handled inside voyageFetch
// (embedContextualSingle → embedContextual → voyageFetch); we also process ONE batch at
// a time (no batch-level concurrency) so we never hammer the endpoint.
//
// Importer coupling: the distill/importer scripts dedup on content_sha256 and SKIP
// existing rows, so a one-shot re-embed of EXISTING rows must be done by THIS backfill
// (which targets rows by missing embedding_v2), NOT by re-running the importer.
// Re-running the importer will NOT re-embed existing memories.
//
// AC-021: must populate embedding_v2 for EVERY active row. Verify after:
//   SELECT count(*) FROM memory.memories
//   WHERE archived_at IS NULL AND COALESCE(status,'active')='active' AND embedding_v2 IS NULL;  -- → 0

import type { Pool } from "pg";
import { pool as defaultPool } from "./pool.js";
import { embedContextualSingle, toVectorLiteral } from "../voyage.js";

/** DB page size — how many pending rows we SELECT per round (for resumability/paging).
 *  The actual Voyage requests are token-budgeted SUB-batches of these rows (see below),
 *  because the 120k-token/request cap binds long before the 1000-doc cap. */
export const BACKFILL_BATCH = 500;

/** Voyage /contextualizedembeddings per-request caps: 120k tokens, 1000 docs, 32k tokens/doc.
 *  We budget conservatively under those. Token estimate = chars/4 (rough, deliberately low
 *  margin baked into the budgets). */
export const TOKEN_BUDGET = 100_000; // safety margin under the 120k/request cap
export const DOC_CAP = 1000; // Voyage hard cap on docs/request
export const MAX_DOC_CHARS = 110_000; // ≈27.5k tokens — keep each doc under the 32k/doc cap
const estTokens = (s: string): number => Math.ceil(s.length / 4);

/** Embedder signature — injectable so the batching/SQL logic is unit-testable with a
 *  MOCK (no live Voyage quota in tests). Returns one vector per input text, in order. */
export type ContextualEmbedder = (
  texts: string[],
  inputType: "document" | "query",
) => Promise<number[][]>;

export interface BackfillDeps {
  pool: Pool;
  embed: ContextualEmbedder;
  batchSize?: number;
  /** Optional progress sink (defaults to console). */
  log?: (msg: string) => void;
}

export interface BackfillResult {
  processed: number;
  batches: number;
}

interface PendingRow {
  id: string;
  title: string;
  content: string;
}

/** How many active rows still lack embedding_v2 (the AC-021 completeness probe). */
export async function countPending(pool: Pool): Promise<number> {
  const { rows } = await pool.query<{ n: string }>(
    `SELECT count(*) AS n FROM memory.memories
     WHERE archived_at IS NULL AND COALESCE(status,'active') = 'active'
       AND embedding_v2 IS NULL`,
  );
  return Number(rows[0].n);
}

/**
 * Backfill embedding_v2 for all active rows that lack it, one batch at a time.
 * RESUMABLE: each batch re-SELECTs pending rows (ORDER BY id) and LIMITs to batchSize,
 * so a crash mid-run loses at most the in-flight batch. Returns counts for the caller
 * to log/assert.
 */
export async function backfillContext(
  deps: BackfillDeps,
): Promise<BackfillResult> {
  const { pool, embed } = deps;
  const batchSize = deps.batchSize ?? BACKFILL_BATCH;
  const log = deps.log ?? ((m: string) => console.error(m));

  let processed = 0;
  let batches = 0;

  for (;;) {
    const { rows } = await pool.query<PendingRow>(
      `SELECT id, title, content FROM memory.memories
       WHERE archived_at IS NULL AND COALESCE(status,'active') = 'active'
         AND embedding_v2 IS NULL
       ORDER BY id
       LIMIT $1`,
      [batchSize],
    );
    if (rows.length === 0) break;

    // One doc = one chunk: each memory's title+content is a single unit, truncated to
    // stay under the 32k-token/doc cap. Split the page into token-budgeted SUB-batches so
    // no single Voyage request exceeds the 120k-token / 1000-doc caps.
    const docs = rows.map((r) => ({
      id: r.id,
      text: `${r.title}\n${r.content}`.slice(0, MAX_DOC_CHARS),
    }));

    let i = 0;
    while (i < docs.length) {
      const sub: typeof docs = [];
      let toks = 0;
      while (i < docs.length && sub.length < DOC_CAP) {
        const t = estTokens(docs[i].text);
        if (sub.length > 0 && toks + t > TOKEN_BUDGET) break;
        sub.push(docs[i]);
        toks += t;
        i++;
      }

      const vectors = await embed(
        sub.map((d) => d.text),
        "document",
      );
      if (vectors.length !== sub.length) {
        throw new Error(
          `embedder returned ${vectors.length} vectors for ${sub.length} rows`,
        );
      }
      for (let j = 0; j < sub.length; j++) {
        await pool.query(
          `UPDATE memory.memories SET embedding_v2 = $1::halfvec WHERE id = $2`,
          [toVectorLiteral(vectors[j]), sub[j].id],
        );
      }

      processed += sub.length;
      batches += 1;
      log(
        `backfill: batch ${batches} (+${sub.length}, ~${toks} tok) → ${processed} embedded`,
      );
    }
  }

  return { processed, batches };
}

// ── CLI entrypoint (npm run backfill:context) ───────────────────────────────────────
// Skipped when imported (tests import backfillContext/countPending directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  (async () => {
    const before = await countPending(defaultPool);
    console.error(
      `backfill:context — ${before} active rows missing embedding_v2`,
    );
    const { processed, batches } = await backfillContext({
      pool: defaultPool,
      embed: embedContextualSingle,
    });
    const after = await countPending(defaultPool);
    console.error(
      `backfill:context done — ${processed} embedded in ${batches} batch(es); ${after} still pending`,
    );
    await defaultPool.end();
    if (after > 0) process.exit(1); // AC-021: not complete → non-zero exit
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
