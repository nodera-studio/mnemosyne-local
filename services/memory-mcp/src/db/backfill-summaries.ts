// PAID, RESUMABLE operator backfill (wave-2, AC-811): populate memory.memories.summary
// for existing rows — including importer raw-INSERT rows that bypass storeMemory. One
// LLM call per row (model: CONSOLIDATE_MODEL, default claude-haiku-4-5), so this spends
// Anthropic quota; cost ≈ pending-rows × ~(min(content chars, 8000)/4 + 256) tokens.
// It is NEVER run on container start. Gate: `npm run backfill:summaries -- --yes`.
//
// Resumability: KEYSET paging over `summary IS NULL` rows (differs from
// backfill-context's re-SELECT paging DELIBERATELY): a row whose summarizer returns
// null (persistent failure) is counted `skipped` and the cursor still advances, so the
// run always terminates — re-SELECT-on-NULL paging would loop forever on a failing
// row. Re-runs restart at cursor 0 and naturally skip rows already summarized.
//
// No Voyage calls here at all: summaries do not change embeddings (search_tsv is
// generated from title+content only), so no re-embed and no tsv refresh is needed.
//
// AC-108: main-guarded, `npm run`-gated, never imported by server.ts, never in CI.
// Zero-network unit tests inject the summarizer (test/backfill-summaries.test.ts).
//
// FLIP GATE (AC-812): this backfill is the moment reranker docs change for summarized
// rows — run the live recall-gate re-record per test/retune.md's runbook afterwards.

import type { Pool } from "pg";
import { pool as defaultPool } from "./pool.js";
import { config } from "../config.js";
import { summarizeMemory } from "../summarize.js";

/** DB page size — one LLM call per row, so pages stay small. */
export const SUMMARY_BACKFILL_BATCH = 50;

export const COST_NOTE =
  "backfill:summaries is a PAID operator script: it sends every active row where summary IS NULL " +
  `to the Anthropic API (model: ${config.consolidateModel}) — cost ≈ pending rows × ` +
  "~(min(content chars, 8000)/4 + 256) tokens per row. Nothing was run. " +
  "Re-run with an explicit consent flag:  npm run backfill:summaries -- --yes";

/** Returns the refusal message when the paid-consent flag is absent, else null. */
export function guardPaidRun(argv: string[]): string | null {
  return argv.includes("--yes") ? null : COST_NOTE;
}

/** How many active rows still lack a summary (the AC-811 completeness probe). */
export async function countPendingSummaries(pool: Pool): Promise<number> {
  const { rows } = await pool.query<{ n: string }>(
    `SELECT count(*) AS n FROM memory.memories
     WHERE archived_at IS NULL AND COALESCE(status,'active') = 'active'
       AND summary IS NULL`,
  );
  return Number(rows[0].n);
}

/** Row summarizer — injectable so tests run with a mock (zero LLM calls). Returns
 *  null on failure; the row is skipped, never aborts the run. */
export type RowSummarizer = (
  title: string,
  content: string,
) => Promise<string | null>;

export interface SummaryBackfillDeps {
  pool: Pool;
  summarize: RowSummarizer;
  batchSize?: number;
  /** Optional progress sink (defaults to console). */
  log?: (msg: string) => void;
}

export interface SummaryBackfillResult {
  processed: number;
  skipped: number;
  batches: number;
}

interface PendingRow {
  id: string;
  title: string;
  content: string;
}

/** Batch-appropriate summarizer timeout for the offline CLI. The 4 s
 *  SUMMARY_TIMEOUT_MS bound protects interactive memory_store latency — a paid
 *  backfill row should wait out a slow completion, not bill-and-skip it. */
export const BACKFILL_SUMMARY_TIMEOUT_MS = 30_000;

/**
 * Backfill `summary` for all active rows that lack it, one keyset page at a time.
 * A null summarizer result counts as `skipped` and the cursor still advances
 * (AC-811: per-row failure skips, never aborts; the run always terminates).
 */
export async function backfillSummaries(
  deps: SummaryBackfillDeps,
): Promise<SummaryBackfillResult> {
  const { pool, summarize } = deps;
  const batchSize = deps.batchSize ?? SUMMARY_BACKFILL_BATCH;
  const log = deps.log ?? ((m: string) => console.error(m));

  let processed = 0;
  let skipped = 0;
  let batches = 0;
  // NULL-seeded cursor: the FIRST page is unbounded. A strict `id > floor` with an
  // all-zero-uuid floor would never visit a row bearing exactly that id, while
  // countPendingSummaries still counts it — a permanently nonzero exit.
  let cursor: string | null = null;

  for (;;) {
    // Explicit tuple type: loop-narrowed `cursor` otherwise feeds the query's
    // inferred values generic, which circles back through the `rows` assignment
    // below (TS7022).
    const params: [string | null, number] = [cursor, batchSize];
    const { rows } = await pool.query<PendingRow>(
      `SELECT id, title, content FROM memory.memories
       WHERE ($1::uuid IS NULL OR id > $1::uuid) AND summary IS NULL
         AND archived_at IS NULL AND COALESCE(status,'active') = 'active'
       ORDER BY id
       LIMIT $2`,
      params,
    );
    if (rows.length === 0) break;

    for (const row of rows) {
      // The RowSummarizer contract is null-on-failure (summarizeMemory never
      // throws), but a throwing injected summarizer must still only skip the row.
      const summary = await summarize(row.title, row.content).catch(() => null);
      if (summary !== null) {
        await pool.query(
          `UPDATE memory.memories SET summary = $1 WHERE id = $2`,
          [summary, row.id],
        );
        processed += 1;
      } else {
        skipped += 1;
      }
    }

    cursor = rows[rows.length - 1].id;
    batches += 1;
    log(
      `backfill:summaries — batch ${batches} (+${rows.length} rows) → ${processed} summarized, ${skipped} skipped`,
    );
  }

  return { processed, skipped, batches };
}

// ── CLI entrypoint (npm run backfill:summaries -- --yes) ─────────────────────────────
// Skipped when imported (tests import the exported functions directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const refusal = guardPaidRun(process.argv.slice(2));
  if (refusal) {
    console.error(refusal);
    process.exit(1);
  }
  if (!config.anthropicApiKey) {
    // Fail fast with the same guidance llmFetch gives — without this, every row
    // would be silently skipped (summarizeMemory returns null when the key is absent).
    console.error(
      "ANTHROPIC_API_KEY is not set — backfill:summaries needs it. Export it " +
        "(optionally with CONSOLIDATE_MODEL) and re-run.",
    );
    process.exit(1);
  }
  (async () => {
    const before = await countPendingSummaries(defaultPool);
    console.error(
      `backfill:summaries — ${before} active rows pending; est ~(min(content,8000)/4 + 256) tokens/row on ${config.consolidateModel}`,
    );
    const { processed, skipped, batches } = await backfillSummaries({
      pool: defaultPool,
      // CLI consent OVERRIDES the SUMMARIZE_ON_STORE write-path gate (the operator
      // said --yes); the key requirement stays (checked above).
      summarize: (t, c) =>
        summarizeMemory(t, c, {
          enabled: config.anthropicApiKey !== "",
          timeoutMs: BACKFILL_SUMMARY_TIMEOUT_MS,
        }),
    });
    const after = await countPendingSummaries(defaultPool);
    console.error(
      `backfill:summaries done — ${processed} summarized, ${skipped} skipped in ${batches} batch(es); ${after} still pending`,
    );
    await defaultPool.end();
    if (after > 0) process.exit(1); // AC-811: not complete → non-zero exit
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
