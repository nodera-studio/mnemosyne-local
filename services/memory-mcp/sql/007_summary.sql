-- Migration 007: stored dense summary for retrieval token-efficiency. Idempotent.
--
-- Naming precedent: memory.entities.summary (001). NULL = not yet summarized.
-- Populated by (a) storeMemory when the operator enables summarize-on-store
-- (ANTHROPIC_API_KEY set AND SUMMARIZE_ON_STORE=1 — src/summarize.ts) and
-- (b) the PAID resumable backfill `npm run backfill:summaries -- --yes`
-- (src/db/backfill-summaries.ts), which also covers importer raw-INSERT rows.
-- No index: the only filtered scan is the backfill's `summary IS NULL` paging,
-- fine as a seq scan at this corpus size.

ALTER TABLE memory.memories
  ADD COLUMN IF NOT EXISTS summary text;
