-- Migration 006 (Wave 2): memory.search_log — the "logs-forever" half of the eval
-- bootstrap. Idempotent.
--
-- WHAT: an append-only log of every live memory_search — query, filters, the fused
-- candidate-pool ids, the final reranked ids, and both latencies. This is the raw
-- material for the monthly gold-rotation harvest (`npm run harvest-eval` proposes
-- frequent / zero-hit / low-overlap queries as gold candidates from these rows).
--
-- WRITE PATH (AC-107): src/memory.ts `logSearch` — fire-and-forget (`void ... .catch`),
-- never awaited, errors swallowed; a failed insert (or a missing table) must not affect
-- search results or latency.
--
-- RETENTION: unbounded growth is curbed by the operator's monthly cleanup, documented
-- in src/db/harvest-eval.ts:
--   DELETE FROM memory.search_log WHERE created_at < now() - interval '90 days';

CREATE TABLE IF NOT EXISTS memory.search_log (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id text  NOT NULL,
  query      text  NOT NULL,
  -- Filter params in effect for this search ({} when none). Wave-4 extends this shape
  -- (tags/after) via the logSearch helper; keep it schemaless jsonb.
  filters    jsonb NOT NULL DEFAULT '{}',
  pool_ids   uuid[] NOT NULL,
  final_ids  uuid[] NOT NULL,
  pool_ms    int,
  total_ms   int,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS search_log_created ON memory.search_log (created_at);
