-- Migration 003 (Wave P): blue/green contextual-embedding column. Idempotent.
--
-- Adds the parallel embedding_v2 column WITHOUT disturbing the live `embedding`
-- column (memory is DURABLE — not regenerated on a schedule). The contextual
-- backfill (src/db/backfill-context.ts) populates it; searchMemory flips to it at the
-- gated redeploy (Step 7), and the legacy `embedding` is dropped only after burn-in
-- (sql/004_drop_legacy_embedding.sql — held).
--
-- NOTE: the matching HNSW index (mem_hnsw_v2) is NOT created here. CREATE INDEX
-- CONCURRENTLY cannot run inside a transaction or a multi-statement simple query, and
-- the batch migrate runner (src/db/migrate.ts) sends each file as one pool.query().
-- The concurrent index is built by the standalone op script src/db/create-v2-index.ts,
-- which opens its own connection and runs the single statement. See that file.

ALTER TABLE memory.memories
  ADD COLUMN IF NOT EXISTS embedding_v2 halfvec(1024);
