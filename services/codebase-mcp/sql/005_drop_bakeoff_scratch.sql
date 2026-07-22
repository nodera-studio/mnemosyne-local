-- HOLD — Migration 005 (Wave 5, post-decision): drop the bake-off scratch column.
-- (The leading `-- HOLD` marker makes the batch migrate runner SKIP this file.)
--
-- ⚠️ DO NOT APPLY YET. This is intentionally "ready but not auto-applied."
-- Apply ONLY after the bake-off decision is recorded in test/bakeoff.md AND, if
-- voyage-context-3 won, after the operator has:
--   1. flipped VOYAGE_CODE_MODEL to the winner (compose.yaml / .env),
--   2. re-embedded the LIVE `embedding` column via a forced full reindex,
--   3. spot-checked code_search quality on the new embedder.
-- Until then, embedding_ctx is the recorded measurement artifact — dropping it discards
-- the bake-off data. If voyage-code-3 won (kept incumbent), the scratch column has served
-- its purpose and can be dropped once the negative result is recorded.
--
-- To apply (deliberately, post-decision):
--   psql "$DATABASE_URL" -f sql/005_drop_bakeoff_scratch.sql
-- (or rename so `npm run migrate` picks it up once the hold is lifted.)

DROP INDEX IF EXISTS codebase.chunk_ctx_hnsw;
ALTER TABLE codebase.code_chunks DROP COLUMN IF EXISTS embedding_ctx;
