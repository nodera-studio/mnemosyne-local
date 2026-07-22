-- HOLD — Migration 004 (Wave P, Step 8): drop the legacy embedding column + its index.
-- (The leading `-- HOLD` marker makes the batch migrate runner SKIP this file.)
--
-- ⚠️ DO NOT APPLY YET. This is intentionally "ready but not auto-applied."
-- Apply ONLY after the burn-in window: searchMemory has served queries from
-- embedding_v2 with no recall regression, store/update have been writing BOTH columns,
-- and you are confident no rollback to `embedding` is needed. Until then the legacy
-- column is the rollback path — dropping it is a one-way step.
--
-- Before applying, also remove the legacy embed() write from memory.ts (storeMemory /
-- updateMemory stop writing `embedding`); otherwise inserts will fail on the missing
-- column.
--
-- NUMBERING COORDINATION (Wave 6): the decision-log wave (Wave 6) also adds a
-- memory-mcp migration. If Wave 6 lands its migration as 003/004 first, RENUMBER this
-- drop to the next free number. This drop is the LAST memory migration applied (it is
-- gated on burn-in), so it should always carry the highest number at integration time.
-- Confirm the final sequence in Wave 7 reconciliation.
--
-- To apply (deliberately, post burn-in):
--   psql "$DATABASE_URL" -f sql/004_drop_legacy_embedding.sql
-- (or rename so `npm run migrate` picks it up once the hold is lifted.)

DROP INDEX IF EXISTS memory.mem_hnsw;
ALTER TABLE memory.memories DROP COLUMN IF EXISTS embedding;
