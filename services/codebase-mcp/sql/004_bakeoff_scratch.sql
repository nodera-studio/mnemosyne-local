-- Migration 004 (Wave 5): bake-off scratch embedding column. Idempotent + additive.
--
-- Adds a PARALLEL embedding_ctx column WITHOUT disturbing the live `embedding` column
-- (voyage-code-3). The running code_search reads `embedding`; the bake-off harness
-- (test/bakeoff.md, src/db/bakeoff-embed.ts) writes the voyage-context-3 arm into
-- embedding_ctx so BOTH embedders can be scored with the FULL pipeline (RRF + rerank)
-- on the same corpus, at Matryoshka 1024 + int8 — apples-to-apples (AC-030).
--
-- This is a SCRATCH column: once the bake-off picks a winner (AC-031) and the operator
-- flips VOYAGE_CODE_MODEL + re-embeds the live `embedding`, it is dropped by the HELD
-- migration sql/005_drop_bakeoff_scratch.sql. The live `embedding` column keeps serving
-- code_search the whole time, so search never breaks during the measurement.
--
-- NOTE: no HNSW index is created on embedding_ctx here. The bake-off scores a bounded
-- candidate pool (RECALL_LIMIT rows) per query, so an exact `<=>` scan on the scratch
-- column is fine for an offline measurement; building a second HNSW on a transient column
-- would burn build time for no benefit. If the corpus is large enough that the scratch
-- scan is slow, build one by hand (CREATE INDEX CONCURRENTLY, outside this batch runner).

ALTER TABLE codebase.code_chunks
  ADD COLUMN IF NOT EXISTS embedding_ctx halfvec(1024);
