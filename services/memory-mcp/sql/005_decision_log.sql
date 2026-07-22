-- Migration 005 (Wave 6): structured decision-log metadata. Idempotent.
--
-- NUMBERING: Wave P already took 003_context_embeddings.sql and 004_drop_legacy_embedding.sql
-- (the latter is a HOLD drop, gated on burn-in). This decision-log migration therefore
-- lands as 005 — AFTER both Wave-P files — so the batch runner applies it in order and the
-- HOLD-skip on 004 does not affect us. (Wave 6's plan text guessed 003/004 under a
-- pre-Wave-P assumption; renumbered to 005 at integration.)
--
-- WHAT: a SPARSE typed shape for decisions on the EXISTING memory.memories table — NOT a
-- new table. These columns only carry meaning for source_kind='decision' rows and stay
-- NULL on every other row (nullable + indexed only via partial/GIN, so non-decision rows
-- pay near-zero cost).
--
-- ── AXIS SPLIT (read before touching these columns) ─────────────────────────────────────
-- memory.memories ALREADY has two lifecycle columns from earlier migrations:
--   • `status`        (002:4)  — the MEMORY lifecycle: active|superseded|archived|closed.
--   • `superseded_by` (002:9)  — the FORWARD pointer (this memory → the one that replaced it).
-- This migration adds a SEPARATE, decision-specific axis so the two never conflate:
--   • `decision_status` — the DECISION lifecycle: active|superseded|deferred (own CHECK).
--   • `supersedes_id`   — the BACKWARD pointer (this decision → the one IT replaced).
-- Canonical chain direction = `supersedes_id` (this decision replaces that one). The
-- write path keeps the existing `superseded_by` in sync as the inverse on supersession
-- (see src/memory.ts) for back-compat, but `supersedes_id` is what the recursive CTE walks.

ALTER TABLE memory.memories
  ADD COLUMN IF NOT EXISTS decision_project text,
  ADD COLUMN IF NOT EXISTS decision_status  text,
  ADD COLUMN IF NOT EXISTS decided_at       timestamptz,
  ADD COLUMN IF NOT EXISTS supersedes_id    uuid,
  ADD COLUMN IF NOT EXISTS decided_in       text,
  ADD COLUMN IF NOT EXISTS related_ids      uuid[];

-- CHECK on decision_status (active|superseded|deferred). Guarded so re-running is safe.
-- NULL passes the CHECK (non-decision rows have NULL decision_status) — keep it nullable.
DO $$ BEGIN
  ALTER TABLE memory.memories
    ADD CONSTRAINT mem_decision_status_chk
    CHECK (decision_status IS NULL OR decision_status IN ('active','superseded','deferred'));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Self-FK for the supersession chain link (supersedes_id → memory.memories.id). Guarded.
DO $$ BEGIN
  ALTER TABLE memory.memories
    ADD CONSTRAINT mem_supersedes_fk
    FOREIGN KEY (supersedes_id) REFERENCES memory.memories(id);
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- Indexes (the partial btree + GIN run fine in the batch runner; only the HNSW needs
-- CONCURRENTLY and so is built by the standalone op script src/db/create-decision-index.ts).
CREATE INDEX IF NOT EXISTS mem_decision
  ON memory.memories (decision_project, decision_status)
  WHERE source_kind = 'decision';
CREATE INDEX IF NOT EXISTS mem_related_gin
  ON memory.memories USING gin (related_ids);
CREATE INDEX IF NOT EXISTS mem_supersedes
  ON memory.memories (supersedes_id);

-- NOTE: the partial HNSW for active decisions is NOT created here. CREATE INDEX
-- CONCURRENTLY cannot run inside a transaction or a multi-statement simple query, and the
-- batch migrate runner (src/db/migrate.ts) sends each file as one pool.query(). It is built
-- by src/db/create-decision-index.ts (own connection, single statement) — same pattern as
-- Wave-P's create-v2-index.ts.
