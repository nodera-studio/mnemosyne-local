-- Graph hardening: give codebase.symbols stable identity for per-file
-- replacement (the schema contract Wave 3's extractor + indexer hook depend on).
-- The symbols / symbol_edges tables ship in 001 but are ZERO-ROW today, so these
-- additive ALTERs are data-migration-free. Raw SQL, idempotent (re-runnable).
--
-- Decisions captured here (do NOT re-derive in later waves):
--   * project_id mirrors codebase.files.project_id / code_chunks.project_id for
--     project-scoped queries. Added NOT NULL directly: the table is empty, so no
--     backfill is needed and an existing row can't violate the constraint.
--   * file_id is the per-file replacement key. ON DELETE CASCADE means deleting a
--     codebase.files row drops its symbols; symbol_edges already cascades off
--     symbols (001:63-64), so a single file delete cleans the whole sub-graph.
--   * symbol_edges keeps from_symbol/to_symbol NOT NULL (001:63-64). Wave 3
--     resolves edges by NAME repo-wide AFTER the per-file loop and inserts ONLY
--     edges that resolved — unresolved by-name edges are LABELED in tool output
--     (Wave 4 / AC-010), never stored as null-target rows. We add nothing here for
--     them. symbol_edges.repository_id (001:62) is sufficient scoping; we do NOT
--     add project_id to symbol_edges (the traversal joins through symbols).
--
-- Cascade note for Wave 3 ordering: because edges are resolved by name repo-wide,
-- an edge may point INTO a symbol in ANOTHER file. A file_id-keyed delete of one
-- file's symbols will cascade-delete edges pointing into those symbols from other
-- files. That is CORRECT — those edges are stale and get re-resolved in the
-- post-walk pass. Wave 3 order: delete file's symbols -> reinsert -> re-resolve
-- edges repo-wide.

-- ── project_id (mirrors files.project_id) ───────────────────────────────────
ALTER TABLE codebase.symbols ADD COLUMN IF NOT EXISTS project_id text;

-- Backfill is moot (zero rows today); enforce NOT NULL once the column exists.
-- Guarded so a re-run after the constraint already holds is a no-op.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'codebase' AND table_name = 'symbols'
      AND column_name = 'project_id' AND is_nullable = 'YES'
  ) THEN
    ALTER TABLE codebase.symbols ALTER COLUMN project_id SET NOT NULL;
  END IF;
END $$;

-- ── file_id FK -> codebase.files(id), per-file replacement key ───────────────
ALTER TABLE codebase.symbols
  ADD COLUMN IF NOT EXISTS file_id uuid REFERENCES codebase.files(id) ON DELETE CASCADE;

-- ── stable identity for upsert/replace per file (AC-005) ────────────────────
-- (repository_id, file_path, name, kind, start_line) uniquely identifies a
-- symbol. Collision risk: two overloaded functions on the SAME line — extremely
-- rare; the second loses to this index (and name-matched edges are already
-- labeled ambiguous). Acceptable for v1.
CREATE UNIQUE INDEX IF NOT EXISTS sym_identity
  ON codebase.symbols (repository_id, file_path, name, kind, start_line);

-- ── per-file delete lookup ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS sym_file ON codebase.symbols (file_id);
