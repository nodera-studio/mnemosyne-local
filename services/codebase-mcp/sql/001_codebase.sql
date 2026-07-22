-- Codebase engine schema (raw SQL — owns halfvec + generated tsvector).
-- Embeddings: voyage-code-3 @ 1024 dims (halfvec). Idempotent.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS codebase;

-- ── files (Merkle leaves for incremental sync) ──────────────────────────────
CREATE TABLE IF NOT EXISTS codebase.files (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  repository_id  text NOT NULL,
  project_id     text NOT NULL,
  path           text NOT NULL,
  language       text,
  content_sha256 text NOT NULL,
  git_commit_sha text,
  indexed_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (repository_id, path)
);
CREATE INDEX IF NOT EXISTS file_repo_path ON codebase.files (repository_id, path);
CREATE INDEX IF NOT EXISTS file_hash      ON codebase.files (content_sha256);

-- ── code chunks (AST-aware later; pragmatic windows for v1) ─────────────────
CREATE TABLE IF NOT EXISTS codebase.code_chunks (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  file_id        uuid NOT NULL REFERENCES codebase.files(id) ON DELETE CASCADE,
  repository_id  text NOT NULL,
  project_id     text NOT NULL,
  file_path      text NOT NULL,
  language       text,
  symbol_name    text,
  symbol_kind    text,
  start_line     integer NOT NULL,
  end_line       integer NOT NULL,
  content        text NOT NULL,
  content_sha256 text NOT NULL,
  imports        jsonb,
  embedding      halfvec(1024),
  search_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS chunk_hnsw ON codebase.code_chunks
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS chunk_fts    ON codebase.code_chunks USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS chunk_repo   ON codebase.code_chunks (repository_id);
CREATE INDEX IF NOT EXISTS chunk_symbol ON codebase.code_chunks (repository_id, symbol_name);
CREATE INDEX IF NOT EXISTS chunk_hash   ON codebase.code_chunks (content_sha256);

-- ── symbol graph (populated in the tree-sitter/PageRank phase) ──────────────
CREATE TABLE IF NOT EXISTS codebase.symbols (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  repository_id text NOT NULL,
  name          text NOT NULL,
  kind          text NOT NULL,
  file_path     text NOT NULL,
  start_line    integer NOT NULL,
  pagerank      real NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sym_repo_name ON codebase.symbols (repository_id, name);
CREATE INDEX IF NOT EXISTS sym_rank      ON codebase.symbols (repository_id, pagerank);

CREATE TABLE IF NOT EXISTS codebase.symbol_edges (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  repository_id text NOT NULL,
  from_symbol   uuid NOT NULL REFERENCES codebase.symbols(id) ON DELETE CASCADE,
  to_symbol     uuid NOT NULL REFERENCES codebase.symbols(id) ON DELETE CASCADE,
  kind          text NOT NULL
);
-- symbol_edges traversal indexes live in 006_edge_covering_indexes.sql (the original
-- single-column symedge_from/symedge_to were removed from this file — the runner
-- re-executes every file per run, and recreating them here only for 006 to re-drop them
-- churned two index builds inside every migrate).
