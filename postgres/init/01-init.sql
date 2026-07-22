-- Runs once, on first cluster initialization.
-- Schema DDL for memory.* and codebase.* is applied later via Drizzle migrations.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS memory;
CREATE SCHEMA IF NOT EXISTS codebase;

DO $$
BEGIN
  RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END $$;
