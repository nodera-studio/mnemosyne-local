-- Memory engine schema (raw SQL — owns halfvec + generated tsvector, which
-- drizzle-kit mis-quotes). Idempotent; safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS memory;

DO $$ BEGIN
  CREATE TYPE memory.memory_type AS ENUM ('episodic','semantic','procedural','entity');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ── memories ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory.memories (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id        text NOT NULL,
  type              memory.memory_type NOT NULL,
  title             text NOT NULL,
  content           text NOT NULL,
  importance        real NOT NULL DEFAULT 0.5,
  access_count      integer NOT NULL DEFAULT 0,
  source_session_id text,
  metadata          jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding         halfvec(1024),
  pinned            boolean NOT NULL DEFAULT false,
  created_at        timestamptz NOT NULL DEFAULT now(),
  last_accessed_at  timestamptz NOT NULL DEFAULT now(),
  archived_at       timestamptz,
  search_tsv        tsvector GENERATED ALWAYS AS (
                      to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))
                    ) STORED
);

CREATE INDEX IF NOT EXISTS mem_hnsw ON memory.memories
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS mem_fts       ON memory.memories USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS mem_proj_type ON memory.memories (project_id, type);
CREATE INDEX IF NOT EXISTS mem_active    ON memory.memories (project_id) WHERE archived_at IS NULL;

-- ── entities ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory.entities (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id  text NOT NULL,
  kind        text NOT NULL,
  external_id text,
  name        text NOT NULL,
  summary     text,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding   halfvec(1024),
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ent_hnsw ON memory.entities
  USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS ent_proj_kind ON memory.entities (project_id, kind);

-- ── entity edges (lightweight graph) ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory.entity_edges (
  id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  from_id   uuid NOT NULL REFERENCES memory.entities(id) ON DELETE CASCADE,
  to_id     uuid NOT NULL REFERENCES memory.entities(id) ON DELETE CASCADE,
  relation  text NOT NULL
);
CREATE INDEX IF NOT EXISTS edge_from ON memory.entity_edges (from_id);
CREATE INDEX IF NOT EXISTS edge_to   ON memory.entity_edges (to_id);
