-- Migration 002: provenance, dedup, and recency/supersession support. Idempotent.

ALTER TABLE memory.memories
  ADD COLUMN IF NOT EXISTS status         text NOT NULL DEFAULT 'active',   -- active|superseded|archived|closed
  ADD COLUMN IF NOT EXISTS source_uri     text,                              -- origin file/db path (provenance + delete-tracking)
  ADD COLUMN IF NOT EXISTS source_kind    text,                              -- claude-memory|research|tech-debt|codex-session|docs|...
  ADD COLUMN IF NOT EXISTS content_sha256 text,                              -- dedup key
  ADD COLUMN IF NOT EXISTS event_date     timestamptz,                       -- date parsed from filename/thread ts (recency)
  ADD COLUMN IF NOT EXISTS superseded_by  uuid REFERENCES memory.memories(id);

CREATE INDEX IF NOT EXISTS mem_status   ON memory.memories (project_id, status);
CREATE INDEX IF NOT EXISTS mem_sha      ON memory.memories (content_sha256);
CREATE INDEX IF NOT EXISTS mem_event_dt ON memory.memories (project_id, event_date);
