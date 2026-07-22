-- Index-run lifecycle observability for the detached indexer.
-- The indexer is spawned with stdio:'ignore', so console output is lost — this
-- table is the ONLY failure/progress signal. Raw SQL, idempotent.

-- phase values: scanning | embedding | graph | resolving | done | error
-- (kept as text, not an enum, for forward-compat with later graph phases).
CREATE TABLE IF NOT EXISTS codebase.index_runs (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      text NOT NULL,
  repository_id   text NOT NULL,
  phase           text NOT NULL,
  files_total     integer,
  files_done      integer,
  chunks_total    integer,
  symbols_total   integer,
  edges_total     integer,
  current_file    text,
  error           text,
  cancel_requested boolean NOT NULL DEFAULT false,
  started_at      timestamptz NOT NULL DEFAULT now(),
  finished_at     timestamptz
);
CREATE INDEX IF NOT EXISTS idx_runs_repo
  ON codebase.index_runs (project_id, repository_id, started_at DESC);
