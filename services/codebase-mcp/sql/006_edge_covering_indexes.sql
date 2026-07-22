-- Covering indexes for symbol_edges traversal (Wave 3, retrieval-improvement program).
-- Both traversal directions get an index-only-scan path keyed on (endpoint, kind) with
-- the opposite endpoint INCLUDEd, so the recursive CTE and the app-side BFS frontier
-- queries (`WHERE kind='call' AND from_symbol = ANY(...)`) never touch the heap.
--
-- NOT built CONCURRENTLY on purpose: migrate.ts runs each file as ONE pool.query
-- (implicit single transaction), where CONCURRENTLY is illegal — and the table is small
-- today, so a plain build is instant. At >1M edges, build replacements by hand instead:
--   CREATE INDEX CONCURRENTLY ... ; DROP INDEX CONCURRENTLY ...
--
-- The original single-column 001 indexes (symedge_from / symedge_to) were strict
-- prefixes of the covering ones and therefore redundant. 001 no longer creates them
-- (removed so re-runs stop rebuilding them just for this file to drop them again); the
-- DROPs below stay for live DBs migrated before that cleanup, and are no-ops elsewhere.

CREATE INDEX IF NOT EXISTS symedge_from_kind
  ON codebase.symbol_edges (from_symbol, kind) INCLUDE (to_symbol);
CREATE INDEX IF NOT EXISTS symedge_to_kind
  ON codebase.symbol_edges (to_symbol, kind) INCLUDE (from_symbol);

DROP INDEX IF EXISTS codebase.symedge_from;
DROP INDEX IF EXISTS codebase.symedge_to;
