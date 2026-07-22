// Graph traversal (Wave 4, Phase 1e). Walks codebase.symbol_edges with a WITH RECURSIVE
// CTE and returns structured rows for code_graph_expand / code_symbol_lookup /
// code_trace_path. Pattern mirrors search.ts: typed result rows + a single pool.query of a
// WITH-CTE SQL string + a formatter, no ORM.
//
// The graph is BY-NAME resolved (Wave 3): a call edge whose target name matched >1 symbol
// in the repo fans out to one symbol_edges row per candidate. Such targets are AMBIGUOUS
// (AC-010) — detected here as a (from_symbol, target name) group with >1 distinct to_symbol
// among call edges — and the tool layer renders "(name-matched, may be ambiguous)".
//
// Bounds (AC-008): depth default 4, HARD cap 10 (clamped server-side BEFORE building SQL —
// never interpolate an unclamped client value), limit default 50. The cycle guard is a
// per-path uuid[] accumulator: a hop into a symbol already on the current path is rejected,
// so the A↔B fixture cycle terminates instead of looping forever.

import type pg from "pg";
import { pool } from "../db/pool.js";

export type Direction = "callees" | "callers" | "both";

export interface GraphRow {
  id: string;
  name: string;
  file: string; // file_path
  line: number; // start_line
  depth: number; // 0 = seed (def)
  edge_type: string; // 'def' | 'call' | 'import' (+ direction for non-seed rows)
  ambiguous: boolean; // true => the hop into this row was a name-matched (>1 candidate) edge
}

export const DEFAULT_DEPTH = 4;
export const MAX_DEPTH = 10;
export const DEFAULT_LIMIT = 50;

// ── Traversal engine routing (Wave 3 graph hardening, AC-401) ────────────────
// Depth ≤ 4 keeps the recursive CTE (with `SET LOCAL work_mem` on a dedicated
// client — the path-array walk is memory-hungry on dense graphs). Depth ≥ 5
// switches to app-side iterative BFS: one frontier query per level, a JS visited
// set, and a per-level cap, so a deep expand can't blow up the CTE working set.
export const BFS_DEPTH_THRESHOLD = 5;
export const BFS_LEVEL_CAP = 5000;
export const CTE_WORK_MEM = "256MB";

/** Router decision (exported for tests): depth ≥ threshold → app-side BFS. */
export function usesBfs(depth: number | undefined): boolean {
  return clampDepth(depth) >= BFS_DEPTH_THRESHOLD;
}

/** Clamp a client-supplied depth into [1, MAX_DEPTH]; default when unset/NaN. */
export function clampDepth(depth: number | undefined): number {
  if (depth === undefined || !Number.isFinite(depth)) return DEFAULT_DEPTH;
  const d = Math.trunc(depth);
  if (d < 1) return 1;
  if (d > MAX_DEPTH) return MAX_DEPTH;
  return d;
}

/** Clamp a client-supplied limit into [1, 1000]; default when unset/NaN. */
export function clampLimit(limit: number | undefined): number {
  if (limit === undefined || !Number.isFinite(limit)) return DEFAULT_LIMIT;
  const l = Math.trunc(limit);
  if (l < 1) return 1;
  if (l > 1000) return 1000;
  return l;
}

export interface SeedInput {
  projectId: string;
  repo?: string;
  symbolId?: string;
  name?: string;
  file?: string;
  line?: number;
  chunkId?: string;
}

/**
 * Resolve a seed to one or more symbol ids (a name may match many; def returns all).
 * Order of precedence: explicit symbolId > name > chunkId > file(+line).
 * Returns [] when nothing resolves (the tool layer then does the file-read fallback).
 */
export async function resolveSeed(seed: SeedInput): Promise<GraphRow[]> {
  if (seed.symbolId) {
    const { rows } = await pool.query<SymbolLite>(
      `SELECT id, name, file_path, start_line FROM codebase.symbols
       WHERE id = $1 ${seed.repo ? "AND repository_id = $2" : ""}`,
      seed.repo ? [seed.symbolId, seed.repo] : [seed.symbolId],
    );
    return rows.map(toDefRow);
  }

  if (seed.name) {
    const params: unknown[] = [seed.projectId, seed.name];
    let filt = "";
    if (seed.repo) {
      params.push(seed.repo);
      filt = `AND repository_id = $${params.length}`;
    }
    const { rows } = await pool.query<SymbolLite>(
      `SELECT id, name, file_path, start_line FROM codebase.symbols
       WHERE project_id = $1 AND name = $2 ${filt}
       ORDER BY file_path, start_line`,
      params,
    );
    return rows.map(toDefRow);
  }

  if (seed.chunkId) {
    const { rows } = await pool.query<{
      file_path: string;
      start_line: number;
      repository_id: string;
    }>(
      `SELECT file_path, start_line, repository_id FROM codebase.code_chunks WHERE id = $1`,
      [seed.chunkId],
    );
    if (rows.length === 0) return [];
    const c = rows[0];
    return enclosingSymbol(
      seed.projectId,
      seed.repo ?? c.repository_id,
      c.file_path,
      c.start_line,
    );
  }

  if (seed.file) {
    return enclosingSymbol(
      seed.projectId,
      seed.repo,
      seed.file,
      seed.line ?? Number.MAX_SAFE_INTEGER,
    );
  }

  return [];
}

interface SymbolLite {
  id: string;
  name: string;
  file_path: string;
  start_line: number;
}

function toDefRow(r: SymbolLite): GraphRow {
  return {
    id: r.id,
    name: r.name,
    file: r.file_path,
    line: r.start_line,
    depth: 0,
    edge_type: "def",
    ambiguous: false,
  };
}

/** The symbol whose declaration most closely precedes `line` in `file` (the enclosing def). */
async function enclosingSymbol(
  projectId: string,
  repo: string | undefined,
  file: string,
  line: number,
): Promise<GraphRow[]> {
  const params: unknown[] = [projectId, file, line];
  let filt = "";
  if (repo) {
    params.push(repo);
    filt = `AND repository_id = $${params.length}`;
  }
  const { rows } = await pool.query<SymbolLite>(
    `SELECT id, name, file_path, start_line FROM codebase.symbols
     WHERE project_id = $1 AND file_path = $2 AND start_line <= $3 ${filt}
     ORDER BY start_line DESC
     LIMIT 1`,
    params,
  );
  // No symbol declared at/above the line (e.g. file-level code) — fall back to the
  // file's first symbol so a file/chunk seed still anchors the graph somewhere useful.
  if (rows.length === 0) {
    const firstParams: unknown[] = [projectId, file];
    let firstFilt = "";
    if (repo) {
      firstParams.push(repo);
      firstFilt = `AND repository_id = $${firstParams.length}`;
    }
    const { rows: first } = await pool.query<SymbolLite>(
      `SELECT id, name, file_path, start_line FROM codebase.symbols
       WHERE project_id = $1 AND file_path = $2 ${firstFilt}
       ORDER BY start_line ASC
       LIMIT 1`,
      firstParams,
    );
    return first.map(toDefRow);
  }
  return rows.map(toDefRow);
}

interface TraverseRow {
  id: string;
  name: string;
  file_path: string;
  start_line: number;
  depth: number;
  edge_type: string;
  ambiguous: boolean;
}

function toGraphRow(r: TraverseRow): GraphRow {
  return {
    id: r.id,
    name: r.name,
    file: r.file_path,
    line: r.start_line,
    depth: r.depth,
    edge_type: r.edge_type,
    ambiguous: r.ambiguous,
  };
}

// Ambiguity detection (AC-010), shared VERBATIM by the CTE and BFS engines so the
// `ambiguous` flag can never diverge between them: a call hop whose target NAME matched
// >1 distinct symbol from the same caller. First find the ambiguous (from_symbol, target
// name) groups, then project each member to_symbol id — those targets were name-matched
// and must be labeled. Embed as trailing CTE clauses (expects a preceding `WITH ... ,`).
const AMBIGUITY_CTES = `amb_groups AS (
      SELECT e.from_symbol, st.name AS to_name
      FROM codebase.symbol_edges e
      JOIN codebase.symbols st ON st.id = e.to_symbol
      WHERE e.kind = 'call'
      GROUP BY e.from_symbol, st.name
      HAVING count(DISTINCT e.to_symbol) > 1
    ),
    amb AS (
      SELECT DISTINCT e.to_symbol AS to_id
      FROM codebase.symbol_edges e
      JOIN codebase.symbols st ON st.id = e.to_symbol
      JOIN amb_groups g
        ON g.from_symbol = e.from_symbol AND g.to_name = st.name
      WHERE e.kind = 'call'
    )`;

/**
 * Run one query on a dedicated pooled client inside a BEGIN/COMMIT with
 * `SET LOCAL work_mem` (AC-401): the recursive path-array CTEs are the only
 * memory-hungry queries in this service, so they get a per-transaction bump
 * instead of a global one. SET LOCAL requires an explicit transaction.
 */
async function queryWithWorkMem<T extends pg.QueryResultRow>(
  sql: string,
  params: unknown[],
): Promise<T[]> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query(`SET LOCAL work_mem = '${CTE_WORK_MEM}'`);
    const { rows } = await client.query<T>(sql, params);
    await client.query("COMMIT");
    return rows;
  } catch (e) {
    try {
      await client.query("ROLLBACK");
    } catch {
      /* the throw below is the real signal */
    }
    throw e;
  } finally {
    client.release();
  }
}

/**
 * Expand the graph from a set of seed symbol ids along call edges. `direction` controls
 * which way call edges are followed; `both` returns the union of callers + callees over
 * the seeds. Both engines emit their own depth-0 `def` rows for the seeds — callers do
 * NOT prepend them.
 *
 * Router (AC-401): depth ≤ 4 → recursive CTE (`expandCte`); depth ≥ 5 → app-side
 * iterative BFS (`expandBfs`). Both engines return the identical GraphRow contract
 * (min-depth per symbol, first-reach edge label, shared ambiguity definition) — the
 * traverse-bfs parity test pins BFS ≡ CTE at overlapping depths.
 */
export async function expand(
  seedIds: string[],
  direction: Direction,
  depth: number,
  limit: number,
): Promise<GraphRow[]> {
  if (seedIds.length === 0) return [];
  return usesBfs(depth)
    ? expandBfs(seedIds, direction, depth, limit)
    : expandCte(seedIds, direction, depth, limit);
}

/**
 * CTE engine (depth ≤ 4 path). Each direction is its OWN recursive CTE carrying a uuid[]
 * path for the cycle guard and a depth counter (UNION ALL + the path-array guard, not
 * UNION, so each acyclic path keeps its own depth); `both` unions the two walks and
 * min-depth-aggregates. One recursive CTE per direction is load-bearing: a single
 * `seed UNION ALL callees-term UNION ALL callers-term` CTE is invalid SQL — Postgres
 * parses it as `(seed UNION ALL callees) UNION ALL callers`, putting a recursive
 * reference inside the non-recursive term ("recursive reference to query \"walk\" must
 * not appear within its non-recursive term").
 */
export async function expandCte(
  seedIds: string[],
  direction: Direction,
  depth: number,
  limit: number,
): Promise<GraphRow[]> {
  if (seedIds.length === 0) return [];
  const maxDepth = clampDepth(depth);
  const lim = clampLimit(limit);

  // Build the recursive walk CTE for one direction. `call` edges only — import edges
  // over-connect (every symbol in an importing file links to the imported one) and would
  // swamp a call trace (AC-402).
  // callees: follow from_symbol -> to_symbol. callers: follow to_symbol -> from_symbol.
  const buildWalk = (dir: "callees" | "callers"): string => {
    const cte = dir === "callees" ? "walk_callees" : "walk_callers";
    const join =
      dir === "callees" ? "e.from_symbol = w.id" : "e.to_symbol = w.id";
    const nextId = dir === "callees" ? "e.to_symbol" : "e.from_symbol";
    const label = dir === "callees" ? "call→" : "call←";
    return `${cte}(id, depth, path, edge_type) AS (
      SELECT s.id, 0, ARRAY[s.id], 'def'::text FROM seeds s
      UNION ALL
      SELECT ${nextId} AS id, w.depth + 1 AS depth, w.path || ${nextId} AS path,
             '${label}'::text AS edge_type
      FROM ${cte} w
      JOIN codebase.symbol_edges e ON ${join}
      WHERE e.kind = 'call'
        AND w.depth < ${maxDepth}
        AND NOT (${nextId} = ANY(w.path))
    )`;
  };

  const walkCtes: string[] = [];
  const walkSelects: string[] = [];
  if (direction === "callees" || direction === "both") {
    walkCtes.push(buildWalk("callees"));
    walkSelects.push("SELECT id, depth, edge_type FROM walk_callees");
  }
  if (direction === "callers" || direction === "both") {
    walkCtes.push(buildWalk("callers"));
    walkSelects.push("SELECT id, depth, edge_type FROM walk_callers");
  }

  const sql = `
    WITH RECURSIVE seeds(id) AS (
      SELECT unnest($1::uuid[])
    ),
    ${walkCtes.join(",\n    ")},
    walk(id, depth, edge_type) AS (
      ${walkSelects.join("\n      UNION ALL\n      ")}
    ),
    -- shallowest reach per symbol (a symbol reachable by several paths shows its min depth).
    -- The label tie-break at equal depth (edge_type = 'call←' sorts callees-first) makes
    -- a both-direction tie deterministic AND mirrors BFS, which expands callees before
    -- callers per level — without it the [1] pick at equal depth is unspecified.
    reached AS (
      SELECT id, min(depth) AS depth,
             (array_agg(edge_type ORDER BY depth, edge_type = 'call←'))[1] AS edge_type
      FROM walk
      GROUP BY id
    ),
    ${AMBIGUITY_CTES}
    SELECT s.id, s.name, s.file_path, s.start_line, r.depth, r.edge_type,
           EXISTS (SELECT 1 FROM amb WHERE amb.to_id = s.id AND r.depth > 0) AS ambiguous
    FROM reached r
    JOIN codebase.symbols s ON s.id = r.id
    ORDER BY r.depth, s.file_path, s.start_line
    LIMIT ${lim}`;

  const rows = await queryWithWorkMem<TraverseRow>(sql, [seedIds]);
  return rows.map(toGraphRow);
}

/**
 * One directional BFS walk: frontier query per level (`= ANY($ids)`, `kind = 'call'` is
 * the load-bearing filter — import edges are excluded from call expansion in BOTH
 * engines, AC-402), JS visited map recording FIRST reach — which IS the CTE walk's
 * `min(depth)`, since BFS discovers min depths level by level. `levelCap` truncates each
 * frontier query so one hyper-connected level can't materialize an unbounded id set.
 * Returns id → min depth (seeds at 0).
 */
async function bfsWalk(
  seedIds: string[],
  dir: "callees" | "callers",
  maxDepth: number,
  levelCap: number,
): Promise<Map<string, number>> {
  const sql =
    dir === "callees"
      ? `SELECT DISTINCT to_symbol AS next FROM codebase.symbol_edges
         WHERE kind = 'call' AND from_symbol = ANY($1::uuid[])
         LIMIT ${levelCap}`
      : `SELECT DISTINCT from_symbol AS next FROM codebase.symbol_edges
         WHERE kind = 'call' AND to_symbol = ANY($1::uuid[])
         LIMIT ${levelCap}`;
  const depthOf = new Map<string, number>();
  for (const id of seedIds) depthOf.set(id, 0);
  let frontier = [...depthOf.keys()];
  for (let d = 1; d <= maxDepth && frontier.length > 0; d++) {
    const { rows } = await pool.query<{ next: string }>(sql, [frontier]);
    const next: string[] = [];
    for (const r of rows) {
      if (depthOf.has(r.next)) continue; // visited: first reach keeps the min depth
      depthOf.set(r.next, d);
      next.push(r.next);
    }
    frontier = next;
  }
  return depthOf;
}

/**
 * BFS engine (depth ≥ 5 path, AC-401): iterative frontier expansion in JS instead of a
 * deep recursive CTE. Mirrors `expandCte` exactly: one independent walk per direction,
 * `both` = the union of the two walks merged on min depth. A node on both walks at EQUAL
 * depth keeps the callees label — the same tie-break the CTE's
 * `(array_agg(edge_type ORDER BY depth, edge_type = 'call←'))[1]` applies.
 *
 * `levelCap` (test-injectable; defaults to BFS_LEVEL_CAP) bounds each frontier query.
 */
export async function expandBfs(
  seedIds: string[],
  direction: Direction,
  depth: number,
  limit: number,
  levelCap: number = BFS_LEVEL_CAP,
): Promise<GraphRow[]> {
  if (seedIds.length === 0) return [];
  const maxDepth = clampDepth(depth);
  const lim = clampLimit(limit);

  const dirs: Array<"callees" | "callers"> = [];
  if (direction === "callees" || direction === "both") dirs.push("callees");
  if (direction === "callers" || direction === "both") dirs.push("callers");

  // Seeds are depth-0 defs, exactly like the CTE's seed rows. Walks are merged
  // callees-first, and only a STRICTLY shallower reach overwrites — so an equal-depth
  // tie keeps the callees label, matching the CTE tie-break.
  const reached = new Map<string, { depth: number; edgeType: string }>();
  for (const id of seedIds) reached.set(id, { depth: 0, edgeType: "def" });
  for (const dir of dirs) {
    const label = dir === "callees" ? "call→" : "call←";
    const walk = await bfsWalk(seedIds, dir, maxDepth, levelCap);
    for (const [id, d] of walk) {
      if (d === 0) continue; // seed rows already recorded as defs
      const prev = reached.get(id);
      if (!prev || d < prev.depth)
        reached.set(id, { depth: d, edgeType: label });
    }
  }

  // Final materialization: one query joining symbols on the reached id set, with the
  // SAME ambiguity definition and ordering as the CTE engine.
  const ids = [...reached.keys()];
  const depths: number[] = [];
  const labels: string[] = [];
  for (const id of ids) {
    const r = reached.get(id)!;
    depths.push(r.depth);
    labels.push(r.edgeType);
  }
  const sql = `
    WITH ids(id, depth, edge_type) AS (
      SELECT * FROM unnest($1::uuid[], $2::int[], $3::text[])
    ),
    ${AMBIGUITY_CTES}
    SELECT s.id, s.name, s.file_path, s.start_line, i.depth, i.edge_type,
           EXISTS (SELECT 1 FROM amb WHERE amb.to_id = s.id AND i.depth > 0) AS ambiguous
    FROM ids i
    JOIN codebase.symbols s ON s.id = i.id
    ORDER BY i.depth, s.file_path, s.start_line
    LIMIT ${lim}`;
  const { rows } = await pool.query<TraverseRow>(sql, [ids, depths, labels]);
  return rows.map(toGraphRow);
}

/**
 * code_graph_expand core: resolve a seed, then return def + callers + callees (or one
 * direction). Returns [] when the seed resolves to no symbol (the tool then file-reads).
 */
export async function graphExpand(input: {
  seed: SeedInput;
  direction?: Direction;
  depth?: number;
  limit?: number;
}): Promise<GraphRow[]> {
  const defRows = await resolveSeed(input.seed);
  if (defRows.length === 0) return [];
  const seedIds = defRows.map((d) => d.id);
  return expand(
    seedIds,
    input.direction ?? "both",
    clampDepth(input.depth),
    clampLimit(input.limit),
  );
}

/**
 * code_trace_path core: a directed callees-only chain from a seed.
 *
 * When `toName` is absent, this returns the depth-ordered callees expansion (the same set
 * `code_graph_expand` would produce in the callees direction) — there is no target, so a
 * concrete predecessor chain is undefined.
 *
 * When `toName` is given, this returns an ACTUAL path — the ordered node chain from the seed
 * to the FIRST reached symbol named `toName` — NOT a sliced depth-ordered set. It walks call
 * edges with a recursive CTE that carries the concrete `uuid[]` path it took (plus the same
 * per-path cycle guard as `expand`), then picks the shortest (then lexically-first) acyclic
 * path whose terminal node is named `toName` and reconstructs the ordered symbol rows from
 * that path array. By-name edges make "shortest" non-unique, so this is A real path, not a
 * proven-unique one — but every consecutive pair is a genuine call edge.
 */
export async function tracePath(input: {
  seed: SeedInput;
  toName?: string;
  depth?: number;
}): Promise<{ rows: GraphRow[]; reachedTo: boolean }> {
  const defRows = await resolveSeed(input.seed);
  if (defRows.length === 0) return { rows: [], reachedTo: false };
  const seedIds = defRows.map((d) => d.id);
  const maxDepth = clampDepth(input.depth);

  // No target: keep the callees-expansion behavior (a path to nowhere is undefined).
  if (!input.toName) {
    const rows = await expand(seedIds, "callees", maxDepth, DEFAULT_LIMIT);
    return { rows, reachedTo: false };
  }

  // Recursive CTE over call edges. `path` is the ordered uuid[] of the route taken; the
  // `NOT (e.to_symbol = ANY(w.path))` guard makes each path acyclic and terminates the
  // A↔B cycle. We keep ONLY paths whose terminal symbol is named $2, then order by length
  // (shortest first) and break ties on the path array so the result is deterministic.
  const sql = `
    WITH RECURSIVE walk(id, depth, path) AS (
      SELECT unnest($1::uuid[]), 0, ARRAY[]::uuid[]
      UNION ALL
      SELECT e.to_symbol, w.depth + 1, w.path || w.id
      FROM walk w
      JOIN codebase.symbol_edges e ON e.from_symbol = w.id
      WHERE e.kind = 'call'
        AND w.depth < ${maxDepth}
        AND NOT (e.to_symbol = ANY(w.path))
        AND NOT (e.to_symbol = w.id)
    ),
    reached AS (
      SELECT (w.path || w.id) AS full_path, array_length(w.path || w.id, 1) AS len
      FROM walk w
      JOIN codebase.symbols s ON s.id = w.id
      WHERE s.name = $2
    )
    SELECT full_path FROM reached
    ORDER BY len ASC, full_path ASC
    LIMIT 1`;

  // This targeted CTE stays a CTE at every depth (it needs the concrete uuid[] path for
  // chain reconstruction — BFS-with-parent-pointers is out of scope), so it gets the same
  // dedicated-client + SET LOCAL work_mem treatment as the ≤4-hop expand path (AC-401).
  // The no-target and unreachable branches call expand() and inherit BFS routing at
  // depth ≥ 5 automatically.
  const pathRows = await queryWithWorkMem<{ full_path: string[] }>(sql, [
    seedIds,
    input.toName,
  ]);
  if (pathRows.length === 0) {
    // Target unreachable within depth: return the plain callees expansion as context.
    const rows = await expand(seedIds, "callees", maxDepth, DEFAULT_LIMIT);
    return { rows, reachedTo: false };
  }

  // Reconstruct the ordered symbol chain from the concrete path array (preserve order).
  const pathIds = pathRows[0].full_path;
  const { rows: symRows } = await pool.query<SymbolLite>(
    `SELECT id, name, file_path, start_line FROM codebase.symbols
     WHERE id = ANY($1::uuid[])`,
    [pathIds],
  );
  const byId = new Map(symRows.map((s) => [s.id, s]));
  const chain: GraphRow[] = pathIds
    .map((id, depth): GraphRow | null => {
      const s = byId.get(id);
      if (!s) return null;
      return {
        id: s.id,
        name: s.name,
        file: s.file_path,
        line: s.start_line,
        depth,
        edge_type: depth === 0 ? "def" : "call→",
        ambiguous: false,
      };
    })
    .filter((r): r is GraphRow => r !== null);
  return { rows: chain, reachedTo: true };
}

/** code_symbol_lookup core: resolve a name to its definition(s), repo+project scoped. */
export async function lookupSymbol(input: {
  projectId: string;
  name: string;
  repo?: string;
  limit?: number;
}): Promise<{ name: string; kind: string; file: string; line: number }[]> {
  const params: unknown[] = [input.projectId, input.name];
  let filt = "";
  if (input.repo) {
    params.push(input.repo);
    filt = `AND repository_id = $${params.length}`;
  }
  const lim = clampLimit(input.limit);
  const { rows } = await pool.query<{
    name: string;
    kind: string;
    file_path: string;
    start_line: number;
  }>(
    `SELECT name, kind, file_path, start_line FROM codebase.symbols
     WHERE project_id = $1 AND name = $2 ${filt}
     ORDER BY file_path, start_line
     LIMIT ${lim}`,
    params,
  );
  return rows.map((r) => ({
    name: r.name,
    kind: r.kind,
    file: r.file_path,
    line: r.start_line,
  }));
}

/** Render graph rows as compact text for the MCP tool response. */
export function formatGraphRows(rows: GraphRow[]): string {
  if (rows.length === 0) return "No graph rows.";
  return rows
    .map((r) => {
      const amb = r.ambiguous ? " (name-matched, may be ambiguous)" : "";
      const indent = "  ".repeat(r.depth);
      return `${indent}[d${r.depth} ${r.edge_type}] ${r.name}  ${r.file}:${r.line}${amb}  [${r.id}]`;
    })
    .join("\n");
}
