// FREE operator script (wave-2 Step 8): propose monthly gold-rotation candidates from
// REAL usage. Reads memory.search_log (migration 006, populated fire-and-forget by
// searchMemory), aggregates per query, and surfaces frequent / zero-hit / low-overlap
// queries that are NOT already in the gold file — emitting the same candidates-file
// shape as distill-eval for HUMAN approval (AC-109). Pure SQL aggregation; no LLM,
// no Voyage, no quota.
//
// Flags: --days 30  --min-count 2  --limit 40
//
// Retention (operator's monthly cleanup, run alongside the harvest):
//   DELETE FROM memory.search_log WHERE created_at < now() - interval '90 days';
//
// Gate: `npm run harvest-eval` — main-guarded (import.meta.url), never on import.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pool as defaultPool } from "./pool.js";

export interface HarvestCandidate {
  query: string;
  /** Most-common final ids for this query — gold HINTS, human-verified before merge. */
  suggestedGold: string[];
  /** Why the query surfaced: frequent | zero-hit | low-overlap. */
  archetype: "frequent" | "zero-hit" | "low-overlap";
  provenance: "log-harvest";
  approved: false;
  /** Times seen in the window (extra context for the reviewer). */
  count: number;
}

interface AggRow {
  query: string;
  n: number;
  zero_hits: number;
  common_final: string[] | null;
  top_pool_overlap: number | null;
}

export interface HarvestDeps {
  pool: {
    query: (text: string, params: unknown[]) => Promise<{ rows: AggRow[] }>;
  };
  days?: number;
  minCount?: number;
  limit?: number;
  /** Optional project scope (the log is cross-project; tests use this for hermeticity). */
  projectId?: string;
  /** Gold file whose queries are excluded (defaults to test/fixtures/recall-eval.json). */
  evalPath?: string;
  /** Output file override (tests point this at a tmp dir). */
  outPath?: string;
  log?: (msg: string) => void;
}

const here = dirname(fileURLToPath(import.meta.url));
const DEFAULT_EVAL_PATH = join(
  here,
  "..",
  "..",
  "test",
  "fixtures",
  "recall-eval.json",
);
const DEFAULT_OUT_DIR = join(here, "..", "..", "test", "fixtures");

/** Final hits found deep in the pool (below rank 10) mark a query the reranker had to
 *  rescue — interesting for gold rotation. */
export const LOW_OVERLAP_THRESHOLD = 0.5;

/** Normalized queries already present in the gold file (v1 and v2 rows both carry
 *  `query`, so this tolerates either shape). */
export function existingGoldQueries(evalPath: string): Set<string> {
  const raw = JSON.parse(readFileSync(evalPath, "utf8")) as {
    rows?: Array<{ query?: string }>;
  };
  const out = new Set<string>();
  for (const r of raw.rows ?? []) {
    if (typeof r.query === "string") out.add(r.query.trim().toLowerCase());
  }
  return out;
}

export async function harvestEval(
  deps: HarvestDeps,
): Promise<{ candidates: HarvestCandidate[]; path: string }> {
  const log = deps.log ?? ((m: string) => console.error(m));
  const days = deps.days ?? 30;
  const minCount = deps.minCount ?? 2;
  const limit = deps.limit ?? 40;

  const params: unknown[] = [days, minCount, limit];
  let projectFilter = "";
  if (deps.projectId) {
    params.push(deps.projectId);
    projectFilter = `AND project_id = $${params.length}`;
  }

  // Per-query aggregation over the window: count, zero-hit count, the most-common
  // final-ids array (mode — the gold hint), and the mean fraction of final ids found
  // in the pool's top 10 (low overlap = the reranker digs deep for this query).
  const { rows } = await deps.pool.query(
    `SELECT query,
            count(*)::int AS n,
            (count(*) FILTER (WHERE cardinality(final_ids) = 0))::int AS zero_hits,
            (mode() WITHIN GROUP (ORDER BY final_ids))::text[] AS common_final,
            avg(
              CASE WHEN cardinality(final_ids) = 0 THEN 0
                   ELSE (SELECT count(*) FROM unnest(final_ids) f
                         WHERE f = ANY (pool_ids[1:10]))::float / cardinality(final_ids)
              END
            )::float AS top_pool_overlap
     FROM memory.search_log
     WHERE created_at > now() - make_interval(days => $1) ${projectFilter}
     GROUP BY query
     HAVING count(*) >= $2
         OR count(*) FILTER (WHERE cardinality(final_ids) = 0) > 0
     ORDER BY count(*) DESC, query
     LIMIT $3`,
    params,
  );

  const evalPath = deps.evalPath ?? DEFAULT_EVAL_PATH;
  const gold = existingGoldQueries(evalPath);

  const candidates: HarvestCandidate[] = [];
  for (const r of rows) {
    if (gold.has(r.query.trim().toLowerCase())) continue;
    const archetype: HarvestCandidate["archetype"] =
      r.zero_hits > 0
        ? "zero-hit"
        : (r.top_pool_overlap ?? 1) < LOW_OVERLAP_THRESHOLD
          ? "low-overlap"
          : "frequent";
    candidates.push({
      query: r.query,
      suggestedGold: r.common_final ?? [],
      archetype,
      provenance: "log-harvest",
      approved: false,
      count: r.n,
    });
  }
  log(
    `harvest-eval: ${rows.length} aggregated queries → ${candidates.length} candidates ` +
      `(window ${days}d, min-count ${minCount}, already-in-gold excluded)`,
  );

  const date = new Date().toISOString().slice(0, 10);
  const path =
    deps.outPath ??
    join(DEFAULT_OUT_DIR, `eval-candidates-${date}-harvest.json`);
  writeFileSync(
    path,
    JSON.stringify(
      {
        _service: "memory-mcp",
        _generated: date,
        _note:
          "Log-harvested rotation candidates. Review each: verify the query, verify every " +
          "suggestedGold id, then merge approved rows into recall-eval.json v2 with " +
          "split:'dev', provenance:'log-harvest' and your approvedBy (+ changelog line). " +
          "Cleanup: DELETE FROM memory.search_log WHERE created_at < now() - interval '90 days';",
        candidates,
      },
      null,
      2,
    ) + "\n",
  );
  return { candidates, path };
}

// ── CLI entrypoint (npm run harvest-eval [-- --days 30 --min-count 2 --limit 40]) ─────
// Skipped when imported (tests import harvestEval directly).
function argValue(argv: string[], flag: string, dflt: number): number {
  const i = argv.indexOf(flag);
  if (i === -1 || i + 1 >= argv.length) return dflt;
  const v = Number(argv[i + 1]);
  return Number.isFinite(v) ? v : dflt;
}

const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  (async () => {
    const argv = process.argv.slice(2);
    const { candidates, path } = await harvestEval({
      pool: defaultPool,
      days: argValue(argv, "--days", 30),
      minCount: argValue(argv, "--min-count", 2),
      limit: argValue(argv, "--limit", 40),
    });
    console.error(
      `harvest-eval: wrote ${candidates.length} candidates → ${path}`,
    );
    await defaultPool.end();
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
