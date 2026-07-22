// Pure formatting for `code_index_status` — extracted so it is unit-testable
// without booting the MCP server (server.ts binds a port at import time).

export interface ChunkAggRow {
  repository_id: string;
  files: number | string;
  chunks: number | string;
  last_indexed: Date | string | null;
}

export interface IndexRunRow {
  repository_id: string;
  phase: string;
  files_done: number | null;
  files_total: number | null;
  chunks_total: number | null;
  current_file: string | null;
  error: string | null;
  started_at: Date | string;
  finished_at: Date | string | null;
}

/**
 * Render `code_index_status` output: per-repo chunk aggregates plus the latest
 * index_runs row appended as a `Last run:` line. Repos with a run but no chunks
 * yet (an in-flight first index, or one that failed early) still surface.
 */
export function formatIndexStatus(
  chunkRows: ChunkAggRow[],
  runRows: IndexRunRow[],
): string {
  const runByRepo = new Map<string, IndexRunRow>();
  for (const r of runRows) runByRepo.set(r.repository_id, r);

  const fmtRun = (r: IndexRunRow): string => {
    const counts = `${r.files_done ?? 0}/${r.files_total ?? "?"} files`;
    const cur = r.current_file ? `, current=${r.current_file}` : "";
    const err = r.error ? `, error="${r.error}"` : "";
    const span = `${r.started_at}→${r.finished_at ?? "running"}`;
    return `    Last run: ${r.phase} (${counts}, ${r.chunks_total ?? 0} chunks${cur}${err}) ${span}`;
  };

  const seen = new Set<string>();
  const lines: string[] = [];
  for (const r of chunkRows) {
    seen.add(r.repository_id);
    lines.push(
      `${r.repository_id}: ${r.files} files, ${r.chunks} chunks (last ${r.last_indexed})`,
    );
    const run = runByRepo.get(r.repository_id);
    if (run) lines.push(fmtRun(run));
  }
  // Repos with a run but no chunks yet (first index in flight or failed early).
  for (const run of runRows) {
    if (seen.has(run.repository_id)) continue;
    lines.push(`${run.repository_id}: 0 chunks (not yet indexed)`);
    lines.push(fmtRun(run));
  }

  return lines.length ? lines.join("\n") : "no repos indexed yet";
}
