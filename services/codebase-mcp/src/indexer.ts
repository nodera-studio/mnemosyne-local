import { readFileSync, statSync, readdirSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import type pg from "pg";
import { pool } from "./db/pool.js";
import { config } from "./config.js";
import { chunkFile, languageFor, fileSha256 } from "./chunker.js";
import { embedCode, toVectorLiteral } from "./voyage.js";
import { extractGraph, type ByNameEdge } from "./graph/extractor.js";

interface Ignorer {
  add(pattern: string | string[]): Ignorer;
  ignores(path: string): boolean;
}
const ignore = createRequire(import.meta.url)("ignore") as () => Ignorer;

const SKIP_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".turbo",
  ".cache",
  "coverage",
  "vendor",
  ".venv",
  "__pycache__",
  ".idea",
  ".vscode",
  ".nx",
  "out",
  ".svelte-kit",
]);
const MAX_FILE_BYTES = 400_000;
const EMBED_BATCH = 64;

function walk(
  dir: string,
  ig: ReturnType<typeof ignore>,
  base: string,
  out: string[],
): void {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    if (SKIP_DIRS.has(e.name)) continue;
    const full = join(dir, e.name);
    const rel = relative(base, full);
    if (rel && ig.ignores(rel)) continue;
    if (e.isDirectory()) walk(full, ig, base, out);
    else if (e.isFile()) out.push(full);
  }
}

/** Open an index_runs row at the start of a run. Returns its id. */
async function startRun(
  repositoryId: string,
  projectId: string,
): Promise<string> {
  const { rows } = await pool.query<{ id: string }>(
    `INSERT INTO codebase.index_runs (project_id, repository_id, phase)
     VALUES ($1, $2, 'scanning') RETURNING id`,
    [projectId, repositoryId],
  );
  return rows[0].id;
}

/** Best-effort progress update; never throws (it must not mask indexing work). */
async function updateRun(
  runId: string,
  fields: Record<string, unknown>,
): Promise<void> {
  const keys = Object.keys(fields);
  if (keys.length === 0) return;
  const sets = keys.map((k, i) => `${k} = $${i + 2}`).join(", ");
  try {
    await pool.query(`UPDATE codebase.index_runs SET ${sets} WHERE id = $1`, [
      runId,
      ...keys.map((k) => fields[k]),
    ]);
  } catch (e) {
    console.error(
      `[index_runs] update failed for ${runId}:`,
      (e as Error).message,
    );
  }
}

/** Has a cooperative cancel been requested for this run? */
async function cancelRequested(runId: string): Promise<boolean> {
  try {
    const { rows } = await pool.query<{ cancel_requested: boolean }>(
      "SELECT cancel_requested FROM codebase.index_runs WHERE id = $1",
      [runId],
    );
    return rows[0]?.cancel_requested === true;
  } catch {
    return false;
  }
}

export async function indexRepo(
  repoDir: string,
  repositoryId: string,
  projectId: string,
  // force/--force-graph (AC-007, refined): graph extraction is DECOUPLED from the
  // content_sha256 skip. Every run extracts + stages the graph (symbols upsert via the
  // file_id-keyed delete+reinsert, edges pushed to stagedEdges, code_chunks.imports
  // written) for EVERY graphable file, regardless of sha — so the repo-wide edge rebuild
  // in resolveEdges is always complete and an incremental reindex never drops unchanged
  // files' edges. The sha-skip now gates ONLY the expensive embed + chunk-content rewrite.
  // `force` additionally bypasses the sha-skip so an unchanged file is re-embedded /
  // re-chunked too (full rebuild); it is no longer the ONLY way to backfill the graph —
  // any reindex now backfills it.
  force = false,
): Promise<void> {
  const runId = await startRun(repositoryId, projectId);
  // Dedicated client holding a SESSION advisory lock for the WHOLE run (AC-006).
  // Two code_reindex calls for the same repo must serialize — the repo-wide post-walk
  // edge resolution would otherwise interleave and corrupt. A separate pooled client
  // is used so the pool can't recycle it mid-run; it is released in finally.
  const lockClient = await pool.connect();
  let lockAcquired = false;
  try {
    await lockClient.query("SELECT pg_advisory_lock(hashtext($1))", [
      repositoryId,
    ]);
    lockAcquired = true;
    const base = resolve(repoDir);
    const ig = ignore();
    for (const f of [".gitignore", ".indexignore"]) {
      try {
        ig.add(readFileSync(join(base, f), "utf8"));
      } catch {
        /* none */
      }
    }
    ig.add([
      "secrets/",
      "*.pem",
      "*.key",
      "*.min.js",
      "*.lock",
      "pnpm-lock.yaml",
      "package-lock.json",
      // Claude Code parallel-agent worktrees: full duplicate checkouts of the same
      // repo. Indexing them floods search with near-identical copies that outrank
      // the canonical files (observed on a large monorepo corpus: tens of thousands
      // of duplicate rows).
      ".claude/worktrees/",
    ]);

    const files: string[] = [];
    walk(base, ig, base, files);
    console.error(`scanning ${repositoryId}: ${files.length} candidate files`);
    await updateRun(runId, { files_total: files.length, files_done: 0 });

    let indexed = 0,
      skipped = 0,
      chunksTotal = 0,
      symbolsTotal = 0,
      cancelled = false;
    // Staged by-name edges collected across the whole repo; resolved to concrete
    // symbol ids in the post-walk pass AFTER every file's symbols exist. Small repos
    // stage in memory; the pg COPY-style insert below batches them into a temp table.
    const stagedEdges: ByNameEdge[] = [];
    for (const full of files) {
      if (await cancelRequested(runId)) {
        cancelled = true;
        break;
      }
      const lang = languageFor(full);
      if (!lang) {
        continue;
      }
      let st;
      try {
        st = statSync(full);
      } catch {
        continue;
      }
      if (st.size === 0 || st.size > MAX_FILE_BYTES) {
        skipped++;
        continue;
      }
      const rel = relative(base, full);
      let content;
      try {
        content = readFileSync(full, "utf8");
      } catch {
        continue;
      }
      const fhash = fileSha256(content);

      // Extract symbols/edges/imports for TS/TSX (no-op for other langs / parse errors).
      const graph = extractGraph(content, rel);

      const existing = await pool.query<{ id: string; content_sha256: string }>(
        "SELECT id, content_sha256 FROM codebase.files WHERE repository_id=$1 AND path=$2",
        [repositoryId, rel],
      );
      const unchanged = existing.rows[0]?.content_sha256 === fhash;

      // ── Graph-always path: the file's content_sha256 is UNCHANGED. The expensive
      //    embed + chunk-content rewrite is the ONLY work gated by the sha-skip, so for
      //    a non-forced run we skip that here — but we STILL extract + (re)write this
      //    file's symbols, code_chunks.imports, AND push its edges to stagedEdges so the
      //    repo-wide rebuild in resolveEdges stays complete (a partial-stage would zero
      //    every unchanged file's edges). `force` falls through to the full path below so
      //    an unchanged file is re-embedded too (full rebuild). ──────────────────────
      if (unchanged && !force) {
        const fileId = existing.rows[0].id;
        await updateRun(runId, { phase: "graph", current_file: rel });
        const client = await pool.connect();
        try {
          await client.query("BEGIN");
          symbolsTotal += await writeSymbols(
            client,
            graph,
            fileId,
            repositoryId,
            projectId,
            rel,
          );
          await client.query(
            "UPDATE codebase.code_chunks SET imports=$2::jsonb WHERE file_id=$1",
            [fileId, JSON.stringify(graph.imports)],
          );
          await client.query("COMMIT");
        } catch (e) {
          await client.query("ROLLBACK");
          throw e;
        } finally {
          client.release();
        }
        stagedEdges.push(...graph.edges);
        skipped++;
        continue;
      }

      const chunks = chunkFile(content);
      if (chunks.length === 0) {
        skipped++;
        continue;
      }

      await updateRun(runId, { phase: "embedding", current_file: rel });
      // ── Chunk-level embed cache (AC-404): before calling Voyage, reuse any stored
      //    vector whose content_sha256 matches — only cache MISSES are embedded. The
      //    file-level sha-skip above stays the outer gate; this covers moved/renamed
      //    files and `force` runs. Reuse crosses files/repos on purpose: the embed model
      //    is deterministic per content, and all rows share ONE model today. The cache
      //    is model-BLIND, so a model flip (bakeoff winner) must CLEAR stored vectors
      //    (the truncate in the bakeoff.md flip step) before its forced reindex —
      //    otherwise old-model vectors would be cache-hit and survive the flip.
      //    The stored halfvec is carried as its ::text literal (already `[...]`-shaped)
      //    straight back into the $::halfvec insert — never through toVectorLiteral
      //    again, so there is no float round-trip. ───────────────────────────────────
      const vecBySha = new Map<string, string>(); // content_sha256 → halfvec literal
      const shas = [...new Set(chunks.map((c) => c.contentSha256))];
      const { rows: cachedVecs } = await pool.query<{
        content_sha256: string;
        embedding: string;
      }>(
        `SELECT DISTINCT ON (content_sha256) content_sha256, embedding::text AS embedding
         FROM codebase.code_chunks
         WHERE content_sha256 = ANY($1) AND embedding IS NOT NULL
         ORDER BY content_sha256`,
        [shas],
      );
      for (const r of cachedVecs) vecBySha.set(r.content_sha256, r.embedding);
      const missShas: string[] = [];
      const missContents: string[] = [];
      const missSeen = new Set<string>();
      for (const c of chunks) {
        if (vecBySha.has(c.contentSha256) || missSeen.has(c.contentSha256))
          continue;
        missSeen.add(c.contentSha256);
        missShas.push(c.contentSha256);
        missContents.push(c.content);
      }
      for (let i = 0; i < missShas.length; i += EMBED_BATCH) {
        const fresh = await embedCode(
          missContents.slice(i, i + EMBED_BATCH),
          "document",
        );
        fresh.forEach((v, j) =>
          vecBySha.set(missShas[i + j], toVectorLiteral(v)),
        );
      }
      // Fail loudly if the embedder returned fewer vectors than inputs (the pre-cache
      // code crashed on undefined here too — a silent NULL embedding is worse).
      if (missShas.some((s) => !vecBySha.has(s))) {
        throw new Error(
          `embedCode returned fewer vectors than inputs for ${rel}`,
        );
      }
      console.error(
        `  ${rel}: embedded ${missShas.length}, cache-hit ${shas.length - missShas.length}`,
      );

      const importsJson = JSON.stringify(graph.imports);
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const fr = await client.query<{ id: string }>(
          `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
           VALUES ($1,$2,$3,$4,$5)
           ON CONFLICT (repository_id, path)
             DO UPDATE SET content_sha256=EXCLUDED.content_sha256, language=EXCLUDED.language, indexed_at=now()
           RETURNING id`,
          [repositoryId, projectId, rel, lang, fhash],
        );
        const fileId = fr.rows[0].id;
        await client.query(
          "DELETE FROM codebase.code_chunks WHERE file_id=$1",
          [fileId],
        );
        for (let i = 0; i < chunks.length; i++) {
          const c = chunks[i];
          await client.query(
            `INSERT INTO codebase.code_chunks
               (file_id,repository_id,project_id,file_path,language,symbol_name,symbol_kind,start_line,end_line,content,content_sha256,imports,embedding)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::halfvec)`,
            [
              fileId,
              repositoryId,
              projectId,
              rel,
              lang,
              c.symbolName,
              c.symbolKind,
              c.startLine,
              c.endLine,
              c.content,
              c.contentSha256,
              importsJson,
              vecBySha.get(c.contentSha256),
            ],
          );
          chunksTotal++;
        }
        // Per-file symbol replacement (AC-005): file_id-keyed delete+reinsert, the
        // symbols analog of the code_chunks delete above. Cascades to this file's
        // outgoing edges; cross-file edges into these symbols are rebuilt post-walk.
        symbolsTotal += await writeSymbols(
          client,
          graph,
          fileId,
          repositoryId,
          projectId,
          rel,
        );
        await client.query("COMMIT");
      } catch (e) {
        await client.query("ROLLBACK");
        throw e;
      } finally {
        client.release();
      }
      stagedEdges.push(...graph.edges);
      indexed++;
      if (indexed % 25 === 0) {
        console.error(`  ${indexed} files, ${chunksTotal} chunks...`);
        await updateRun(runId, {
          files_done: indexed,
          chunks_total: chunksTotal,
        });
      }
    }

    // ── Post-walk by-name edge resolution (AC-004) — the REPO-LEVEL phase. ────────
    // Names only resolve once ALL files' symbols exist, so we rebuild the whole repo's
    // edges here, not per file. A name may match MULTIPLE target symbols (ambiguous):
    // we insert one row per resolved candidate and let code_graph_expand (Wave 4) label
    // a target as ambiguous when the same (from, toName) resolved to >1 row (AC-010).
    await updateRun(runId, { phase: "resolving", current_file: null });
    const edgesTotal = await resolveEdges(repositoryId, stagedEdges);
    console.error(
      `done ${repositoryId}: ${indexed} indexed, ${skipped} skipped, ${chunksTotal} chunks total`,
    );
    await updateRun(runId, {
      phase: "done",
      files_done: indexed,
      chunks_total: chunksTotal,
      symbols_total: symbolsTotal,
      edges_total: edgesTotal,
      current_file: null,
      error: cancelled ? "cancelled by request" : null,
      finished_at: new Date(),
    });
  } catch (e) {
    // Detached process: this row is the ONLY failure signal. Keep the error-write
    // defensive so a DB error during error-handling can't mask the original throw.
    const message = (e as Error).message ?? String(e);
    try {
      await pool.query(
        `UPDATE codebase.index_runs
         SET phase='error', error=$2, finished_at=now()
         WHERE id=$1`,
        [runId, message],
      );
    } catch (writeErr) {
      console.error(
        "[index_runs] failed to record error row:",
        (writeErr as Error).message,
      );
    }
    throw e;
  } finally {
    // Release the per-repo advisory lock on the SAME client that acquired it, even on
    // throw — otherwise a crashed run wedges the repo. Escape hatch for a leaked lock:
    // SELECT pg_advisory_unlock_all() on the stuck session.
    if (lockAcquired) {
      try {
        await lockClient.query("SELECT pg_advisory_unlock(hashtext($1))", [
          repositoryId,
        ]);
      } catch (unlockErr) {
        console.error(
          "[advisory-lock] unlock failed:",
          (unlockErr as Error).message,
        );
      }
    }
    lockClient.release();
  }
}

/**
 * Write one file's symbols (file_id-keyed delete+reinsert). Returns the count inserted.
 * Mirrors the code_chunks delete+reinsert pattern; the (repository_id, file_path, name,
 * kind, start_line) unique index (sql/003) gives stable identity, so re-indexing a file
 * never duplicates rows. ON CONFLICT DO NOTHING absorbs the rare same-line overload.
 */
async function writeSymbols(
  client: pg.PoolClient,
  graph: { symbols: { name: string; kind: string; startLine: number }[] },
  fileId: string,
  repositoryId: string,
  projectId: string,
  filePath: string,
): Promise<number> {
  await client.query("DELETE FROM codebase.symbols WHERE file_id=$1", [fileId]);
  let n = 0;
  for (const s of graph.symbols) {
    await client.query(
      `INSERT INTO codebase.symbols
         (repository_id, project_id, file_id, name, kind, file_path, start_line)
       VALUES ($1,$2,$3,$4,$5,$6,$7)
       ON CONFLICT (repository_id, file_path, name, kind, start_line) DO NOTHING`,
      [repositoryId, projectId, fileId, s.name, s.kind, filePath, s.startLine],
    );
    n++;
  }
  return n;
}

/**
 * Repo-wide by-name edge resolution (AC-004). Full rebuild: delete all of the repo's
 * edges, then resolve every staged by-name edge to concrete symbol ids by joining on
 * name within the repository. Staged into a TEMP table (over a huge in-memory VALUES
 * list) so it scales past a few thousand files. Returns the resolved edge count.
 *
 * Two edge kinds land in symbol_edges:
 *  - `call`: join from_name → symbol AND to_name → symbol within the repo (both must
 *    resolve). A name matching >1 symbol fans out; Wave 4 labels the >1 case ambiguous
 *    (AC-010). Calls to built-ins (trim/toFixed) resolve to no symbol and are dropped.
 *  - `import`: from_name is null (file-level), so we anchor FROM every symbol declared
 *    in the importing file (from_file) and link TO the imported symbol (to_name). This
 *    gives symbol_edges the file→file import structure of the golden fixture while
 *    staying symbol→symbol (the FK shape). The import LOCAL binding is also surfaced in
 *    code_chunks.imports; symbol_edges carries the resolved dependency edge.
 */
/**
 * Lock-window note (Wave 3 graph hardening, AC-403): the heavy name-resolution JOINs are
 * staged into session TEMP tables BEFORE any transaction opens; the transaction (= the
 * window in which readers of symbol_edges see the repo's graph mid-swap) contains ONLY
 * `DELETE + INSERT ... FROM resolved_edges`. Staging reads `codebase.symbols` with no
 * transaction open — safe because the per-repo advisory lock (taken by indexRepo BEFORE
 * any staging, held for the whole run) serializes index runs, so symbols cannot mutate
 * under the staging reads. Exported (with an injectable client) so the staging/txn
 * statement ORDER is structurally testable.
 */
export async function resolveEdges(
  repositoryId: string,
  edges: ByNameEdge[],
  injectedClient?: pg.PoolClient,
): Promise<number> {
  const client = injectedClient ?? (await pool.connect());
  try {
    const callEdges = edges.filter(
      (e): e is ByNameEdge & { fromName: string } =>
        e.kind === "call" && e.fromName !== null,
    );
    const importEdges = edges.filter((e) => e.kind === "import");
    const staged = [...callEdges, ...importEdges];

    // ── Phase 1: stage + resolve, NO transaction, no lock on symbol_edges. ──────
    // TEMP tables are session-scoped and pooled sessions are reused, so drop any
    // leftovers first (ON COMMIT DROP is useless here — outside an explicit txn it
    // would drop the table at the end of the CREATE statement itself).
    await client.query("DROP TABLE IF EXISTS staged_edges");
    await client.query("DROP TABLE IF EXISTS resolved_edges");
    if (staged.length > 0) {
      await client.query(
        `CREATE TEMP TABLE staged_edges
           (from_name text, from_file text, to_name text, kind text)`,
      );
      const BATCH = 500;
      for (let i = 0; i < staged.length; i += BATCH) {
        const slice = staged.slice(i, i + BATCH);
        const values: string[] = [];
        const params: unknown[] = [];
        slice.forEach((e, j) => {
          const base = j * 4;
          values.push(`($${base + 1},$${base + 2},$${base + 3},$${base + 4})`);
          params.push(e.fromName, e.fromFile, e.toName, e.kind);
        });
        await client.query(
          `INSERT INTO staged_edges (from_name, from_file, to_name, kind)
           VALUES ${values.join(",")}`,
          params,
        );
      }

      await client.query(
        `CREATE TEMP TABLE resolved_edges
           (from_symbol uuid, to_symbol uuid, kind text)`,
      );
      // call edges: from_name → symbol, to_name → symbol (both in-repo). The FROM side is
      // anchored by BOTH name AND file_path (from_file is staged at insert time) so a call
      // inside `foo` links only from the `foo` that actually contains it — not from every
      // same-named `foo` across the repo. The TO side stays by-name (the intentional,
      // labeled ambiguity: a name matching >1 target fans out and Wave 4 flags it AC-010).
      // DISTINCT lives here (outside the txn) so the txn's INSERT is a plain copy.
      await client.query(
        `INSERT INTO resolved_edges (from_symbol, to_symbol, kind)
         SELECT DISTINCT sf.id, st.id, 'call'
         FROM staged_edges se
         JOIN codebase.symbols sf
           ON sf.repository_id = $1 AND sf.name = se.from_name
              AND sf.file_path = se.from_file
         JOIN codebase.symbols st
           ON st.repository_id = $1 AND st.name = se.to_name
         WHERE se.kind = 'call'`,
        [repositoryId],
      );
      // import edges: anchor from every symbol in the importing file → imported symbol.
      await client.query(
        `INSERT INTO resolved_edges (from_symbol, to_symbol, kind)
         SELECT DISTINCT sf.id, st.id, 'import'
         FROM staged_edges se
         JOIN codebase.symbols sf
           ON sf.repository_id = $1 AND sf.file_path = se.from_file
         JOIN codebase.symbols st
           ON st.repository_id = $1 AND st.name = se.to_name AND st.file_path <> se.from_file
         WHERE se.kind = 'import'`,
        [repositoryId],
      );
    }

    // ── Phase 2: the ONLY transaction — DELETE + INSERT-from-staged (AC-403). ───
    // Full repo-level edge rebuild; an empty stage still wipes the repo's edges
    // (a repo whose files lost all their edges must converge to zero rows).
    let inserted = 0;
    await client.query("BEGIN");
    try {
      await client.query(
        "DELETE FROM codebase.symbol_edges WHERE repository_id=$1",
        [repositoryId],
      );
      if (staged.length > 0) {
        const res = await client.query(
          `INSERT INTO codebase.symbol_edges (repository_id, from_symbol, to_symbol, kind)
           SELECT $1, from_symbol, to_symbol, kind FROM resolved_edges`,
          [repositoryId],
        );
        inserted = res.rowCount ?? 0;
      }
      await client.query("COMMIT");
    } catch (e) {
      await client.query("ROLLBACK");
      throw e;
    }
    return inserted;
  } finally {
    // Session-scoped TEMP tables must not leak into the next pooled checkout.
    try {
      await client.query("DROP TABLE IF EXISTS staged_edges");
      await client.query("DROP TABLE IF EXISTS resolved_edges");
    } catch {
      /* best-effort cleanup; the session may already be dead */
    }
    if (!injectedClient) client.release();
  }
}

// CLI: node dist/indexer.js <repoDirOrName> [projectId] [repositoryId] [force]
// Force/--force-graph (AC-007) is accepted as a positional token ("force"/"--force-graph"),
// or via FORCE_GRAPH=1 in the env — server.ts plumbs it through the spawned env.
// Guarded so importing this module (e.g. from tests) does not auto-run the CLI.
function runCli(): void {
  const raw = process.argv.slice(2);
  const forceTokens = new Set(["force", "--force-graph", "--force"]);
  const force =
    process.env.FORCE_GRAPH === "1" || raw.some((a) => forceTokens.has(a));
  const positional = raw.filter((a) => !forceTokens.has(a));
  const arg = positional[0];
  if (!arg) {
    console.error(
      "usage: indexer <repoDirName|absPath> [projectId] [repositoryId] [force]",
    );
    process.exit(1);
  }
  const repoDir = arg.startsWith("/") ? arg : join(config.reposRoot, arg);
  const projectId = positional[1] ?? config.defaultProjectId;
  const repositoryId =
    positional[2] ?? arg.replace(/\/+$/, "").split("/").pop()!;
  indexRepo(repoDir, repositoryId, projectId, force)
    .then(() => pool.end())
    .catch((e) => {
      console.error(e);
      process.exit(1);
    });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  runCli();
}
