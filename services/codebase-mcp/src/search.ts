import { pool } from "./db/pool.js";
import { embedCode, rerank, toVectorLiteral } from "./voyage.js";
import { config } from "./config.js";
import { mergeAdjacentHits, type MergeableCodeHit } from "./merge-hits.js";

export interface CodeHit {
  chunkId: string;
  filePath: string;
  startLine: number;
  endLine: number;
  symbolName: string | null;
  language: string | null;
  snippet: string;
  mergedCount?: number;
}

export interface ScoredCodeHit extends CodeHit {
  score: number;
}

interface Row {
  id: string;
  repository_id: string;
  file_path: string;
  start_line: number;
  end_line: number;
  symbol_name: string | null;
  language: string | null;
  content: string;
  rrf: number;
}

export const RECALL_LIMIT = 50;
export const RRF_K = 60;

// Named (and exported) so retrievalConfig() serializes the SAME values the code runs
// with — the snapshot can never drift from the pipeline (AC-104 groundwork).
export const RERANK_DOC_TRUNCATION = 1500; // chars of path+content handed to the reranker
export const SNIPPET_LINES = 4; // lines kept in a CodeHit snippet
export const SNIPPET_CHARS = 240; // char cap applied after the line cut

/**
 * The FULL retrieval configuration, serialized into every eval-run artifact (AC-104)
 * so any recorded number is reproducible. JSON-safe by construction (plain literals —
 * no functions, no Dates); wave-2's artifact writer and wave-6's bakeoff serialize it
 * verbatim.
 */
export function retrievalConfig() {
  return {
    service: "codebase-mcp",
    rrfK: RRF_K,
    candidatePool: config.candidatePool,
    recallLimit: RECALL_LIMIT,
    codeEmbedModel: config.codeEmbedModel,
    codeContextModel: config.codeContextModel,
    rerankModel: config.rerankModel,
    maxMergedLines: config.maxMergedLines,
    rerankDocTruncation: RERANK_DOC_TRUNCATION,
    snippetLines: SNIPPET_LINES,
    snippetChars: SNIPPET_CHARS,
  };
}

// ── Two-phase search core (Wave 1) ───────────────────────────────────────────────────
// searchCode = fuseCodeCandidates (ONE RRF SQL path → candidate pool) → rerankCodeHits
// (rerank → top-k shaping). Both phases are exported so the eval harness can score the
// pool layer (Recall@25) and the full pipeline (nDCG@10) through the SAME code the live
// handler runs (AC-102). Kept per-service duplicated with memory-mcp (in-family — this
// repo has NO shared package; see src/recall-math.ts header). src/db/bakeoff-embed.ts's
// `searchArm` stays the bakeoff-only column-parametrized clone — do NOT unify.

/** A pooled candidate with its per-arm RRF ranks (NULL when absent from that arm). */
export interface FusedCodeCandidate extends Row {
  bm25_rank: number | null;
  vec_rank: number | null;
}

export interface SearchCodeResult {
  hits: CodeHit[];
  poolSize: number;
  requestedLimit: number;
  truncated: boolean;
  retriedWithoutFilters: boolean;
  droppedFilters: Record<string, string>;
}

function escapeLike(input: string): string {
  return input
    .replaceAll("\\", "\\\\")
    .replaceAll("%", "\\%")
    .replaceAll("_", "\\_");
}

function formatDroppedFilters(filters: Record<string, string>): string {
  return Object.entries(filters)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
}

export function toPublicCodeHits(hits: ScoredCodeHit[]): CodeHit[] {
  return hits.map(({ score: _score, ...hit }) => hit);
}

/**
 * Phase 1: hybrid recall + RRF fusion. Embeds the query with the live code embedder and
 * returns the fused candidate pool: `config.candidatePool` rows max, RRF-ordered, with
 * per-arm ranks.
 */
export async function fuseCodeCandidates(input: {
  projectId: string;
  query: string;
  repo?: string;
  language?: string;
  path?: string;
  extension?: string;
  /** Pre-computed query embedding — pass it to skip the embed call (e.g. the
   *  zero-pool retry reuses the first attempt's vector; one paid embed per search). */
  qvec?: number[];
}): Promise<FusedCodeCandidate[]> {
  const qvec = input.qvec ?? (await embedCode([input.query], "query"))[0];
  const params: unknown[] = [
    input.query,
    toVectorLiteral(qvec),
    input.projectId,
    config.candidatePool,
  ];
  let filt = "";
  if (input.repo) {
    params.push(input.repo);
    filt += ` AND c.repository_id = $${params.length}`;
  }
  if (input.language) {
    params.push(input.language);
    filt += ` AND c.language = $${params.length}`;
  }
  if (input.path) {
    params.push(`%${escapeLike(input.path)}%`);
    filt += ` AND c.file_path LIKE $${params.length} ESCAPE '\\'`;
  }
  if (input.extension) {
    const ext = input.extension.replace(/^\.+/, "");
    params.push(`%.${escapeLike(ext)}`);
    filt += ` AND c.file_path LIKE $${params.length} ESCAPE '\\'`;
  }

  const sql = `
    WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),
    bm25 AS (
      SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.search_tsv, q.tsq) DESC) AS rank
      FROM codebase.code_chunks c, q
      WHERE c.project_id = $3 AND c.search_tsv @@ q.tsq ${filt}
      LIMIT ${RECALL_LIMIT}
    ),
    vec AS (
      SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> $2::halfvec) AS rank
      FROM codebase.code_chunks c
      WHERE c.project_id = $3 AND c.embedding IS NOT NULL ${filt}
      ORDER BY c.embedding <=> $2::halfvec
      LIMIT ${RECALL_LIMIT}
    )
    SELECT c.id, c.repository_id, c.file_path, c.start_line, c.end_line, c.symbol_name, c.language, c.content,
           bm25.rank::int AS bm25_rank, vec.rank::int AS vec_rank,
           (COALESCE(1.0/(${RRF_K}+bm25.rank),0) + COALESCE(1.0/(${RRF_K}+vec.rank),0))::float8 AS rrf
    FROM codebase.code_chunks c
    LEFT JOIN bm25 ON bm25.id = c.id
    LEFT JOIN vec  ON vec.id  = c.id
    WHERE bm25.id IS NOT NULL OR vec.id IS NOT NULL
    ORDER BY rrf DESC
    LIMIT $4`;

  const { rows } = await pool.query<FusedCodeCandidate>(sql, params);
  return rows;
}

/**
 * Phase 2: rerank the candidate pool (falling back to RRF order when the reranker
 * returns nothing) and shape the top-k hits (4-line/240-char snippet, 4-decimal score).
 */
export async function rerankCodeHits(
  query: string,
  rows: FusedCodeCandidate[],
  k: number,
): Promise<ScoredCodeHit[]> {
  const docs = rows.map((r) =>
    `${r.file_path}\n${r.content}`.slice(0, RERANK_DOC_TRUNCATION),
  );
  const ranked = await rerank(query, docs, rows.length);
  const order = ranked.length
    ? ranked
    : rows.map((_, i) => ({ index: i, score: 0 }));

  return order.slice(0, k).map(({ index, score }) => {
    const r = rows[index];
    return {
      chunkId: r.id,
      filePath: r.file_path,
      startLine: r.start_line,
      endLine: r.end_line,
      symbolName: r.symbol_name,
      language: r.language,
      snippet: r.content
        .split("\n")
        .slice(0, SNIPPET_LINES)
        .join("\n")
        .slice(0, SNIPPET_CHARS),
      score: Number(score.toFixed(4)),
    };
  });
}

function toMergeableCodeHits(
  hits: CodeHit[],
  rows: FusedCodeCandidate[],
): MergeableCodeHit[] {
  const repoByChunkId = new Map(rows.map((r) => [r.id, r.repository_id]));
  return hits.map((hit) => {
    const repositoryId = repoByChunkId.get(hit.chunkId);
    if (repositoryId === undefined) {
      throw new Error(`missing repository for code chunk ${hit.chunkId}`);
    }
    return { ...hit, repositoryId };
  });
}

function stripMergeRepository(hits: MergeableCodeHit[]): CodeHit[] {
  return hits.map(({ repositoryId: _repositoryId, ...hit }) => hit);
}

export async function searchCode(input: {
  projectId: string;
  query: string;
  repo?: string;
  language?: string;
  path?: string;
  extension?: string;
  k: number;
}): Promise<SearchCodeResult> {
  // Embed ONCE up front and hand the vector to both fuse calls — the zero-pool retry
  // must not pay a second Voyage embed (MEDIUM-001).
  const [qvec] = await embedCode([input.query], "query");
  let rows = await fuseCodeCandidates({ ...input, qvec });
  const droppedFilters: Record<string, string> = {};
  if (input.path) droppedFilters.path = input.path;
  if (input.extension) droppedFilters.extension = input.extension;
  const hasOptionalFilters = Object.keys(droppedFilters).length > 0;
  let retriedWithoutFilters = false;
  if (rows.length === 0 && hasOptionalFilters) {
    retriedWithoutFilters = true;
    rows = await fuseCodeCandidates({
      projectId: input.projectId,
      query: input.query,
      repo: input.repo,
      language: input.language,
      qvec,
    });
  }
  const scored =
    rows.length === 0 ? [] : await rerankCodeHits(input.query, rows, input.k);
  const hits = stripMergeRepository(
    mergeAdjacentHits(
      toMergeableCodeHits(toPublicCodeHits(scored), rows),
      config.maxMergedLines,
    ),
  );
  return {
    hits,
    poolSize: rows.length,
    requestedLimit: input.k,
    truncated: scored.length === input.k && rows.length > input.k,
    retriedWithoutFilters,
    droppedFilters,
  };
}

export function formatHits(
  hits: CodeHit[],
  options: Partial<SearchCodeResult> = {},
): string {
  const lines: string[] = [];
  if (
    hits.length > 0 &&
    options.retriedWithoutFilters &&
    options.droppedFilters
  ) {
    lines.push(
      `Note: no results matched filters {${formatDroppedFilters(options.droppedFilters)}}; showing unfiltered results. Filters are optional — retry with different values to narrow.`,
    );
  }
  if (hits.length === 0) return [...lines, "No matching code."].join("\n");
  lines.push(
    hits
      .map((h, i) => {
        const mergedNote =
          h.mergedCount && h.mergedCount > 1
            ? ` (spans ${h.mergedCount} chunks)`
            : "";
        return `${i + 1}. ${h.filePath}:${h.startLine}-${h.endLine}${h.symbolName ? ` (${h.symbolName})` : ""}${mergedNote}  [${h.chunkId}]\n   ${h.snippet.replace(/\n/g, " ⏎ ")}`;
      })
      .join("\n"),
  );
  if (
    options.truncated &&
    options.requestedLimit !== undefined &&
    options.poolSize !== undefined
  ) {
    lines.push(
      `Showing top ${options.requestedLimit} of ${options.poolSize} candidates. Prefer several small targeted searches — optionally narrow with repo, path, extension, or language. Fetch full context with code_get_file.`,
    );
  }
  return lines.join("\n");
}
