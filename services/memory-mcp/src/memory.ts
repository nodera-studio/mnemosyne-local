import { pool } from "./db/pool.js";
import {
  embed,
  embedContextualSingle,
  rerank,
  toVectorLiteral,
} from "./voyage.js";
import { config } from "./config.js";
import { summarizeMemory } from "./summarize.js";
import { contentSha256, insertMemoryRow } from "./db/insert-memory.js";

// ── Fusion constant (Wave P → wave-7 bakeoff) ────────────────────────────────────────
// RRF k stays a module const (the code default); the blend/decay knobs moved to
// config.blendConfig (src/config.ts, wave-7 AC-702) so the bakeoff winner can ship as
// compose env pins with the code default unchanged. `fuseCandidates` accepts an
// injectable `rrfK` for the wave-7 k-sweep axis; retrievalConfig() serializes the
// shipped DEFAULT (RRF_K) — sweep arms record their per-call k in their own run
// artifacts. A non-60 winner would ship as an env pin too (decision record:
// test/retune.md).
const RRF_K = 60;

/** Bump when `blendScores` SEMANTICS change (not when env pins retune the knobs) —
 *  serialized top-level into every run artifact so numbers are comparable only within
 *  one scoring generation. "blend-2" = the wave-7 configurable blend/decay engine. */
export const SCORING_VERSION = "blend-2";

export type MemoryType = "episodic" | "semantic" | "procedural" | "entity";

// ── Blend/decay configuration (wave-7, AC-702/AC-706) ───────────────────────────────

export type BlendForm = "additive" | "multiplicative";
export type DecayShape = "exp" | "power";

export interface BlendWeights {
  relevance: number;
  recency: number;
  importance: number;
}

/** Rows exempt from age decay (recency pinned to 1.0): by memory type or source_kind
 *  (e.g. `entity` rows and `decision` records — reference material, not news). */
export interface DecayExempt {
  types: MemoryType[];
  sourceKinds: string[];
}

export interface DecayConfig {
  shape: DecayShape;
  /** τ is the 1/e time constant in days (NOT the half-life — see config.ts). */
  tauDays: number;
  /** Per-type τ override; resolution is `tauDaysByType[type] ?? tauDays` (AC-706). */
  tauDaysByType: Partial<Record<MemoryType, number>>;
  /** Exponent for the `power` shape: `1 / (1 + age/τ)^powerExponent`. */
  powerExponent: number;
  exempt: DecayExempt;
}

export interface BlendConfig {
  form: BlendForm;
  weights: BlendWeights;
  decay: DecayConfig;
}

export interface Hit {
  id: string;
  title: string;
  type: MemoryType;
  snippet: string;
  /** Stored dense summary (wave-2, AC-808) — NULL until summarized. */
  summary: string | null;
  importance: number;
  createdAt: string;
  eventDate: string | null;
  effectiveDate: string;
  status: string;
}

export interface ScoredHit extends Hit {
  score: number;
}

interface CandidateRow {
  id: string;
  title: string;
  content: string;
  summary: string | null;
  type: MemoryType;
  importance: number;
  created_at: Date;
  event_date?: Date | null;
  source_kind: string | null;
  status: string;
  rrf: number;
}

const RECALL_LIMIT = 50; // per-arm (BM25 / vector) recall before fusion

// Named (and exported) so retrievalConfig() serializes the SAME values the code runs
// with — the snapshot can never drift from the pipeline (AC-104 groundwork).
export const RERANK_DOC_TRUNCATION = 1200; // chars of title+content handed to the reranker
export const SNIPPET_CHARS = 180; // whitespace-collapsed snippet length in Hit

// ts_headline options for query-aware snippets (display-only — deliberately NOT part of
// retrievalConfig(): headlines decorate the final hits and never touch pool or ranking).
// The selectors are PRIVATE-USE sentinels that cannot occur in stored content: a real
// lexical match is detected by sentinel presence, then the sentinels are rendered as
// markdown-safe `**` for display. Detecting on literal `**` would false-positive —
// ts_headline's no-match fallback is a document-prefix headline, and this corpus is
// markdown-heavy, so content's own `**` must never masquerade as a match. ≤2 fragments
// keep the snippet compact.
export const SNIPPET_START_SEL = "\uE000";
export const SNIPPET_STOP_SEL = "\uE001";
export const SNIPPET_HEADLINE_OPTS = `StartSel=${SNIPPET_START_SEL}, StopSel=${SNIPPET_STOP_SEL}, MaxFragments=2, MaxWords=20, MinWords=8, FragmentDelimiter=" … "`;

/**
 * The FULL retrieval configuration, serialized into every eval-run artifact (AC-104)
 * so any recorded number is reproducible. JSON-safe by construction (plain literals —
 * no functions, no Dates); wave-2's artifact writer and wave-6's bakeoff serialize it
 * verbatim.
 */
export function retrievalConfig() {
  const b = config.blendConfig;
  return {
    service: "memory-mcp",
    scoringVersion: SCORING_VERSION,
    rrfK: RRF_K,
    // Nested blend snapshot (AC-702) — fresh literals (never references into config)
    // so a serialized artifact can never be mutated through, or mutate, live config.
    blend: {
      form: b.form,
      weights: { ...b.weights },
      decay: {
        shape: b.decay.shape,
        tauDays: b.decay.tauDays,
        tauDaysByType: { ...b.decay.tauDaysByType },
        powerExponent: b.decay.powerExponent,
        exempt: {
          types: [...b.decay.exempt.types],
          sourceKinds: [...b.decay.exempt.sourceKinds],
        },
      },
    },
    candidatePool: config.candidatePool,
    recallLimit: RECALL_LIMIT,
    embedModel: config.embedModel,
    contextModel: config.contextModel,
    rerankModel: config.rerankModel,
    rerankDocTruncation: RERANK_DOC_TRUNCATION,
    // Wave-2 (AC-810): reranker docs are title\n(summary\n)content — the doc
    // composition is a ranking-bearing knob, so artifacts must carry it (AC-104).
    rerankDocIncludesSummary: true,
    snippetChars: SNIPPET_CHARS,
  };
}

// ── Decision-log typed shape (Wave 6, AC-040) ────────────────────────────────────────
// A sparse typed shape carried on memory.memories for source_kind='decision' rows. These
// fields are NULL on every other row. See sql/005_decision_log.sql for the axis split
// (decision_status vs status; supersedes_id vs superseded_by).
export const DECISION_STATUSES = ["active", "superseded", "deferred"] as const;
export type DecisionStatus = (typeof DECISION_STATUSES)[number];

export interface DecisionFields {
  decisionProject?: string;
  decisionStatus?: DecisionStatus;
  decidedAt?: string | Date;
  supersedesId?: string;
  decidedIn?: string;
  relatedIds?: string[];
}

// ── Write-path dedup (Wave 5, AC-301/AC-302) ─────────────────────────────────────────
// The row-level invariants (content_sha256 recipe, dup thresholds, the invariant
// INSERT itself) moved to src/db/insert-memory.ts — the ONE shared write path that
// storeMemory and the three services/importer/*.mjs scripts all route through
// (tech-debt 4761d86c). Re-exported here so existing consumers keep their imports.
export {
  contentSha256,
  DUP_COSINE_THRESHOLD,
  DUP_CANDIDATE_TOP_S,
} from "./db/insert-memory.js";
export type { DupCandidate } from "./db/insert-memory.js";

export async function storeMemory(
  input: {
    projectId: string;
    type: MemoryType;
    title: string;
    content: string;
    importance?: number;
    metadata?: Record<string, unknown>;
    sourceSessionId?: string;
    sourceKind?: string;
  } & DecisionFields,
): Promise<{ id: string; title: string; duplicate?: boolean }> {
  const isDecision = input.sourceKind === "decision";
  // AC-040: validate the typed shape in code (the DB CHECK is the backstop) so the
  // connector gets a clean error instead of a raw constraint violation.
  if (isDecision && input.decisionStatus !== undefined) {
    if (!DECISION_STATUSES.includes(input.decisionStatus)) {
      throw new Error(
        `invalid decisionStatus "${input.decisionStatus}" — must be one of ${DECISION_STATUSES.join(", ")}`,
      );
    }
  }
  // Decision columns: only populated for source_kind='decision' rows; NULL otherwise so
  // non-decision memories keep their existing shape. decided_at defaults to now() for a
  // decision when not supplied.
  const decisionStatus = isDecision ? (input.decisionStatus ?? "active") : null;
  const decidedAt = isDecision
    ? input.decidedAt
      ? new Date(input.decidedAt)
      : new Date()
    : null;
  const decisionProject = isDecision ? (input.decisionProject ?? null) : null;
  const supersedesId = isDecision ? (input.supersedesId ?? null) : null;
  const decidedIn = isDecision ? (input.decidedIn ?? null) : null;
  const relatedIds = isDecision ? (input.relatedIds ?? null) : null;

  // AC-301: exact-dup short-circuit BEFORE any embed spend. Scoped to the project's
  // ACTIVE, non-archived rows only — a superseded/archived row with the same hash must
  // not block a re-store (the re-store is how a retired fact comes back).
  //
  // Decision stores NEVER short-circuit: their supersession axis is user-authored
  // (supersedesId flips the prior decision below), and a content match against an
  // arbitrary active row must not silently swallow that bookkeeping. Duplicate decision
  // content is legal — the decision LOG is the point.
  //
  // Known TOCTOU: the SELECT runs outside the insert transaction and mem_sha carries no
  // unique constraint (deliberate — status is mutable, so the partial-index semantics
  // can't be expressed). Two concurrent identical stores can both insert; the batch
  // sweep (AC-306) is the backstop.
  const sha = contentSha256(input.content);
  if (!isDecision) {
    const dup = await pool.query<{ id: string; title: string }>(
      `SELECT id, title FROM memory.memories
       WHERE project_id = $1 AND content_sha256 = $2
         AND archived_at IS NULL AND COALESCE(status,'active') = 'active'
       LIMIT 1`,
      [input.projectId, sha],
    );
    if (dup.rows.length > 0) {
      return { ...dup.rows[0], duplicate: true };
    }
  }

  // Burn-in: write BOTH the legacy `embedding` (voyage-4-large) and the contextual
  // `embedding_v2` (voyage-context-4) so a rollback stays possible. After the Step 8
  // burn-in window, the legacy column + its embed() call are dropped.
  // The summary rides the same Promise.all: when the summarize gate is closed
  // (default) it resolves null instantly, so the write path gains zero latency;
  // when open, summarizeMemory's own 4 s timeout bounds it (AC-808).
  const text = `${input.title}\n${input.content}`;
  const [[vec], [vec2], summary] = await Promise.all([
    embed([text], "document"),
    embedContextualSingle([text], "document"),
    summarizeMemory(input.title, input.content),
  ]);

  // A transaction so the supersession bookkeeping (insert the new decision + flip the
  // superseded row) stays consistent — the chain never half-updates.
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // The shared invariant path (src/db/insert-memory.ts): AC-302 neighbor pass
    // (top-s cosine>threshold → metadata.dupCandidates, NO LLM) + the invariant
    // INSERT (content_sha256 + BOTH embedding columns). Runs on this transaction's
    // client, so the neighbor pass + insert stay inside the transaction as before.
    const inserted = await insertMemoryRow(
      client,
      {
        projectId: input.projectId,
        type: input.type,
        title: input.title,
        content: input.content,
        importance: input.importance,
        metadata: input.metadata,
        sourceSessionId: input.sourceSessionId,
        sourceKind: input.sourceKind,
        summary,
        decisionProject,
        decisionStatus,
        decidedAt,
        supersedesId,
        decidedIn,
        relatedIds,
      },
      { vec, vec2 },
    );
    const newId = inserted.id;

    // AC-041 supersession write: mark the superseded decision as superseded and set its
    // forward pointer (superseded_by) to this new decision, keeping the inverse of
    // supersedes_id in sync. Canonical chain direction stays supersedes_id (backward).
    if (supersedesId) {
      // Scope the supersession to a DECISION in the SAME project (Fix 4): a decision must
      // not flip another project's row or a non-decision memory. The id/project/kind guard
      // makes a cross-project or wrong-kind supersedesId a silent no-op (0 rows updated).
      await client.query(
        `UPDATE memory.memories
           SET decision_status = 'superseded', superseded_by = $1
         WHERE id = $2 AND project_id = $3 AND source_kind = 'decision'`,
        [newId, supersedesId, input.projectId],
      );
    }

    await client.query("COMMIT");
    return { id: inserted.id, title: inserted.title };
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

// ── Supersession chain (Wave 6, AC-041) ──────────────────────────────────────────────
export interface DecisionChainRow {
  id: string;
  title: string;
  decision_status: string | null;
  decision_project: string | null;
  decided_at: Date | null;
  supersedes_id: string | null;
  decided_in: string | null;
  depth: number;
}

/**
 * Walk the supersession chain BACKWARD from a decision via supersedes_id
 * (this decision → the one it superseded → …), as a recursive CTE. Carries a `path`
 * array + a `NOT (id = ANY(path))` guard so a malformed supersedes_id cycle terminates
 * instead of infinite-looping. Returns the chain ordered by depth (seed = depth 0).
 */
export async function decisionChain(id: string): Promise<DecisionChainRow[]> {
  const { rows } = await pool.query<DecisionChainRow>(
    `WITH RECURSIVE chain AS (
       SELECT m.id, m.title, m.decision_status, m.decision_project, m.decided_at,
              m.supersedes_id, m.decided_in, 0 AS depth, ARRAY[m.id] AS path
       FROM memory.memories m
       WHERE m.id = $1
       UNION ALL
       SELECT m.id, m.title, m.decision_status, m.decision_project, m.decided_at,
              m.supersedes_id, m.decided_in, c.depth + 1, c.path || m.id
       FROM memory.memories m
       JOIN chain c ON m.id = c.supersedes_id
       WHERE NOT (m.id = ANY(c.path))
     )
     SELECT id, title, decision_status, decision_project, decided_at,
            supersedes_id, decided_in, depth
     FROM chain
     ORDER BY depth`,
    [id],
  );
  return rows;
}

export interface MemoryRow {
  id: string;
  title: string;
  type: MemoryType;
  source_kind: string | null;
  status: string;
  importance: number;
  event_date: Date | null;
}

/**
 * Structured enumeration (NOT semantic search) — list/filter memories by
 * source_kind / type / status / tag, newest first, with counts. Powers
 * "list all open tech debts", "which implementations shipped", etc.
 */
export async function listMemories(input: {
  projectId: string;
  sourceKind?: string;
  type?: MemoryType;
  status?: string;
  tag?: string;
  limit: number;
}): Promise<{
  rows: MemoryRow[];
  total: number;
  active: number;
  resolved: number;
}> {
  const params: unknown[] = [input.projectId];
  let where = `project_id = $1 AND archived_at IS NULL`;
  // default to active unless a specific status (or 'all') is requested
  if (input.status && input.status !== "all") {
    params.push(input.status);
    where += ` AND COALESCE(status,'active') = $${params.length}`;
  } else if (!input.status) {
    where += ` AND COALESCE(status,'active') = 'active'`;
  }
  if (input.sourceKind) {
    params.push(input.sourceKind);
    where += ` AND source_kind = $${params.length}`;
  }
  if (input.type) {
    params.push(input.type);
    where += ` AND type = $${params.length}::memory.memory_type`;
  }
  if (input.tag) {
    params.push(input.tag);
    where += ` AND metadata->'tags' ? $${params.length}`;
  }

  const listParams = [...params, input.limit];
  const { rows } = await pool.query<MemoryRow>(
    `SELECT id, title, type, source_kind, COALESCE(status,'active') AS status, importance, event_date
     FROM memory.memories WHERE ${where}
     ORDER BY COALESCE(event_date, created_at) DESC
     LIMIT $${listParams.length}`,
    listParams,
  );

  // counts over the same kind filter (ignoring the status filter), for "X open / Y resolved"
  const countParams: unknown[] = [input.projectId];
  let cf = "";
  if (input.sourceKind) {
    countParams.push(input.sourceKind);
    cf = `AND source_kind = $${countParams.length}`;
  }
  if (input.type) {
    countParams.push(input.type);
    cf += ` AND type = $${countParams.length}::memory.memory_type`;
  }
  const c = await pool.query<{
    total: string;
    active: string;
    resolved: string;
  }>(
    `SELECT count(*) AS total,
            count(*) FILTER (WHERE COALESCE(status,'active')='active') AS active,
            count(*) FILTER (WHERE status='resolved') AS resolved
     FROM memory.memories WHERE project_id = $1 AND archived_at IS NULL ${cf}`,
    countParams,
  );
  return {
    rows,
    total: Number(c.rows[0].total),
    active: Number(c.rows[0].active),
    resolved: Number(c.rows[0].resolved),
  };
}

// ── Two-phase search core (Wave 1) ───────────────────────────────────────────────────
// searchMemory = fuseCandidates (ONE RRF SQL path → candidate pool) → rerankAndBlend
// (rerank → blend → shape). Both phases are exported so the eval harness can score the
// pool layer (Recall@25) and the full pipeline (nDCG@10) through the SAME code the live
// handler runs (AC-102) — never a duplicated SQL path.

/** A pooled candidate with its per-arm RRF ranks (NULL when absent from that arm). */
export interface FusedCandidate extends CandidateRow {
  bm25_rank: number | null;
  vec_rank: number | null;
}

export interface SearchMemoryResult {
  hits: Hit[];
  poolSize: number;
  requestedLimit: number;
  truncated: boolean;
  retriedWithoutFilters: boolean;
  droppedFilters: Record<string, string>;
}

function definedFilters(
  filters: Record<string, unknown>,
): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(filters).filter(([, v]) => {
      if (Array.isArray(v)) return v.length > 0;
      return v !== undefined;
    }),
  );
}

function formatDroppedFilters(filters: Record<string, string>): string {
  return Object.entries(filters)
    .map(([key, value]) => `${key}=${value}`)
    .join(", ");
}

function toIsoDateFields(row: {
  created_at: Date;
  event_date?: Date | null;
}): Pick<Hit, "createdAt" | "eventDate" | "effectiveDate"> {
  const createdAt = row.created_at.toISOString();
  const eventDate = row.event_date ? row.event_date.toISOString() : null;
  return {
    createdAt,
    eventDate,
    effectiveDate: eventDate ?? createdAt,
  };
}

export function toPublicHits(hits: ScoredHit[]): Hit[] {
  return hits.map(({ score: _score, ...hit }) => hit);
}

/**
 * Query-aware snippet decoration (AC-801/802): one extra small SQL query computes
 * ts_headline fragments for EXACTLY the final ≤limit hit ids — never the pool — using
 * the SAME text-search config + query function as the BM25 arm ('english' +
 * plainto_tsquery), so a headline lights up iff the ranking query matched lexically.
 * A hit's snippet is replaced only when the whitespace-collapsed headline actually
 * carries a match SENTINEL (SNIPPET_START_SEL): a sentinel-less headline is
 * ts_headline's no-match fallback (vector-only hit, stop-words-only query — even when
 * the content's own literal `**` markdown appears in it) and the deterministic
 * 180-char prefix is better. Accepted sentinels are rendered as `**` for display.
 * Display-only and error-swallowing by design (logSearch pattern): on ANY
 * failure the hits are returned unchanged — snippets must never fail a search.
 */
export async function applyQuerySnippets(
  query: string,
  hits: Hit[],
): Promise<Hit[]> {
  if (hits.length === 0) return hits;
  try {
    // Options passed as a text PARAMETER (never interpolated); left(content, 50000)
    // caps headline parsing cost on giant rows. Raw content is deliberately not
    // selected — only the headline travels back.
    const { rows } = await pool.query<{ id: string; headline: string }>(
      `SELECT id, ts_headline('english', left(content, 50000),
                              plainto_tsquery('english', $1), $3) AS headline
       FROM memory.memories
       WHERE id = ANY($2::uuid[])`,
      [query, hits.map((h) => h.id), SNIPPET_HEADLINE_OPTS],
    );
    const headlineById = new Map(rows.map((r) => [r.id, r.headline]));
    return hits.map((h) => {
      const cleaned = (headlineById.get(h.id) ?? "")
        .replace(/\s+/g, " ")
        .trim();
      if (cleaned.length === 0 || !cleaned.includes(SNIPPET_START_SEL)) {
        return h;
      }
      return {
        ...h,
        snippet: cleaned
          .replaceAll(SNIPPET_START_SEL, "**")
          .replaceAll(SNIPPET_STOP_SEL, "**"),
      };
    });
  } catch {
    return hits; // AC-802: snippet decoration must never surface into the search path
  }
}

/**
 * Phase 1: hybrid recall + RRF fusion. Embeds the query (contextual — the corpus is
 * voyage-context-4) and returns the fused candidate pool: `config.candidatePool` rows
 * max, RRF-ordered, with per-arm ranks. `input.limit` is the FINAL hit limit used by
 * `rerankAndBlend`, NOT the pool size — it is accepted here so `searchMemory` can pass
 * its input through unchanged.
 */
export async function fuseCandidates(input: {
  projectId: string;
  query: string;
  type?: MemoryType;
  tags?: string[];
  after?: string;
  limit?: number;
  /** Pre-computed query embedding — pass it to skip the embed call (e.g. the
   *  zero-pool retry reuses the first attempt's vector; one paid embed per search). */
  qvec?: number[];
  /** RRF k override for the wave-7 k-sweep axis (bakeoff only — the live handler
   *  always runs the RRF_K module default). Non-finite or ≤ 0 throws. */
  rrfK?: number;
}): Promise<FusedCandidate[]> {
  // Validate the k override BEFORE any spend (embed) or SQL — reject garbage loudly.
  // k travels as a bound SQL parameter ($5::float8), never string interpolation.
  const rrfK = input.rrfK ?? RRF_K;
  if (!Number.isFinite(rrfK) || rrfK <= 0) {
    throw new Error(
      `fuseCandidates: invalid rrfK ${String(input.rrfK)} — must be a finite number > 0`,
    );
  }
  // BOTH sides contextual (critical): the corpus is voyage-context-4 (embedding_v2), so
  // the query MUST be embedded by the same model family or recall craters.
  const qvec =
    input.qvec ?? (await embedContextualSingle([input.query], "query"))[0];
  const params: unknown[] = [
    input.query,
    toVectorLiteral(qvec),
    input.projectId,
    config.candidatePool,
    rrfK,
  ];
  let typeFilter = "";
  if (input.type) {
    params.push(input.type);
    typeFilter = `AND m.type = $${params.length}::memory.memory_type`;
  }
  let refinementFilter = "";
  if (input.tags && input.tags.length > 0) {
    params.push(input.tags);
    refinementFilter += ` AND m.metadata->'tags' ?| $${params.length}::text[]`;
  }
  if (input.after) {
    params.push(input.after);
    refinementFilter += ` AND COALESCE(m.event_date, m.created_at) >= $${params.length}::timestamptz`;
  }

  const sql = `
    WITH q AS (SELECT plainto_tsquery('english', $1) AS tsq),
    bm25 AS (
      SELECT m.id, row_number() OVER (ORDER BY ts_rank_cd(m.search_tsv, q.tsq) DESC) AS rank
      FROM memory.memories m, q
      WHERE m.archived_at IS NULL AND COALESCE(m.status,'active') = 'active' AND m.project_id = $3
        AND m.search_tsv @@ q.tsq ${typeFilter}${refinementFilter}
      LIMIT ${RECALL_LIMIT}
    ),
    vec AS (
      SELECT m.id, row_number() OVER (ORDER BY m.embedding_v2 <=> $2::halfvec) AS rank
      FROM memory.memories m
      WHERE m.archived_at IS NULL AND COALESCE(m.status,'active') = 'active' AND m.project_id = $3
        AND m.embedding_v2 IS NOT NULL ${typeFilter}${refinementFilter}
      ORDER BY m.embedding_v2 <=> $2::halfvec
      LIMIT ${RECALL_LIMIT}
    )
    SELECT m.id, m.title, m.content, m.summary, m.type, m.importance, m.created_at, m.event_date,
           m.source_kind,
           COALESCE(m.status,'active') AS status,
           bm25.rank::int AS bm25_rank, vec.rank::int AS vec_rank,
           (COALESCE(1.0/($5::float8+bm25.rank),0) + COALESCE(1.0/($5::float8+vec.rank),0))::float8 AS rrf
    FROM memory.memories m
    LEFT JOIN bm25 ON bm25.id = m.id
    LEFT JOIN vec  ON vec.id  = m.id
    WHERE bm25.id IS NOT NULL OR vec.id IS NOT NULL
    ORDER BY rrf DESC
    LIMIT $4`;

  const { rows } = await pool.query<FusedCandidate>(sql, params);
  return rows;
}

/** The row shape `blendScores` needs: the temporal/importance fields of a candidate
 *  plus the rerank `relevance` already attached (missing rerank score ⇒ caller sets 0). */
export interface BlendableCandidate {
  type: MemoryType;
  importance: number;
  created_at: Date;
  event_date?: Date | null;
  source_kind?: string | null;
  relevance: number;
}

/**
 * Reranker document composition (wave-2, AC-810): `title\n` + (`summary\n` when
 * non-null) + `content`, truncated AS A WHOLE to RERANK_DOC_TRUNCATION — prefix
 * shape, never summary-only. Slicing the JOINED string (not the content alone) is
 * what keeps the doc-length invariant and makes a summary displace content tail
 * within the same budget; with summary = NULL the output is byte-identical to the
 * pre-wave-2 `title\ncontent` doc, so golden pins hold with zero edits.
 */
export function buildRerankDoc(r: {
  title: string;
  content: string;
  summary?: string | null;
}): string {
  return [r.title, ...(r.summary ? [r.summary] : []), r.content]
    .join("\n")
    .slice(0, RERANK_DOC_TRUNCATION);
}

/**
 * Pure blend scoring (wave-7, AC-701/AC-706): computes each candidate's `final` score
 * from its rerank relevance + recency + importance under `cfg`, and returns the rows
 * DESC-sorted by the LEGACY comparator exactly (`b.final - a.final`, NO tie-breakers —
 * V8's stable sort preserves input order on ties, as before).
 *
 * Semantics (AC-706):
 *  - `effectiveDate = event_date ?? created_at` is the ONLY temporal source;
 *  - τ resolves per-type: `tauDaysByType[type] ?? tauDays`;
 *  - decay shapes: `exp` = e^(−age/τ); `power` = 1/(1 + age/τ)^powerExponent;
 *  - exempt rows (type ∈ exempt.types OR source_kind ∈ exempt.sourceKinds) get
 *    recency = 1.0 (no age penalty);
 *  - negative ages (future `event_date`) are NOT clamped (current behavior preserved);
 *  - forms: `additive` = w.rel·rel + w.rec·recency + w.imp·importance (the legacy
 *    formula); `multiplicative` = rel · (1 + w.rec·recency + w.imp·importance).
 */
export function blendScores<T extends BlendableCandidate>(
  scored: T[],
  cfg: BlendConfig,
  now: number = Date.now(),
): Array<T & { final: number }> {
  const out = scored.map((r) => {
    let recency: number;
    const exempt =
      cfg.decay.exempt.types.includes(r.type) ||
      (r.source_kind != null &&
        cfg.decay.exempt.sourceKinds.includes(r.source_kind));
    if (exempt) {
      recency = 1.0;
    } else {
      const ageDays =
        (now - new Date(r.event_date ?? r.created_at).getTime()) / 86_400_000;
      const tau = cfg.decay.tauDaysByType[r.type] ?? cfg.decay.tauDays;
      recency =
        cfg.decay.shape === "exp"
          ? Math.exp(-ageDays / tau)
          : 1 / Math.pow(1 + ageDays / tau, cfg.decay.powerExponent);
    }
    const final =
      cfg.form === "additive"
        ? cfg.weights.relevance * r.relevance +
          cfg.weights.recency * recency +
          cfg.weights.importance * r.importance
        : r.relevance *
          (1 +
            cfg.weights.recency * recency +
            cfg.weights.importance * r.importance);
    return { ...r, final };
  });
  out.sort((a, b) => b.final - a.final);
  return out;
}

/**
 * Phase 2: rerank the candidate pool (summary-aware docs via `buildRerankDoc`),
 * blend relevance with recency + importance (`blendScores` under
 * `config.blendConfig`), and shape the top-`limit` hits (180-char
 * whitespace-collapsed snippet, 4-decimal score).
 */
export async function rerankAndBlend(
  query: string,
  rows: FusedCandidate[],
  limit: number,
): Promise<ScoredHit[]> {
  const docs = rows.map((r) => buildRerankDoc(r));
  const ranked = await rerank(query, docs, rows.length);
  const relById = new Map<string, number>();
  for (const r of ranked) relById.set(rows[r.index].id, r.score);

  // Missing rerank score ⇒ relevance 0, as before the wave-7 split.
  const scored = blendScores(
    rows.map((r) => ({ ...r, relevance: relById.get(r.id) ?? 0 })),
    config.blendConfig,
  );

  return scored.slice(0, limit).map((r) => ({
    id: r.id,
    title: r.title,
    type: r.type,
    snippet: r.content.replace(/\s+/g, " ").slice(0, SNIPPET_CHARS),
    summary: r.summary,
    importance: r.importance,
    ...toIsoDateFields(r),
    status: r.status,
    score: Number(r.final.toFixed(4)),
  }));
}

// ── search_log (Wave 2, AC-107) ──────────────────────────────────────────────────────

/** One live search's log entry. Wave-4 extends `filters` (tags/after) — add fields
 *  there, not new columns. */
export interface SearchLogEntry {
  projectId: string;
  query: string;
  filters: Record<string, unknown>;
  poolIds: string[];
  finalIds: string[];
  poolMs: number;
  totalMs: number;
}

/**
 * FIRE-AND-FORGET write to memory.search_log (the harvest-eval raw material). Never
 * awaited, error swallowed (AC-107): a failed insert — or a missing table on a
 * pre-006 DB — must not affect search results or add latency. The promise is created
 * before `searchMemory` returns but deliberately not attached to the result.
 */
export function logSearch(entry: SearchLogEntry): void {
  void pool
    .query(
      `INSERT INTO memory.search_log
         (project_id, query, filters, pool_ids, final_ids, pool_ms, total_ms)
       VALUES ($1, $2, $3, $4::uuid[], $5::uuid[], $6, $7)`,
      [
        entry.projectId,
        entry.query,
        JSON.stringify(entry.filters),
        entry.poolIds,
        entry.finalIds,
        entry.poolMs,
        entry.totalMs,
      ],
    )
    .catch(() => {
      /* AC-107: logging must never surface into the search path */
    });
}

export async function searchMemory(input: {
  projectId: string;
  query: string;
  type?: MemoryType;
  tags?: string[];
  after?: string;
  limit: number;
}): Promise<SearchMemoryResult> {
  const t0 = Date.now();
  // Embed ONCE up front and hand the vector to both fuse calls — the zero-pool retry
  // must not pay a second Voyage embed (MEDIUM-001).
  const [qvec] = await embedContextualSingle([input.query], "query");
  let rows = await fuseCandidates({ ...input, qvec });
  let poolMs = Date.now() - t0;
  const droppedFilters: Record<string, string> = {};
  if (input.tags && input.tags.length > 0) {
    droppedFilters.tags = input.tags.join(",");
  }
  if (input.after) droppedFilters.after = input.after;
  const hasOptionalFilters = Object.keys(droppedFilters).length > 0;
  let retriedWithoutFilters = false;
  if (rows.length === 0 && hasOptionalFilters) {
    retriedWithoutFilters = true;
    // Re-measure so the logged pool_ms covers the retried pool — the one whose ids
    // land in pool_ids (LOW-006). The retry reuses qvec, so this window is SQL-only.
    const retryT0 = Date.now();
    rows = await fuseCandidates({
      projectId: input.projectId,
      query: input.query,
      type: input.type,
      limit: input.limit,
      qvec,
    });
    poolMs = Date.now() - retryT0;
  }
  const scoredHits =
    rows.length === 0
      ? []
      : await rerankAndBlend(input.query, rows, input.limit);
  // Snippet decoration happens AFTER ranking and BEFORE logging — logSearch records
  // ids only, so the log is unaffected either way (AC-801; getRecent is untouched).
  const hits = await applyQuerySnippets(input.query, toPublicHits(scoredHits));
  logSearch({
    projectId: input.projectId,
    query: input.query,
    filters: definedFilters({
      type: input.type,
      tags: input.tags,
      after: input.after,
      retried: retriedWithoutFilters ? true : undefined,
    }),
    poolIds: rows.map((r) => r.id),
    finalIds: hits.map((h) => h.id),
    poolMs,
    totalMs: Date.now() - t0,
  });
  return {
    hits,
    poolSize: rows.length,
    requestedLimit: input.limit,
    truncated: hits.length === input.limit && rows.length > input.limit,
    retriedWithoutFilters,
    droppedFilters,
  };
}

// Default memory_get response budget: 6000 chars ≈ 1500 tokens at ~4 chars/token.
export const MEMORY_GET_DEFAULT_MAX_CHARS = 6000;

/**
 * Budget a memory_get row to `maxChars` of content (AC-805/806). Truncates ONLY the
 * `content` field value — metadata stays intact and the JSON stays parseable — and
 * marks the cut explicitly with `truncated`/`totalChars` plus a re-fetch note naming
 * the escape hatches. `maxChars <= 0` means unlimited; under-budget rows (and rows
 * with a non-string content) pass through unchanged, with no truncation fields.
 */
export function budgetMemoryBody(
  row: Record<string, unknown>,
  maxChars: number,
): Record<string, unknown> {
  if (maxChars <= 0) return row;
  const content = row.content;
  if (typeof content !== "string" || content.length <= maxChars) return row;
  return {
    ...row,
    content: content.slice(0, maxChars),
    truncated: true,
    totalChars: content.length,
    note: `content truncated to ${maxChars} of ${content.length} chars — call memory_get again with full=true (or maxChars=0) for the complete body`,
  };
}

export async function getMemory(
  id: string,
): Promise<Record<string, unknown> | null> {
  const { rows } = await pool.query(
    `UPDATE memory.memories
       SET access_count = access_count + 1, last_accessed_at = now()
     WHERE id = $1
     RETURNING id, project_id, type, title, content, summary, importance, access_count,
               metadata, source_session_id, pinned, created_at, last_accessed_at,
               status, event_date, superseded_by`,
    [id],
  );
  return rows[0] ?? null;
}

export async function getRecent(input: {
  projectId: string;
  limit: number;
  type?: MemoryType;
}): Promise<Hit[]> {
  const params: unknown[] = [input.projectId, input.limit];
  let typeFilter = "";
  if (input.type) {
    typeFilter = "AND type = $3::memory.memory_type";
    params.push(input.type);
  }
  const { rows } = await pool.query<CandidateRow>(
    `SELECT id, title, content, summary, type, importance, created_at, event_date, source_kind,
            COALESCE(status,'active') AS status, 0 AS rrf
     FROM memory.memories
     WHERE archived_at IS NULL AND COALESCE(status,'active') = 'active' AND project_id = $1 ${typeFilter}
     ORDER BY created_at DESC
     LIMIT $2`,
    params,
  );
  return rows.map((r) => ({
    id: r.id,
    title: r.title,
    type: r.type,
    snippet: r.content.replace(/\s+/g, " ").slice(0, SNIPPET_CHARS),
    summary: r.summary,
    importance: r.importance,
    ...toIsoDateFields(r),
    status: r.status,
  }));
}

export async function updateMemory(
  id: string,
  fields: {
    title?: string;
    content?: string;
    importance?: number;
    pinned?: boolean;
    status?: string;
  },
): Promise<{ id: string } | null> {
  const sets: string[] = [];
  const params: unknown[] = [];
  let i = 1;
  if (fields.title !== undefined) {
    sets.push(`title = $${i++}`);
    params.push(fields.title);
  }
  if (fields.content !== undefined) {
    sets.push(`content = $${i++}`);
    params.push(fields.content);
  }
  if (fields.importance !== undefined) {
    sets.push(`importance = $${i++}`);
    params.push(fields.importance);
  }
  if (fields.pinned !== undefined) {
    sets.push(`pinned = $${i++}`);
    params.push(fields.pinned);
  }
  if (fields.status !== undefined) {
    sets.push(`status = $${i++}`);
    params.push(fields.status);
  }

  if (fields.title !== undefined || fields.content !== undefined) {
    const cur = await pool.query<{ title: string; content: string }>(
      `SELECT title, content FROM memory.memories WHERE id = $1`,
      [id],
    );
    if (cur.rows.length === 0) return null;
    const title = fields.title ?? cur.rows[0].title;
    const content = fields.content ?? cur.rows[0].content;
    // Burn-in: re-embed into BOTH columns (legacy + contextual) so a rollback works.
    const text = `${title}\n${content}`;
    const [vec] = await embed([text], "document");
    const [vec2] = await embedContextualSingle([text], "document");
    sets.push(`embedding = $${i++}::halfvec`);
    params.push(toVectorLiteral(vec));
    sets.push(`embedding_v2 = $${i++}::halfvec`);
    params.push(toVectorLiteral(vec2));
    if (fields.content !== undefined) {
      // Keep the AC-301 dedup key true to the row: a content edit that leaves the old
      // hash behind would let a later store of the OLD content short-circuit onto a row
      // that no longer says it.
      sets.push(`content_sha256 = $${i++}`);
      params.push(contentSha256(content));
      // The write-time dupCandidates flags recorded the OLD content's neighbors — a
      // content edit invalidates them (stale sims would feed the consolidation judge).
      // The batch self-join sweep (AC-306) rediscovers real pairs from the fresh
      // embedding, so dropping the hint costs nothing but stale-judge spend.
      sets.push(`metadata = metadata - 'dupCandidates' - 'dupFlaggedAt'`);
    }
    // Never-stale invariant (wave-2): a title/content change recomputes the summary;
    // a null result (gate closed or summarizer failure) CLEARS the stale one — the
    // next backfill run regenerates it.
    const summary = await summarizeMemory(title, content);
    sets.push(`summary = $${i++}`);
    params.push(summary);
  }
  if (sets.length === 0) return { id };
  params.push(id);
  const { rows } = await pool.query<{ id: string }>(
    `UPDATE memory.memories SET ${sets.join(", ")} WHERE id = $${i} RETURNING id`,
    params,
  );
  return rows[0] ?? null;
}

export async function archiveMemory(id: string): Promise<boolean> {
  const { rowCount } = await pool.query(
    `UPDATE memory.memories SET archived_at = now() WHERE id = $1 AND archived_at IS NULL`,
    [id],
  );
  return (rowCount ?? 0) > 0;
}

export async function getEntity(input: {
  projectId: string;
  nameOrId: string;
}): Promise<Record<string, unknown> | null> {
  const { rows } = await pool.query(
    `SELECT id, project_id, kind, external_id, name, summary, metadata, created_at, updated_at
     FROM memory.entities
     WHERE project_id = $1 AND (name ILIKE $2 OR id::text = $2)
     ORDER BY updated_at DESC LIMIT 1`,
    [input.projectId, input.nameOrId],
  );
  if (rows.length === 0) return null;
  const entity = rows[0] as { id: string };
  const edges = await pool.query(
    `SELECT e.relation, e.to_id, t.name AS to_name, t.kind AS to_kind
     FROM memory.entity_edges e JOIN memory.entities t ON t.id = e.to_id
     WHERE e.from_id = $1`,
    [entity.id],
  );
  return { ...entity, edges: edges.rows };
}

export function formatHits(
  hits: Hit[],
  options: Partial<SearchMemoryResult> = {},
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
  if (hits.length === 0) return [...lines, "No matching memories."].join("\n");
  lines.push(
    hits
      .map((h, i) => {
        // Status renders only when it deviates from the default — live search/recent
        // hits are always 'active', so a constant `status: active` is token waste
        // (LOW-001; the hit OBJECT keeps its `status` field either way).
        const statusNote = h.status === "active" ? "" : `, status: ${h.status}`;
        // Wave-2: the stored dense summary renders as a third indented line when
        // present; NULL-summary hits keep the exact pre-wave-2 two-line form.
        const summaryLine = h.summary ? `\n   summary: ${h.summary}` : "";
        return `${i + 1}. [${h.type}] ${h.title}  (${h.effectiveDate.slice(0, 10)}${statusNote}, id: ${h.id})\n   ${h.snippet}${summaryLine}`;
      })
      .join("\n"),
  );
  if (
    options.truncated &&
    options.requestedLimit !== undefined &&
    options.poolSize !== undefined
  ) {
    lines.push(
      `Showing top ${options.requestedLimit} of ${options.poolSize} candidates. Prefer several small targeted searches over one broad one — optionally narrow with type, tags, or after.`,
    );
  }
  return lines.join("\n");
}
