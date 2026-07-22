// Shared "insert with invariants" write path (tech-debt 4761d86c — importer
// write-path unification). ONE function owns the row-level invariants that every
// memory.memories insert must carry, whatever produced the row:
//
//   • content_sha256 — sha256 hex over CONTENT ONLY (the importers' original
//     recipe; title deliberately NOT hashed) — the AC-301 dedup key.
//   • metadata.dupCandidates — top-s cosine>threshold neighbors recorded at write
//     time (AC-302; a prioritization flag for batch consolidation, never an action).
//   • BOTH embedding columns — legacy `embedding` + contextual `embedding_v2`
//     (burn-in dual write; searchMemory filters `embedding_v2 IS NOT NULL`, so a
//     v2-less row is invisible to recall until a manual backfill).
//
// Consumers: storeMemory (src/memory.ts — passes its transaction client) and the
// three plain-node importers (services/importer/*.mjs — import the COMPILED
// dist/db/insert-memory.js and pass their own pg.Pool). Because importers embed in
// batches with their own rate limiting, the embeddings are INJECTED, pre-computed —
// this module never calls Voyage (and must stay dependency-free: node:crypto only,
// no config/pool/voyage imports, so the importers' runtime footprint stays tiny).

import { createHash } from "node:crypto";
import type { MemoryType } from "../memory.js"; // type-only — erased at compile, no runtime cycle

/**
 * Contract version of this module, asserted by the .mjs importers immediately after
 * their dynamic import of the COMPILED dist copy. dist/ is gitignored, so a stale
 * build can be silently present — the assert turns "stale dist with a changed
 * contract" from silent bad writes into a hard "rebuild memory-mcp" failure.
 * BUMP THIS whenever insertMemoryRow's signature, SQL, or invariants change.
 */
export const INSERT_MEMORY_CONTRACT = 1;

/**
 * Exact-dup key: sha256 hex over CONTENT ONLY — deliberately the importers' recipe
 * (`services/importer/distill-transcripts.mjs` `sha256(content)`), so write-path and
 * importer dedup keys never diverge. Title is NOT hashed: a retitled restatement of the
 * same content is still the same content.
 */
export function contentSha256(content: string): string {
  return createHash("sha256").update(content).digest("hex");
}

/** Cosine-similarity floor above which a neighbor is flagged as a dup CANDIDATE
 *  (AC-302 write-path flag; also the wave-5 consolidation pair threshold). */
export const DUP_COSINE_THRESHOLD = 0.9;
/** Top-s neighbors inspected per new memory on the write path. */
export const DUP_CANDIDATE_TOP_S = 3;

export interface DupCandidate {
  id: string;
  sim: number;
}

/** Serialize an embedding to a pgvector/halfvec literal (the voyage.ts recipe,
 *  duplicated here so this module stays import-free of config-bearing modules). */
export function toVectorLiteral(v: number[]): string {
  return `[${v.join(",")}]`;
}

/** Structural handle over pg.Pool / pg.PoolClient — anything with a `query`. Kept
 *  structural so the compiled dist never drags @types/pg into importer consumers. */
export interface Queryable {
  query(
    text: string,
    values?: unknown[],
  ): Promise<{ rows: Array<Record<string, unknown>> }>;
}

/** Pre-computed embeddings for the row: `vec` = legacy `embedding` (burn-in dual
 *  write), `vec2` = contextual `embedding_v2` (the recall-bearing column). */
export interface InsertVectors {
  vec: number[];
  vec2: number[];
}

export interface InsertableMemory {
  projectId: string;
  type: MemoryType;
  title: string;
  content: string;
  /** Defaults to 0.5 (the column default). */
  importance?: number;
  /** Caller metadata — dupCandidates/dupFlaggedAt are merged IN by this function. */
  metadata?: Record<string, unknown>;
  sourceSessionId?: string | null;
  sourceKind?: string | null;
  /** Memory lifecycle (active|superseded|archived|closed). Defaults to 'active',
   *  identical to the column default — importers pass 'superseded' for stub files. */
  status?: string;
  /** Provenance path (importers' idempotent re-ingest key). */
  sourceUri?: string | null;
  /** Date parsed from filename/thread ts (recency source). */
  eventDate?: string | Date | null;
  /** Stored dense summary — storeMemory's summarize-on-store gate result; importers
   *  pass nothing (NULL, backfilled later). */
  summary?: string | null;
  /** Decision-log typed shape (Wave 6, AC-040) — pre-RESOLVED by the caller
   *  (storeMemory nulls these for non-decisions); importers never set them. */
  decisionProject?: string | null;
  decisionStatus?: string | null;
  decidedAt?: Date | null;
  supersedesId?: string | null;
  decidedIn?: string | null;
  relatedIds?: string[] | null;
}

/**
 * Insert ONE memory row with every write-path invariant applied:
 *
 *  1. computes content_sha256 from the content (never trusts a caller-supplied hash);
 *  2. runs the AC-302 neighbor pass — top-s cosine neighbors of `vec2` above
 *     DUP_COSINE_THRESHOLD (project-scoped, active, non-archived, v2-embedded rows
 *     only) recorded into metadata.dupCandidates + dupFlaggedAt;
 *  3. INSERTs with BOTH embedding columns.
 *
 * Runs its two statements on the `db` handle it is given: storeMemory passes its
 * open transaction client (neighbor pass + insert stay inside the transaction);
 * importers pass a plain pool (per-row autocommit, as their inserts always were).
 * Deliberately NO exact-dup short-circuit here — that policy (project-scoped,
 * active-only, non-decision only, BEFORE embed spend) belongs to storeMemory;
 * importers pre-dedupe against their own hash/path sets before embedding.
 */
export async function insertMemoryRow(
  db: Queryable,
  input: InsertableMemory,
  vectors: InsertVectors,
): Promise<{ id: string; title: string; dupCandidates: DupCandidate[] }> {
  const sha = contentSha256(input.content);

  // AC-302: cheap write-path pass — one extra indexed vector query, NO LLM.
  // Skipped for non-active rows (importer 'superseded' stubs): the flags exist to
  // prioritize batch consolidation of LIVE rows, so on a stub they are inert
  // metadata bought with a wasted ANN query.
  let dupCandidates: DupCandidate[] = [];
  if ((input.status ?? "active") === "active") {
    const { rows: neighborRows } = await db.query(
      `SELECT id, 1 - (embedding_v2 <=> $1::halfvec) AS sim
       FROM memory.memories
       WHERE project_id = $2 AND archived_at IS NULL
         AND COALESCE(status,'active') = 'active' AND embedding_v2 IS NOT NULL
       ORDER BY embedding_v2 <=> $1::halfvec
       LIMIT ${DUP_CANDIDATE_TOP_S}`,
      [toVectorLiteral(vectors.vec2), input.projectId],
    );
    dupCandidates = neighborRows
      .filter((n) => Number(n.sim) > DUP_COSINE_THRESHOLD)
      .map((n) => ({
        id: String(n.id),
        sim: Number(Number(n.sim).toFixed(4)),
      }));
  }
  const metadata: Record<string, unknown> = { ...(input.metadata ?? {}) };
  if (dupCandidates.length > 0) {
    metadata.dupCandidates = dupCandidates;
    metadata.dupFlaggedAt = new Date().toISOString();
  }

  const { rows } = await db.query(
    `INSERT INTO memory.memories
       (project_id, type, title, content, importance, metadata, source_session_id, source_kind,
        embedding, embedding_v2, content_sha256,
        decision_project, decision_status, decided_at, supersedes_id, decided_in, related_ids,
        summary, status, source_uri, event_date)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::halfvec,$10::halfvec,$11,
             $12,$13,$14,$15,$16,$17,$18,$19,$20,$21)
     RETURNING id, title`,
    [
      input.projectId,
      input.type,
      input.title,
      input.content,
      input.importance ?? 0.5,
      JSON.stringify(metadata),
      input.sourceSessionId ?? null,
      input.sourceKind ?? null,
      toVectorLiteral(vectors.vec),
      toVectorLiteral(vectors.vec2),
      sha,
      input.decisionProject ?? null,
      input.decisionStatus ?? null,
      input.decidedAt ?? null,
      input.supersedesId ?? null,
      input.decidedIn ?? null,
      input.relatedIds ?? null,
      input.summary ?? null,
      input.status ?? "active",
      input.sourceUri ?? null,
      input.eventDate ?? null,
    ],
  );
  const row = rows[0] as { id: string; title: string };
  return { id: row.id, title: row.title, dupCandidates };
}
