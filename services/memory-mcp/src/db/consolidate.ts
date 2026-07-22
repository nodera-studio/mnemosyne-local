// PAID, operator-gated memory consolidation (wave-5 Step 3, AC-303/AC-305/AC-306).
//
// Two-pass, ADD-only, NEVER destructive:
//   pass 1 (SQL, free)  — candidate pairs with cosine similarity > DUP_COSINE_THRESHOLD.
//     The BATCH SELF-JOIN SWEEP is the completeness source of truth (AC-306): it needs
//     only project/status/kind + embedding_v2, so direct-SQL importer rows that bypassed
//     storeMemory (no content_sha256, no metadata.dupCandidates) are still discovered and
//     CAN lose. Write-path `metadata.dupCandidates` flags are merged in purely as a
//     prioritization optimization (they are judged first), never as the candidate source.
//   pass 2 (LLM, PAID)  — an assertion-equivalence judge (same CLAIM, not same topic;
//     ADD-only bias: doubt → DISTINCT) over numbered pair batches with strict-JSON
//     integer verdicts (1 = EQUIVALENT, 0 = DISTINCT).
//
// Apply phase (ONLY with --apply): the loser is marked status='superseded' +
// superseded_by=<winner> — content is never edited or deleted (AC-303), pinned rows never
// lose, decision rows (source_kind='decision') never enter (their supersession axis is
// user-authored — see src/memory.ts decision fields). Chains are permitted; the eval
// harness resolves superseded_by forward chains (AC-106), so gold is never orphaned.
//
// Fail-open judge rule (AC-305): a transport-level failure (judge throws, times out, or
// returns a wholly unparseable response) marks the WHOLE batch UNJUDGED (verdict -1); a
// single pair whose verdict violates the schema (unknown integer, missing `keep` on an
// EQUIVALENT verdict, a `keep` that would make a pinned row lose) marks THAT pair
// UNJUDGED while sibling pairs stand. UNJUDGED pairs: zero rows modified, NO
// judged-marker written — they re-enter the next run. The run always continues.
//
// Dry-run is the DEFAULT: the identical pipeline with zero UPDATEs, writing the report
// artifact test/runs/consolidate-<date>.json — the human-review surface, the undo log
// (pair list = exact flip-back targets), and the resumability checkpoint: the report is
// (re)written with complete:false after EVERY sweep page and EVERY judge batch, so a
// crash at any point resumes without re-buying already-judged verdicts. In apply mode
// the fully-judged report additionally lands BEFORE the first row flips (undo-log
// guarantee). The CLI never judge-and-applies fresh: `--apply` requires a reviewed
// same-day dry-run report (or an explicit --report <path>) — see resolveCliAction.
//
// AC-108: PAID — runs ONLY via `npm run consolidate -- --yes [--apply]`, main-guarded,
// never imported by server.ts, never in CI. Tests inject a mock judge (zero live calls).

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pool as defaultPool } from "./pool.js";
import { judgeComplete } from "../llm.js";
import { config } from "../config.js";
import { DUP_COSINE_THRESHOLD } from "../memory.js";

/** a-row page size for the self-join sweep (resumability granularity). */
export const SCAN_BATCH = 500;
/** Pairs presented to the judge per LLM call (one transport failure = one batch lost). */
export const JUDGE_BATCH = 8;
/** Neighbors inspected per row in the sweep (mirrors the write-path top-s). */
export const SWEEP_TOP_S = 3;
/** Per-side content truncation in the judge prompt. */
export const PAIR_CONTENT_CHARS = 700;
/** Rough per-pair token cost for the consent estimate line. */
export const EST_TOKENS_PER_PAIR = 500;

export type Verdict = 1 | 0 | -1; // 1 EQUIVALENT, 0 DISTINCT, -1 UNJUDGED (harness-assigned only)
export type PairSource = "dupCandidates" | "selfJoin" | "both";

/** A candidate pair. Slot `a` is ALWAYS the OLDER row (created_at, ties by id) — the
 *  judged-marker lives on `a`, and `keep` refers to these normalized slots. */
export interface CandidatePair {
  a: string;
  b: string;
  sim: number;
  source: PairSource;
}

export interface JudgedPair extends CandidatePair {
  verdict: Verdict;
  keep?: "a" | "b";
  reason?: string;
}

/** Row facts the judge prompt + schema validation need. */
export interface PairRowMeta {
  id: string;
  title: string;
  content: string;
  /** Wave-2 dense summary (nullable) — the judge reads it over content when present. */
  summary: string | null;
  createdAt: string;
  pinned: boolean;
}

export interface ConsolidateReport {
  config: {
    threshold: number;
    model: string;
    batch: number;
    judgeBatch: number;
    topS: number;
    /** Last processed sweep a-row id — the resume point when complete=false. */
    cursor: string | null;
    complete: boolean;
    apply: boolean;
    projectId: string;
    generatedAt: string;
  };
  pairs: JudgedPair[];
  judged: number;
  unjudged: number;
  wouldSupersede: number;
  applied: number;
}

export interface ConsolidateDeps {
  pool: {
    query: (
      text: string,
      params?: unknown[],
    ) => Promise<{ rows: Record<string, unknown>[]; rowCount?: number | null }>;
  };
  /** Injected judge completion (tests mock this — zero live calls). */
  judge: (system: string, user: string) => Promise<string>;
  projectId: string;
  /** DEFAULT false = dry-run (report only, zero row changes). */
  apply?: boolean;
  scanBatch?: number;
  judgeBatch?: number;
  topS?: number;
  threshold?: number;
  /** Report artifact path (tests point this at a tmp dir). */
  reportPath?: string;
  /** Resume an interrupted run: prior judged pairs stand, prior UNJUDGED pairs
   *  re-enter the judge queue, and the sweep resumes after config.cursor. */
  priorReport?: ConsolidateReport;
  model?: string;
  log?: (msg: string) => void;
}

export const CONSOLIDATE_COST_NOTE =
  "consolidate is a PAID operator script: candidate pairs (cosine > 0.90) are sent to the " +
  `Anthropic judge (model: ${config.consolidateModel}; est ~${EST_TOKENS_PER_PAIR} tokens/pair). Nothing was run.\n` +
  "Re-run with an explicit consent flag:\n" +
  "  npm run consolidate -- --yes            # dry-run (DEFAULT): report only, zero row changes\n" +
  "  npm run consolidate -- --yes --apply    # marks losers status='superseded' + superseded_by=<winner>";

/** Returns the refusal message when the paid-consent flag is absent, else null. */
export function guardConsolidateRun(argv: string[]): string | null {
  return argv.includes("--yes") ? null : CONSOLIDATE_COST_NOTE;
}

// ── Judge prompt (AC-305) ─────────────────────────────────────────────────────────────

export const JUDGE_SYSTEM = `You judge whether two memories assert the SAME CLAIM — not merely the same topic.

You receive numbered pairs. Answer with ONLY strict JSON, no prose, no code fences,
one entry per pair number:
{"1": {"verdict": 1, "keep": "a", "reason": "..."}, "2": {"verdict": 0}}

Rules:
- "verdict" must be the INTEGER 1 (EQUIVALENT) or 0 (DISTINCT) — never strings, never
  any other number.
- "keep" ("a" or "b") and a short "reason" are REQUIRED when the verdict is 1; omit
  both when the verdict is 0.
- Restatements or added-detail memories are NOT equivalent: if one memory adds any
  requirement, step, number, or operational detail the other does not assert, answer 0.
  Worked example: A = "Use Postgres RLS for tenant isolation"; B = "Use Postgres RLS
  for tenant isolation and set \`app.current_org_id\` before every query" → verdict 0,
  because B adds an operational requirement A does not assert.
- When in doubt, answer 0 (DISTINCT). Only a clear same-claim duplicate is a 1.
- For verdict 1, keep the richer/newer assertion; ties keep the newer memory (the later
  created date).
- A side marked "pinned" may never be discarded: either keep the pinned side or
  answer 0.`;

const collapse = (s: string) => s.replace(/\s+/g, " ").trim();

/** Numbered-pair user message for one judge batch. Ids are included in brackets so the
 *  operator can audit the prompt against the report. */
export function buildJudgeUser(
  batch: CandidatePair[],
  rowsById: Map<string, PairRowMeta>,
): string {
  return batch
    .map((p, i) => {
      const side = (slot: "a" | "b", id: string) => {
        const r = rowsById.get(id);
        if (!r) return `  ${slot} [${id}] (missing)`;
        const pin = r.pinned ? ", pinned" : "";
        // Summary-aware delta (wave-3, retrieval-token-efficiency plan): prefer the
        // wave-2 dense summary over raw content — cheaper per pair AND not
        // tail-truncated like a long content prefix. Content is the fallback for
        // rows the backfill has not reached (summary NULL) AND — matching
        // buildRerankDoc's truthiness — for an out-of-band empty-string summary,
        // so the judge never reads an empty body.
        const body = r.summary || r.content;
        return (
          `  ${slot} [${r.id}] (created ${r.createdAt}${pin}): ${collapse(r.title)}\n` +
          `     ${collapse(body).slice(0, PAIR_CONTENT_CHARS)}`
        );
      };
      return `Pair ${i + 1}:\n${side("a", p.a)}\n${side("b", p.b)}`;
    })
    .join("\n\n");
}

/** Transport-level judge failure: the WHOLE batch is UNJUDGED (AC-305). */
export class JudgeTransportError extends Error {}

/**
 * Validate one batch's judge output against the strict schema. Wholly unparseable
 * (no JSON object, JSON.parse failure, non-object root) → JudgeTransportError (caller
 * marks the whole batch UNJUDGED). Per-pair violations (missing entry, non-integer or
 * unknown verdict, missing/invalid `keep` on verdict 1, a `keep` that would make a
 * pinned row lose) → verdict -1 for THAT pair; sibling pairs stand.
 */
export function parseVerdicts(
  raw: string,
  batch: CandidatePair[],
  rowsById: Map<string, PairRowMeta>,
): JudgedPair[] {
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end <= start) {
    throw new JudgeTransportError(
      `judge output has no JSON object: ${raw.slice(0, 200)}`,
    );
  }
  let obj: unknown;
  try {
    obj = JSON.parse(raw.slice(start, end + 1));
  } catch (e) {
    throw new JudgeTransportError(
      `judge output is unparseable JSON: ${(e as Error).message}`,
    );
  }
  if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
    throw new JudgeTransportError("judge output root is not a JSON object");
  }
  const entries = obj as Record<string, unknown>;

  return batch.map((p, i) => {
    const unjudged: JudgedPair = { ...p, verdict: -1 };
    const entry = entries[String(i + 1)];
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      return unjudged;
    }
    const { verdict, keep, reason } = entry as {
      verdict?: unknown;
      keep?: unknown;
      reason?: unknown;
    };
    if (typeof verdict !== "number" || !Number.isInteger(verdict)) {
      return unjudged;
    }
    if (verdict === 0) return { ...p, verdict: 0 };
    if (verdict !== 1) return unjudged; // unknown integer → UNJUDGED
    if (keep !== "a" && keep !== "b") return unjudged; // missing keep on EQUIVALENT
    const loserId = keep === "a" ? p.b : p.a;
    if (rowsById.get(loserId)?.pinned) return unjudged; // pinned may never lose
    return {
      ...p,
      verdict: 1,
      keep,
      ...(typeof reason === "string" ? { reason } : {}),
    };
  });
}

// ── Candidate collection (SQL only — free) ───────────────────────────────────────────

interface RawPairRow {
  a_id: string;
  a_title: string;
  a_content: string;
  a_summary: string | null;
  a_created_at: Date;
  a_pinned: boolean;
  a_judged: string[];
  b_id: string | null;
  b_title: string | null;
  b_content: string | null;
  b_summary: string | null;
  b_created_at: Date | null;
  b_pinned: boolean | null;
  b_judged: string[] | null;
  sim: number | null;
}

interface CollectedPairs {
  pairs: CandidatePair[];
  rowsById: Map<string, PairRowMeta>;
  cursor: string | null;
  complete: boolean;
}

const pairKey = (x: string, y: string) => [x, y].sort().join("|");

function registerRow(
  rowsById: Map<string, PairRowMeta>,
  id: string,
  title: string,
  content: string,
  summary: string | null,
  createdAt: Date,
  pinned: boolean,
): void {
  if (!rowsById.has(id)) {
    rowsById.set(id, {
      id,
      title,
      content,
      summary,
      createdAt: createdAt.toISOString(),
      pinned,
    });
  }
}

/** Older row first (created_at, ties by id) — slot `a` carries the judged-marker. */
function normalizeSlots(
  x: { id: string; createdAt: string },
  y: { id: string; createdAt: string },
): [string, string] {
  if (x.createdAt < y.createdAt) return [x.id, y.id];
  if (x.createdAt > y.createdAt) return [y.id, x.id];
  return x.id < y.id ? [x.id, y.id] : [y.id, x.id];
}

/**
 * Collect candidate pairs: dupCandidates flags first (prioritization only), then the
 * paged batch self-join sweep (the AC-306 completeness source of truth — requires ONLY
 * an embedding, never content_sha256 or metadata flags). Pairs already judged (partner
 * id present in the OLDER row's metadata.consolidation.judged array) are skipped;
 * UNJUDGED pairs were never marked, so they re-enter here.
 */
export async function collectPairs(
  deps: Pick<
    ConsolidateDeps,
    "pool" | "projectId" | "scanBatch" | "topS" | "threshold"
  > & {
    resumeCursor?: string | null;
    /** Checkpoint hook: called after every non-final sweep page with the page cursor
     *  and every pair collected so far — the caller persists a resumable report. */
    onPage?: (cursor: string | null, pairsSoFar: CandidatePair[]) => void;
  },
): Promise<CollectedPairs> {
  const { pool, projectId } = deps;
  const scanBatch = deps.scanBatch ?? SCAN_BATCH;
  const topS = deps.topS ?? SWEEP_TOP_S;
  const threshold = deps.threshold ?? DUP_COSINE_THRESHOLD;

  const rowsById = new Map<string, PairRowMeta>();
  const byKey = new Map<string, CandidatePair>();
  const judgedByOlder = new Map<string, Set<string>>();

  const noteJudged = (id: string, judged: string[] | null | undefined) => {
    if (!judged || judged.length === 0) return;
    const set = judgedByOlder.get(id) ?? new Set<string>();
    for (const j of judged) set.add(j);
    judgedByOlder.set(id, set);
  };

  const addPair = (row: RawPairRow, source: "dupCandidates" | "selfJoin") => {
    if (row.b_id === null || row.sim === null || Number(row.sim) <= threshold) {
      return;
    }
    registerRow(
      rowsById,
      row.a_id,
      row.a_title,
      row.a_content,
      row.a_summary,
      row.a_created_at,
      row.a_pinned,
    );
    registerRow(
      rowsById,
      row.b_id,
      row.b_title ?? "",
      row.b_content ?? "",
      row.b_summary,
      row.b_created_at ?? row.a_created_at,
      row.b_pinned ?? false,
    );
    noteJudged(row.a_id, row.a_judged);
    noteJudged(row.b_id, row.b_judged);

    const [a, b] = normalizeSlots(
      rowsById.get(row.a_id)!,
      rowsById.get(row.b_id)!,
    );
    // Skip pairs the OLDER row already carries a judged-marker for (re-runs).
    if (judgedByOlder.get(a)?.has(b)) return;

    const key = pairKey(a, b);
    const existing = byKey.get(key);
    if (existing) {
      if (existing.source !== source) existing.source = "both";
      existing.sim = Math.max(existing.sim, Number(Number(row.sim).toFixed(4)));
    } else {
      byKey.set(key, { a, b, sim: Number(Number(row.sim).toFixed(4)), source });
    }
  };

  // Shared active/non-decision predicate — decision rows have their OWN user-authored
  // supersession axis and never enter similarity consolidation.
  const LIVE = (alias: string) =>
    `${alias}.archived_at IS NULL
     AND COALESCE(${alias}.status,'active') = 'active'
     AND ${alias}.source_kind IS DISTINCT FROM 'decision'`;

  // Guarded judged-array read: metadata is caller-suppliable, so a scalar at
  // {consolidation,judged} must read as [] instead of crashing jsonb_array_elements_text.
  const JUDGED = (alias: string) =>
    `ARRAY(SELECT jsonb_array_elements_text(
       CASE WHEN jsonb_typeof(${alias}.metadata #> '{consolidation,judged}') = 'array'
            THEN ${alias}.metadata #> '{consolidation,judged}' ELSE '[]'::jsonb END))`;

  // Pass A — write-path dupCandidates flags (prioritization optimization ONLY).
  // Robust against malformed caller metadata: a non-array dupCandidates reads as []
  // (the CASE runs INSIDE the lateral — a WHERE-level jsonb_typeof guard does not
  // reliably protect the lateral call), a non-numeric sim reads as NULL (dropped by
  // addPair), and the partner join compares p.id::text so an arbitrary non-uuid id
  // string can never hit a ::uuid cast error — it simply matches nothing.
  const flagged = await pool.query(
    `SELECT m.id AS a_id, m.title AS a_title, m.content AS a_content,
            m.summary AS a_summary,
            m.created_at AS a_created_at, m.pinned AS a_pinned,
            ${JUDGED("m")} AS a_judged,
            p.id AS b_id, p.title AS b_title, p.content AS b_content,
            p.summary AS b_summary,
            p.created_at AS b_created_at, p.pinned AS b_pinned,
            ${JUDGED("p")} AS b_judged,
            CASE WHEN (c ->> 'sim') ~ '^-?[0-9]+(\\.[0-9]+)?([eE][+-]?[0-9]+)?$'
                 THEN (c ->> 'sim')::float8 ELSE NULL END AS sim
     FROM memory.memories m
     CROSS JOIN LATERAL jsonb_array_elements(
       CASE WHEN jsonb_typeof(m.metadata -> 'dupCandidates') = 'array'
            THEN m.metadata -> 'dupCandidates' ELSE '[]'::jsonb END) c
     JOIN memory.memories p ON p.id::text = (c ->> 'id')
     WHERE m.project_id = $1 AND ${LIVE("m")}
       AND p.project_id = $1 AND ${LIVE("p")}`,
    [projectId],
  );
  for (const row of flagged.rows as unknown as RawPairRow[]) {
    addPair(row, "dupCandidates");
  }

  // Pass B — the paged self-join sweep (source of truth, AC-306).
  let cursor: string | null = deps.resumeCursor ?? null;
  let complete = false;
  for (;;) {
    const page = await pool.query(
      `SELECT a.id AS a_id, a.title AS a_title, a.content AS a_content,
              a.summary AS a_summary,
              a.created_at AS a_created_at, a.pinned AS a_pinned,
              ARRAY(SELECT jsonb_array_elements_text(a.judged)) AS a_judged,
              n.id AS b_id, n.title AS b_title, n.content AS b_content,
              n.summary AS b_summary,
              n.created_at AS b_created_at, n.pinned AS b_pinned,
              ARRAY(SELECT jsonb_array_elements_text(n.judged)) AS b_judged,
              n.sim::float8 AS sim
       FROM (
         SELECT id, title, content, summary, created_at, pinned, embedding_v2,
                CASE WHEN jsonb_typeof(metadata #> '{consolidation,judged}') = 'array'
                     THEN metadata #> '{consolidation,judged}' ELSE '[]'::jsonb END AS judged
         FROM memory.memories a0
         WHERE a0.project_id = $1 AND ${LIVE("a0")}
           AND a0.embedding_v2 IS NOT NULL
           AND ($2::uuid IS NULL OR a0.id > $2::uuid)
         ORDER BY id
         LIMIT $3
       ) a
       LEFT JOIN LATERAL (
         SELECT b.id, b.title, b.content, b.summary, b.created_at, b.pinned,
                CASE WHEN jsonb_typeof(b.metadata #> '{consolidation,judged}') = 'array'
                     THEN b.metadata #> '{consolidation,judged}' ELSE '[]'::jsonb END AS judged,
                1 - (b.embedding_v2 <=> a.embedding_v2) AS sim
         FROM memory.memories b
         WHERE b.project_id = $1 AND b.id <> a.id AND ${LIVE("b")}
           AND b.embedding_v2 IS NOT NULL
         ORDER BY b.embedding_v2 <=> a.embedding_v2
         LIMIT $4
       ) n ON true
       ORDER BY a.id`,
      [projectId, cursor, scanBatch, topS],
    );
    const rows = page.rows as unknown as RawPairRow[];
    const aIds = new Set(rows.map((r) => r.a_id));
    for (const row of rows) addPair(row, "selfJoin");
    if (aIds.size > 0) cursor = [...aIds].sort().at(-1) ?? cursor;
    if (aIds.size < scanBatch) {
      complete = true;
      break;
    }
    deps.onPage?.(cursor, [...byKey.values()]);
  }

  // dupCandidates-sourced pairs first — the flags are a PRIORITIZATION optimization.
  const priority: Record<PairSource, number> = {
    both: 0,
    dupCandidates: 0,
    selfJoin: 1,
  };
  const pairs = [...byKey.values()].sort(
    (x, y) =>
      priority[x.source] - priority[y.source] ||
      y.sim - x.sim ||
      pairKey(x.a, x.b).localeCompare(pairKey(y.a, y.b)),
  );
  return { pairs, rowsById, cursor, complete };
}

// ── Judging (fail-open, AC-305) ───────────────────────────────────────────────────────

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

/** Judge all pairs in batches. Any transport-level failure (judge throws / wholly
 *  unparseable output) marks that WHOLE batch UNJUDGED and the run CONTINUES. */
export async function judgePairs(
  deps: Pick<ConsolidateDeps, "judge" | "judgeBatch" | "log"> & {
    /** Checkpoint hook: called after every non-final judge batch with verdicts so far
     *  and the pairs still queued — the caller persists a resumable report so a crash
     *  never re-buys already-judged pairs. */
    onBatch?: (judgedSoFar: JudgedPair[], remaining: CandidatePair[]) => void;
  },
  pairs: CandidatePair[],
  rowsById: Map<string, PairRowMeta>,
): Promise<JudgedPair[]> {
  const log = deps.log ?? ((m: string) => console.error(m));
  const judgeBatch = deps.judgeBatch ?? JUDGE_BATCH;
  const out: JudgedPair[] = [];
  let processed = 0;
  for (const batch of chunk(pairs, judgeBatch)) {
    try {
      const raw = await deps.judge(
        JUDGE_SYSTEM,
        buildJudgeUser(batch, rowsById),
      );
      out.push(...parseVerdicts(raw, batch, rowsById));
    } catch (e) {
      // Fail OPEN: transport failure → the whole batch is UNJUDGED (-1); zero rows
      // will be modified for these pairs and no judged-marker is written, so they
      // re-enter the next run.
      log(
        `consolidate: judge batch of ${batch.length} UNJUDGED (${(e as Error).message})`,
      );
      out.push(...batch.map((p) => ({ ...p, verdict: -1 as const })));
    }
    processed += batch.length;
    if (processed < pairs.length) {
      deps.onBatch?.([...out], pairs.slice(processed));
    }
  }
  return out;
}

// ── Apply phase (ONLY with --apply) ───────────────────────────────────────────────────

/** Supersede one loser: status flip + forward pointer, guarded so a pinned, already-
 *  superseded, or decision row can never be flipped — content is NEVER touched. The
 *  winner must itself still be active: two overlapping stale reports with opposite
 *  verdicts could otherwise create an A↔B supersession cycle. */
const SUPERSEDE_SQL = `
  UPDATE memory.memories
  SET status = 'superseded', superseded_by = $1
  WHERE id = $2 AND pinned = false
    AND COALESCE(status,'active') = 'active'
    AND source_kind IS DISTINCT FROM 'decision'
    AND EXISTS (
      SELECT 1 FROM memory.memories w
      WHERE w.id = $1 AND w.archived_at IS NULL
        AND COALESCE(w.status,'active') = 'active'
    )`;

/** Append the partner id to the OLDER row's metadata.consolidation.judged array
 *  (idempotent). UNJUDGED pairs never reach this. metadata is caller-suppliable via
 *  memory_store, so a non-object `consolidation` or non-array `judged` is overwritten
 *  with a clean shape rather than turning the marker write into a silent no-op (which
 *  would re-judge — re-spend — the pair on every future run). */
const MARKER_SQL = `
  UPDATE memory.memories
  SET metadata = jsonb_set(
    jsonb_set(metadata, '{consolidation}',
              CASE WHEN jsonb_typeof(metadata -> 'consolidation') = 'object'
                   THEN metadata -> 'consolidation' ELSE '{}'::jsonb END, true),
    '{consolidation,judged}',
    CASE WHEN jsonb_typeof(metadata #> '{consolidation,judged}') = 'array'
         THEN metadata #> '{consolidation,judged}' ELSE '[]'::jsonb END
      || to_jsonb($2::text),
    true)
  WHERE id = $1
    AND NOT COALESCE(
      jsonb_typeof(metadata #> '{consolidation,judged}') = 'array'
      AND (metadata #> '{consolidation,judged}') ? $2,
      false
    )`;

/** Apply judged verdicts: verdict-1 losers flip to superseded; every JUDGED pair
 *  (verdict 0 or 1) gets its marker; UNJUDGED (-1) pairs modify nothing. */
export async function applyJudged(
  pool: ConsolidateDeps["pool"],
  judged: JudgedPair[],
  log: (msg: string) => void,
): Promise<{ superseded: number; markers: number }> {
  let superseded = 0;
  let markers = 0;
  for (const p of judged) {
    if (p.verdict === -1) continue;
    if (p.verdict === 1 && p.keep) {
      const winner = p.keep === "a" ? p.a : p.b;
      const loser = p.keep === "a" ? p.b : p.a;
      const res = await pool.query(SUPERSEDE_SQL, [winner, loser]);
      const n = res.rowCount ?? 0;
      superseded += n;
      if (n > 0) log(`consolidate: ${loser} superseded by ${winner}`);
    }
    await pool.query(MARKER_SQL, [p.a, p.b]);
    markers += 1;
  }
  return { superseded, markers };
}

/**
 * Apply a previously-written (human-reviewed) DRY-RUN report without re-judging:
 * verdict-1 pairs supersede, every judged pair gets its marker. The same DB guards
 * (pinned/status/kind) re-apply — a stale report can only no-op, never corrupt.
 */
export async function applyFromReport(
  pool: ConsolidateDeps["pool"],
  report: ConsolidateReport,
  log: (msg: string) => void = (m) => console.error(m),
  expectedProjectId?: string,
): Promise<{ superseded: number; markers: number }> {
  if (expectedProjectId && report.config.projectId !== expectedProjectId) {
    // UUID-targeted, so still safe — but a cross-project report is almost certainly
    // an operator env mix-up worth flagging before the flips land.
    log(
      `consolidate: WARNING — report was generated for project "${report.config.projectId}" ` +
        `but the current project is "${expectedProjectId}"; applying to the report's rows anyway`,
    );
  }
  return applyJudged(pool, report.pairs, log);
}

// ── Orchestration + report ────────────────────────────────────────────────────────────

const here = dirname(fileURLToPath(import.meta.url));
const RUNS_DIR = join(here, "..", "..", "test", "runs");

export function defaultReportPath(date = new Date()): string {
  return join(RUNS_DIR, `consolidate-${date.toISOString().slice(0, 10)}.json`);
}

/** Merge freshly-collected pairs with a prior report's re-entered UNJUDGED pairs:
 *  carried (already-judged) pairs are dropped, duplicates dedupe by pair key. */
function mergePending(
  fresh: CandidatePair[],
  reentered: CandidatePair[],
  carriedKeys: Set<string>,
): CandidatePair[] {
  const out: CandidatePair[] = [];
  const seen = new Set<string>();
  for (const p of [...fresh, ...reentered]) {
    const key = pairKey(p.a, p.b);
    if (carriedKeys.has(key) || seen.has(key)) continue;
    seen.add(key);
    out.push(p);
  }
  return out;
}

export async function consolidate(
  deps: ConsolidateDeps,
): Promise<{ report: ConsolidateReport; reportPath: string }> {
  const log = deps.log ?? ((m: string) => console.error(m));
  const apply = deps.apply ?? false;
  const prior = deps.priorReport;
  const reportPath = deps.reportPath ?? defaultReportPath();

  // The report file is the resumability checkpoint: it is (re)written after every
  // sweep page and every judge batch with complete:false, so a crash at ANY point
  // leaves a resumable report — already-bought verdicts are never re-bought.
  // config.complete=true means the whole RUN (collection + judging) finished.
  const writeReport = (
    pairs: JudgedPair[],
    cursor: string | null,
    complete: boolean,
    applied: number,
  ): ConsolidateReport => {
    const judged = pairs.filter((p) => p.verdict !== -1).length;
    const report: ConsolidateReport = {
      config: {
        threshold: deps.threshold ?? DUP_COSINE_THRESHOLD,
        model: deps.model ?? config.consolidateModel,
        batch: deps.scanBatch ?? SCAN_BATCH,
        judgeBatch: deps.judgeBatch ?? JUDGE_BATCH,
        topS: deps.topS ?? SWEEP_TOP_S,
        cursor,
        complete,
        apply,
        projectId: deps.projectId,
        generatedAt: new Date().toISOString(),
      },
      pairs,
      judged,
      unjudged: pairs.length - judged,
      wouldSupersede: pairs.filter((p) => p.verdict === 1).length,
      applied,
    };
    mkdirSync(dirname(reportPath), { recursive: true });
    writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
    return report;
  };

  // Prior interrupted/completed run: judged pairs stand (never re-judged, never
  // re-bought); UNJUDGED pairs re-enter the queue; an incomplete sweep resumes
  // after its cursor.
  const carried: JudgedPair[] = [];
  const reentered: CandidatePair[] = [];
  for (const p of prior?.pairs ?? []) {
    if (p.verdict === -1) {
      reentered.push({ a: p.a, b: p.b, sim: p.sim, source: p.source });
    } else {
      carried.push(p);
    }
  }
  const carriedKeys = new Set(carried.map((c) => pairKey(c.a, c.b)));
  const asUnjudged = (ps: CandidatePair[]): JudgedPair[] =>
    ps.map((p) => ({ ...p, verdict: -1 as const }));

  // 1. Candidate collection (free), checkpointing after every page.
  const collected = await collectPairs({
    pool: deps.pool,
    projectId: deps.projectId,
    scanBatch: deps.scanBatch,
    topS: deps.topS,
    threshold: deps.threshold,
    resumeCursor: prior && !prior.config.complete ? prior.config.cursor : null,
    onPage: (cursor, pairsSoFar) =>
      writeReport(
        [
          ...carried,
          ...asUnjudged(mergePending(pairsSoFar, reentered, carriedKeys)),
        ],
        cursor,
        false,
        0,
      ),
  });
  const pending = mergePending(collected.pairs, reentered, carriedKeys);
  if (prior) {
    // Re-entered pairs may reference rows the fresh collection did not touch.
    await hydrateRowMeta(deps.pool, pending, collected.rowsById);
  }

  log(
    `consolidate: ${pending.length} candidate pair(s) to judge ` +
      `(threshold > ${deps.threshold ?? DUP_COSINE_THRESHOLD}) — ` +
      `est ~${pending.length * EST_TOKENS_PER_PAIR} judge tokens` +
      (apply ? " [APPLY]" : " [dry-run]"),
  );

  // 2. Judge (PAID — the injected judge; fail-open per batch), checkpointing after
  //    every batch so a crash mid-judging keeps every verdict bought so far.
  const freshJudged = await judgePairs(
    {
      judge: deps.judge,
      judgeBatch: deps.judgeBatch,
      log,
      onBatch: (judgedSoFar, remaining) =>
        writeReport(
          [...carried, ...judgedSoFar, ...asUnjudged(remaining)],
          collected.cursor,
          false,
          0,
        ),
    },
    pending,
    collected.rowsById,
  );
  const allPairs = [...carried, ...freshJudged];

  // 3. Apply (ONLY with --apply): dry-run performs ZERO row changes (AC-303).
  //    The fully-judged report lands on disk BEFORE the first row flips, so a crash
  //    mid-apply always leaves the undo log (pair list = exact flip-back targets).
  //    Carried pairs from a resumed run are included — re-applying is idempotent
  //    (the UPDATE's status/pinned/kind guards make a stale pair a no-op).
  let appliedCounts = { superseded: 0, markers: 0 };
  if (apply) {
    writeReport(allPairs, collected.cursor, true, 0);
    appliedCounts = await applyJudged(deps.pool, allPairs, log);
  }

  // 4. Final report artifact — review surface, undo log, resumability checkpoint.
  const report = writeReport(
    allPairs,
    collected.cursor,
    true,
    appliedCounts.superseded,
  );
  log(
    `consolidate: ${report.judged} judged, ${report.unjudged} UNJUDGED, ` +
      `${report.wouldSupersede} equivalent pair(s)` +
      (apply
        ? `, ${report.applied} row(s) superseded`
        : " (dry-run — zero rows changed)") +
      ` → ${reportPath}`,
  );
  return { report, reportPath };
}

/** Fetch title/content/created_at/pinned for pair ids the collection pass did not
 *  load (re-entered UNJUDGED pairs from a prior report). */
async function hydrateRowMeta(
  pool: ConsolidateDeps["pool"],
  pairs: CandidatePair[],
  rowsById: Map<string, PairRowMeta>,
): Promise<void> {
  const missing = [...new Set(pairs.flatMap((p) => [p.a, p.b]))].filter(
    (id) => !rowsById.has(id),
  );
  if (missing.length === 0) return;
  const { rows } = await pool.query(
    `SELECT id, title, content, summary, created_at, pinned
     FROM memory.memories WHERE id = ANY($1::uuid[])`,
    [missing],
  );
  for (const r of rows as unknown as Array<{
    id: string;
    title: string;
    content: string;
    summary: string | null;
    created_at: Date;
    pinned: boolean;
  }>) {
    registerRow(
      rowsById,
      r.id,
      r.title,
      r.content,
      r.summary,
      r.created_at,
      r.pinned,
    );
  }
}

// ── CLI decision table (exported pure — unit-tested offline) ─────────────────────────

export type CliAction =
  | { kind: "refuse"; message: string }
  | { kind: "applyReport"; report: ConsolidateReport; note: string }
  | { kind: "run"; apply: boolean; priorReport?: ConsolidateReport };

/**
 * Resolve what `npm run consolidate -- --yes [...]` should do, given today's report
 * state. The invariant behind every branch: `--apply` NEVER judge-and-applies fresh —
 * a human-reviewed dry-run report (same UTC day, or an explicit --report path) is
 * required, so no un-reviewed verdict can ever flip a row via the CLI.
 */
export function resolveCliAction(opts: {
  apply: boolean;
  existing: ConsolidateReport | null;
  explicitReport?: ConsolidateReport | null;
}): CliAction {
  const { apply, existing, explicitReport } = opts;

  if (explicitReport) {
    if (!apply) {
      return {
        kind: "refuse",
        message: "consolidate: --report is only valid together with --apply",
      };
    }
    return {
      kind: "applyReport",
      report: explicitReport,
      note: "applying explicitly-passed report",
    };
  }

  if (apply) {
    if (!existing) {
      return {
        kind: "refuse",
        message:
          "consolidate --apply: no same-day dry-run report found — run the dry-run first " +
          "(npm run consolidate -- --yes), review the report, then --apply. " +
          "Reports are keyed by UTC date; to apply a report from another day, pass " +
          "--report <path> explicitly.",
      };
    }
    if (existing.config.apply) {
      // A crashed (or finished) apply run: re-applying the same report is idempotent
      // (the DB guards no-op already-flipped pairs), so this is the crash-recovery path.
      return {
        kind: "applyReport",
        report: existing,
        note: "re-applying today's apply report (idempotent — crash recovery)",
      };
    }
    if (!existing.config.complete) {
      return {
        kind: "refuse",
        message:
          "consolidate --apply: today's report is an INCOMPLETE dry-run checkpoint — " +
          "finish the dry-run first (npm run consolidate -- --yes resumes it), review, " +
          "then --apply.",
      };
    }
    return {
      kind: "applyReport",
      report: existing,
      note: "applying reviewed dry-run report",
    };
  }

  // Dry-run.
  if (existing?.config.apply) {
    return {
      kind: "refuse",
      message:
        "consolidate: today's report is an APPLY record (the undo log for applied " +
        "supersessions) — a fresh dry-run would overwrite it. It is preserved as-is; " +
        "re-run tomorrow, or move the report aside deliberately first.",
    };
  }
  // Resume an interrupted run, or reuse a completed dry-run's bought verdicts
  // (carried pairs are never re-judged; only new/UNJUDGED pairs hit the judge).
  return { kind: "run", apply: false, priorReport: existing ?? undefined };
}

// ── CLI entrypoint (npm run consolidate -- --yes [--apply] [--report <path>]) ────────
// Skipped when imported (tests import the exported functions directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const argv = process.argv.slice(2);
  const refusal = guardConsolidateRun(argv);
  if (refusal) {
    console.error(refusal);
    process.exit(1);
  }
  const apply = argv.includes("--apply");
  const reportFlagIdx = argv.indexOf("--report");
  const reportFlagPath =
    reportFlagIdx >= 0 ? (argv[reportFlagIdx + 1] ?? null) : null;
  if (
    reportFlagIdx >= 0 &&
    (!reportFlagPath || reportFlagPath.startsWith("--"))
  ) {
    console.error("consolidate: --report requires a path argument");
    process.exit(1);
  }
  (async () => {
    const reportPath = defaultReportPath();
    const readReport = (p: string): ConsolidateReport =>
      JSON.parse(readFileSync(p, "utf8")) as ConsolidateReport;
    const existing = existsSync(reportPath) ? readReport(reportPath) : null;
    const explicitReport = reportFlagPath ? readReport(reportFlagPath) : null;

    const action = resolveCliAction({ apply, existing, explicitReport });
    if (action.kind === "refuse") {
      console.error(action.message);
      process.exitCode = 1;
      return;
    }

    if (action.kind === "applyReport") {
      console.error(
        `consolidate --apply: ${action.note} (${reportFlagPath ?? reportPath})`,
      );
      const res = await applyFromReport(
        defaultPool,
        action.report,
        undefined,
        config.defaultProjectId,
      );
      const applied: ConsolidateReport = {
        ...action.report,
        config: {
          ...action.report.config,
          apply: true,
          generatedAt: new Date().toISOString(),
        },
        applied: res.superseded,
      };
      const appliedPath = (reportFlagPath ?? reportPath).replace(
        /\.json$/,
        "-applied.json",
      );
      writeFileSync(appliedPath, JSON.stringify(applied, null, 2) + "\n");
      console.error(
        `consolidate --apply: ${res.superseded} row(s) superseded, ` +
          `${res.markers} judged-marker(s) → ${appliedPath}`,
      );
      await defaultPool.end();
      return;
    }

    await consolidate({
      pool: defaultPool,
      judge: (system, user) => judgeComplete(system, user),
      projectId: config.defaultProjectId,
      apply: action.apply,
      priorReport: action.priorReport,
    });
    await defaultPool.end();
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
