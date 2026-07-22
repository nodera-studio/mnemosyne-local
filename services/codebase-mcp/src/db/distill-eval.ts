// PAID, operator-gated eval-candidate distiller — codebase twin of memory-mcp's
// src/db/distill-eval.ts (wave-2 Step 7, AC-108/AC-109). Per-service duplication is
// the repo convention.
//
// Samples code chunks stratified by (language, top-level dir), asks the model for
// realistic CODE-SEARCH queries per archetype, dedupes, and writes
// test/fixtures/eval-candidates-<date>.json for HUMAN review. Suggested gold entries
// are FILE PATHS (the stable code gold id — chunk ids churn on reindex). Every
// candidate is `approved: false`; gold files change only via human-approved merges.
//
// PAID gate (AC-108): runs ONLY via `npm run distill-eval -- --yes` — main-guarded,
// never imported by server.ts, never in CI, refuses without `--yes`. Zero LLM calls
// in tests (deps are injected).

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pool as defaultPool } from "./pool.js";
import { judgeComplete } from "../llm.js";
import { config } from "../config.js";

/** Target archetype mix (documented in the prompt). */
export const ARCHETYPES = [
  "conceptual-where-how",
  "exact-symbol",
  "usage-pattern",
  "error-or-bug-shaped",
  "filter-shaped",
] as const;

export const COST_NOTE =
  "distill-eval is a PAID operator script: it sends sampled chunk rows to the Anthropic API " +
  `(model: ${config.distillModel}) and spends quota. Nothing was run. ` +
  "Re-run with an explicit consent flag:  npm run distill-eval -- --yes";

/** Returns the refusal message when the paid-consent flag is absent, else null. */
export function guardPaidRun(argv: string[]): string | null {
  return argv.includes("--yes") ? null : COST_NOTE;
}

export interface EvalCandidate {
  query: string;
  /** Gold FILE-PATH hints — the human verifies each before it enters the eval file. */
  suggestedGold: string[];
  archetype: string;
  provenance: "distilled";
  approved: false;
}

export interface SampledChunk {
  file_path: string;
  language: string | null;
  symbol_name: string | null;
  excerpt: string;
}

export interface DistillDeps {
  pool: {
    query: (
      text: string,
      params: unknown[],
    ) => Promise<{ rows: SampledChunk[] }>;
  };
  /** Injected LLM completion (tests mock this — zero live calls). */
  complete: (system: string, user: string) => Promise<string>;
  projectId?: string;
  /** Max sampled chunks per (language, top-level dir) stratum. */
  perStratum?: number;
  /** Output file override (tests point this at a tmp dir). */
  outPath?: string;
  log?: (msg: string) => void;
}

const here = dirname(fileURLToPath(import.meta.url));
const DEFAULT_OUT_DIR = join(here, "..", "..", "test", "fixtures");

/** Stratified corpus sample: up to `perStratum` random chunks per
 *  (language, top-level dir) stratum, so one repo area cannot dominate the prompt. */
export async function sampleCorpus(
  pool: DistillDeps["pool"],
  projectId: string,
  perStratum: number,
): Promise<SampledChunk[]> {
  const { rows } = await pool.query(
    `SELECT file_path, language, symbol_name, excerpt FROM (
       SELECT file_path, language, symbol_name, left(content, 400) AS excerpt,
              row_number() OVER (
                PARTITION BY COALESCE(language, ''), split_part(file_path, '/', 1)
                ORDER BY random()
              ) AS rn
       FROM codebase.code_chunks
       WHERE project_id = $1
     ) s
     WHERE rn <= $2
     ORDER BY language NULLS FIRST, file_path`,
    [projectId, perStratum],
  );
  return rows;
}

const SYSTEM_PROMPT = `You generate EVAL QUERY CANDIDATES for a code-search system.
Given sampled code chunks (path | language | symbol | excerpt), propose realistic search
queries a developer would type to find them. Cover ALL of these archetypes, spreading the
candidates across them: ${ARCHETYPES.join(", ")}.

Rules:
- Each candidate must plausibly be answered by one or more of the sampled files.
- suggestedGold lists FILE PATHS (from the sample) the query should retrieve — hints only.
- Conceptual queries must not quote symbol names verbatim; exact-symbol queries may.
- Respond with ONLY a JSON array, no prose, no code fences:
  [{"query": "...", "suggestedGold": ["<file path>"], "archetype": "<one of the archetypes>"}]`;

/** Parse the model output into candidate rows (same tolerance as the memory twin). */
export function parseCandidates(raw: string): EvalCandidate[] {
  const start = raw.indexOf("[");
  const end = raw.lastIndexOf("]");
  if (start === -1 || end <= start) {
    throw new Error(
      `distill-eval: model output has no JSON array: ${raw.slice(0, 200)}`,
    );
  }
  const arr = JSON.parse(raw.slice(start, end + 1)) as Array<{
    query?: unknown;
    suggestedGold?: unknown;
    archetype?: unknown;
  }>;
  const out: EvalCandidate[] = [];
  for (const c of arr) {
    if (typeof c.query !== "string" || c.query.trim() === "") continue;
    out.push({
      query: c.query.trim(),
      suggestedGold: Array.isArray(c.suggestedGold)
        ? c.suggestedGold.filter((g): g is string => typeof g === "string")
        : [],
      archetype: typeof c.archetype === "string" ? c.archetype : "unknown",
      provenance: "distilled",
      approved: false,
    });
  }
  return out;
}

/** Dedupe on normalized query text (first occurrence wins). */
export function dedupeCandidates(cands: EvalCandidate[]): EvalCandidate[] {
  const seen = new Set<string>();
  return cands.filter((c) => {
    const key = c.query.trim().toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export const APPROVAL_INSTRUCTIONS =
  "Review each candidate: verify the query is realistic, verify EVERY suggestedGold path " +
  "actually answers it (fix/trim as needed), then merge approved rows into " +
  "test/fixtures/code-eval.json v2 with split:'dev', provenance:'distilled' and your " +
  "approvedBy — and append a changelog line. The test split changes only at flip gates (AC-109).";

export async function distillEval(
  deps: DistillDeps,
): Promise<{ candidates: EvalCandidate[]; path: string }> {
  const log = deps.log ?? ((m: string) => console.error(m));
  const projectId = deps.projectId ?? config.defaultProjectId;
  const perStratum = deps.perStratum ?? 8;

  const sample = await sampleCorpus(deps.pool, projectId, perStratum);
  if (sample.length === 0) {
    throw new Error(
      `distill-eval: no indexed chunks for project "${projectId}" — nothing to distill`,
    );
  }
  log(`distill-eval: sampled ${sample.length} chunks (≤${perStratum}/stratum)`);

  const user = sample
    .map(
      (c) =>
        `${c.file_path} | ${c.language ?? "-"} | ${c.symbol_name ?? "-"} | ${c.excerpt.replace(/\s+/g, " ")}`,
    )
    .join("\n");
  const raw = await deps.complete(SYSTEM_PROMPT, user);
  const candidates = dedupeCandidates(parseCandidates(raw));
  log(`distill-eval: ${candidates.length} deduped candidates`);

  const date = new Date().toISOString().slice(0, 10);
  const path =
    deps.outPath ?? join(DEFAULT_OUT_DIR, `eval-candidates-${date}.json`);
  writeFileSync(
    path,
    JSON.stringify(
      {
        _service: "codebase-mcp",
        _generated: date,
        _note: APPROVAL_INSTRUCTIONS,
        candidates,
      },
      null,
      2,
    ) + "\n",
  );
  return { candidates, path };
}

// ── CLI entrypoint (npm run distill-eval -- --yes) ────────────────────────────────────
// Skipped when imported (tests import the exported functions directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const refusal = guardPaidRun(process.argv.slice(2));
  if (refusal) {
    console.error(refusal);
    process.exit(1);
  }
  (async () => {
    const { candidates, path } = await distillEval({
      pool: defaultPool,
      complete: (system, user) =>
        judgeComplete(system, user, { model: config.distillModel }),
    });
    console.error(
      `distill-eval: wrote ${candidates.length} candidates → ${path}`,
    );
    console.error(`NEXT (human): ${APPROVAL_INSTRUCTIONS}`);
    await defaultPool.end();
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
