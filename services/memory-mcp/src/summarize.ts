// Write-time dense summary helper (wave-2, AC-808/809). Produces the 1–3 sentence
// `memory.memories.summary` used as a reranker-doc prefix (see buildRerankDoc).
//
// Contract (AC-808): NEVER throws, NEVER blocks the write past the timeout, and the
// write path stays LLM-free by default. DOUBLE gate — the Anthropic key must be
// present AND SUMMARIZE_ON_STORE=1 — because compose passes ANTHROPIC_API_KEY through
// for operator scripts; the key alone must not start spending on every memory_store.
// When disabled it returns null immediately: zero imports, zero network.
//
// AC-809: src/llm.ts (the only in-service Anthropic client) is loaded via a LAZY
// `await import("./llm.js")` inside the function body — this module has NO top-level
// import of llm.js, so server.ts's static import graph never reaches it (AC-108).

import { config } from "./config.js";

export const SUMMARY_TIMEOUT_MS = 4000;
export const SUMMARY_MAX_TOKENS = 256;
/** Cost bound on the prompt — content beyond this is not sent. */
export const SUMMARY_INPUT_MAX_CHARS = 8000;

export const SUMMARY_SYSTEM =
  "Summarize this memory in 1-3 dense, factual sentences. Keep concrete identifiers (files, ids, numbers, names) verbatim. No preamble, no markdown, no quotes around the whole answer.";

export type SummaryJudge = (
  system: string,
  user: string,
  opts: { model?: string; maxTokens?: number },
) => Promise<string>;

/**
 * Summarize a memory's title+content into 1–3 dense sentences, or return null.
 * Null on: gate disabled, judge failure, timeout, or an empty/whitespace answer —
 * the caller writes `summary = NULL` and the store/update always succeeds.
 */
export async function summarizeMemory(
  title: string,
  content: string,
  deps: { judge?: SummaryJudge; timeoutMs?: number; enabled?: boolean } = {},
): Promise<string | null> {
  const enabled =
    deps.enabled ?? (config.anthropicApiKey !== "" && config.summarizeOnStore);
  if (!enabled) return null;
  const timeoutMs = deps.timeoutMs ?? SUMMARY_TIMEOUT_MS;
  try {
    const judge =
      deps.judge ??
      // Lazy dynamic import (AC-809): llm.js enters the module graph only when the
      // gate is open — never via server.ts's static imports.
      (await import("./llm.js")).judgeComplete;
    const user = `${title}\n\n${content.slice(0, SUMMARY_INPUT_MAX_CHARS)}`;
    let timer: NodeJS.Timeout | undefined;
    // Promise.race ABANDONS (does not abort) the losing judge call on timeout; that
    // is acceptable — llmFetch's own retry ceiling bounds the dangling request.
    const result = await Promise.race([
      judge(SUMMARY_SYSTEM, user, {
        model: config.consolidateModel,
        maxTokens: SUMMARY_MAX_TOKENS,
      }),
      new Promise<null>((resolve) => {
        timer = setTimeout(() => resolve(null), timeoutMs);
      }),
    ]).finally(() => clearTimeout(timer));
    if (result === null) return null;
    const trimmed = result.trim();
    return trimmed === "" ? null : trimmed;
  } catch {
    return null; // AC-808: a summarizer failure must never fail the write
  }
}
