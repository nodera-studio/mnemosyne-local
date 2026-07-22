// Minimal hand-rolled Anthropic Messages client for the PAID operator scripts
// (distill-eval now; consolidate in wave-5). Mirrors src/voyage.ts: raw fetch with
// retry on 429/5xx honoring Retry-After, no vendor SDK (repo convention).
//
// AC-108: this module is imported ONLY by operator scripts behind `npm run` +
// import.meta.url main-guards — never by server.ts, never in CI. The key is
// deliberately NOT in assertServerConfig; when it is absent, the failure message
// below tells the operator exactly what to set.

import { config } from "./config.js";

const BASE = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION = "2023-06-01";
const MAX_ATTEMPTS = 4;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** POST to the Anthropic API with retry on 429 / 5xx (honors Retry-After, else exp
 *  backoff + jitter) — the voyage.ts retry shape, different auth headers. */
export async function llmFetch(path: string, body: unknown): Promise<Response> {
  if (!config.anthropicApiKey) {
    throw new Error(
      "ANTHROPIC_API_KEY is not set — the PAID operator scripts (distill-eval, consolidate) " +
        "need it. Export it (optionally with DISTILL_MODEL / CONSOLIDATE_MODEL) and re-run. " +
        "The MCP server itself does not need this key (deliberately absent from assertServerConfig).",
    );
  }
  // `for (;;)` so every exit is explicit: the final (or non-retryable) attempt always
  // throws below — no unreachable post-loop throw.
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: {
        "x-api-key": config.anthropicApiKey,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (res.ok) return res;

    const retryable = res.status === 429 || res.status >= 500;
    if (retryable && attempt < MAX_ATTEMPTS - 1) {
      // Drain the failed body so the connection is released before the retry (an
      // unconsumed fetch body pins its socket); the content is irrelevant here.
      await res.text().catch(() => {});
      const retryAfter = Number(res.headers.get("retry-after"));
      const waitMs =
        retryAfter > 0
          ? retryAfter * 1000
          : Math.min(8000, 500 * 2 ** attempt) +
            Math.floor(Math.random() * 250);
      await sleep(waitMs);
      continue;
    }
    throw new Error(`anthropic ${path} ${res.status}: ${await res.text()}`);
  }
}

/**
 * One-shot judge/distill completion: system + user → concatenated text blocks.
 * Model defaults to `config.consolidateModel`; distill-eval passes
 * `config.distillModel` explicitly.
 */
export async function judgeComplete(
  system: string,
  user: string,
  opts: { model?: string; maxTokens?: number } = {},
): Promise<string> {
  const res = await llmFetch("/messages", {
    model: opts.model ?? config.consolidateModel,
    max_tokens: opts.maxTokens ?? 4096,
    system,
    messages: [{ role: "user", content: user }],
  });
  const json = (await res.json()) as {
    content?: Array<{ type: string; text?: string }>;
    stop_reason?: string;
  };
  const text = (json.content ?? [])
    .filter((b) => b.type === "text")
    .map((b) => b.text ?? "")
    .join("");
  if (!text) {
    throw new Error(
      `anthropic /messages returned no text (stop_reason: ${json.stop_reason ?? "unknown"})`,
    );
  }
  return text;
}
