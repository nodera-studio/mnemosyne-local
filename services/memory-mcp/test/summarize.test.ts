// Wave-2 — write-time summary helper + reranker-doc composition. Pure/zero-network
// (injectable-deps style of backfill-context.test.ts):
//
//   AC-808 — summarizeMemory is DOUBLE-gated (key AND SUMMARIZE_ON_STORE=1), returns
//            null on any failure/timeout/empty answer, and NEVER throws — the write
//            path stays LLM-free by default and never fails on a summarizer error.
//   AC-809 — src/llm.ts stays out of server.ts's STATIC import graph: summarize.ts
//            uses a lazy `await import("./llm.js")` only (text-level conformance).
//   AC-810 — buildRerankDoc = title\n(summary\n)content truncated AS A WHOLE to
//            RERANK_DOC_TRUNCATION; summary=NULL docs are byte-identical to the
//            pre-wave-2 form.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  SUMMARY_INPUT_MAX_CHARS,
  SUMMARY_MAX_TOKENS,
  SUMMARY_SYSTEM,
  SUMMARY_TIMEOUT_MS,
  summarizeMemory,
  type SummaryJudge,
} from "../src/summarize.js";
import { RERANK_DOC_TRUNCATION, buildRerankDoc } from "../src/memory.js";

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = join(here, "..", "src");

describe("summarizeMemory (AC-808, zero network)", () => {
  it("exports the locked constants", () => {
    expect(SUMMARY_TIMEOUT_MS).toBe(4000);
    expect(SUMMARY_MAX_TOKENS).toBe(256);
    expect(SUMMARY_INPUT_MAX_CHARS).toBe(8000);
  });

  it("enabled:false → null and the judge is NEVER called", async () => {
    const judge = vi.fn<SummaryJudge>();
    const out = await summarizeMemory("t", "c", { judge, enabled: false });
    expect(out).toBeNull();
    expect(judge).not.toHaveBeenCalled();
  });

  it("default gate (SUMMARIZE_ON_STORE unset in tests) → null, judge never called", async () => {
    // The double gate resolves from config: the flag is off in the test env, so
    // even a provided judge must never fire (write path stays LLM-free by default).
    const judge = vi.fn<SummaryJudge>();
    const out = await summarizeMemory("t", "c", { judge });
    expect(out).toBeNull();
    expect(judge).not.toHaveBeenCalled();
  });

  it("happy path: passes system/user/model opts and trims whitespace", async () => {
    const judge = vi.fn<SummaryJudge>(async () => "  a dense summary \n");
    const out = await summarizeMemory("Title X", "Body Y", {
      judge,
      enabled: true,
    });
    expect(out).toBe("a dense summary");
    expect(judge).toHaveBeenCalledTimes(1);
    const [system, user, opts] = judge.mock.calls[0];
    expect(system).toBe(SUMMARY_SYSTEM);
    expect(user).toBe("Title X\n\nBody Y");
    expect(opts.maxTokens).toBe(SUMMARY_MAX_TOKENS);
  });

  it("caps the prompt at SUMMARY_INPUT_MAX_CHARS of content", async () => {
    const judge = vi.fn<SummaryJudge>(async () => "s");
    const big = "x".repeat(SUMMARY_INPUT_MAX_CHARS + 5000);
    await summarizeMemory("T", big, { judge, enabled: true });
    const user = judge.mock.calls[0][1];
    expect(user.length).toBe("T\n\n".length + SUMMARY_INPUT_MAX_CHARS);
  });

  it("judge throws → null (never propagates)", async () => {
    const judge = vi.fn<SummaryJudge>(async () => {
      throw new Error("boom");
    });
    await expect(
      summarizeMemory("t", "c", { judge, enabled: true }),
    ).resolves.toBeNull();
  });

  it("judge hangs past timeoutMs → null (real timers)", async () => {
    const judge = vi.fn<SummaryJudge>(
      () => new Promise<string>(() => {}), // never settles — abandoned by the race
    );
    const out = await summarizeMemory("t", "c", {
      judge,
      enabled: true,
      timeoutMs: 50,
    });
    expect(out).toBeNull();
  });

  it("empty / whitespace-only answer → null", async () => {
    for (const answer of ["", "   ", "\n\t "]) {
      const judge = vi.fn<SummaryJudge>(async () => answer);
      expect(
        await summarizeMemory("t", "c", { judge, enabled: true }),
      ).toBeNull();
    }
  });
});

describe("AC-809: llm.ts stays out of the server's static import graph (pure)", () => {
  const staticLlmImport = /^import .* from ["'].*\/llm\.js["']/m;

  it.each(["memory.ts", "summarize.ts", "server.ts"])(
    "src/%s has no static llm.js import",
    (file) => {
      const text = readFileSync(join(srcDir, file), "utf8");
      expect(staticLlmImport.test(text)).toBe(false);
    },
  );

  it("summarize.ts reaches llm.js via the lazy dynamic form only", () => {
    const text = readFileSync(join(srcDir, "summarize.ts"), "utf8");
    expect(text).toContain('await import("./llm.js")');
  });
});

describe("buildRerankDoc (AC-810, pure)", () => {
  it("summary=null → byte-identical to the pre-wave-2 title\\ncontent doc", () => {
    const title = "A title";
    const content = "some content ".repeat(200); // > 1200 chars joined
    const legacy = `${title}\n${content}`.slice(0, RERANK_DOC_TRUNCATION);
    expect(buildRerankDoc({ title, content, summary: null })).toBe(legacy);
    // summary omitted entirely behaves the same
    expect(buildRerankDoc({ title, content })).toBe(legacy);
  });

  it("with summary → title\\nsummary\\ncontent prefix, capped as a WHOLE", () => {
    const title = "A title";
    const summary = "A dense summary.";
    const content = "short body";
    const doc = buildRerankDoc({ title, content, summary });
    expect(doc).toBe(`${title}\n${summary}\n${content}`);
    expect(doc.length).toBeLessThanOrEqual(RERANK_DOC_TRUNCATION);
  });

  it("long content: the summary displaces content tail within the same budget", () => {
    const title = "T";
    const summary = "S".repeat(100);
    const content = "c".repeat(5000);
    const doc = buildRerankDoc({ title, content, summary });
    expect(doc.length).toBe(RERANK_DOC_TRUNCATION);
    // Prefix shape holds — never summary-only: the joined string is sliced, so the
    // doc still starts title\nsummary\n and ends inside the content.
    expect(doc.startsWith(`${title}\n${summary}\nc`)).toBe(true);
    expect(doc).toBe(
      `${title}\n${summary}\n${content}`.slice(0, RERANK_DOC_TRUNCATION),
    );
  });
});
