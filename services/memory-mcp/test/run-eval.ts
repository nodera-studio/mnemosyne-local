// PAID operator eval-run recorder — `npm run eval:record -- --yes [--split test|dev]
// [--label <name>]` (wave-5 Step 5 support; the flip-gate mechanism for TEST-split
// artifacts, e.g. <date>-pre-consolidation.json / <date>-post-consolidation.json).
//
// Runs the wave-2 two-layer runner (`runRecallEval`) on the chosen split through the
// LIVE pipeline and writes a run artifact under test/runs/. Every split query is
// embedded through live Voyage — PAID (AC-108): main-guarded, refuses without --yes,
// never imported by server.ts, never in CI. Zero LLM/judge cost.
//
// SPLIT POLICY (AC-109): --split test spends the FROZEN test split — sanctioned ONLY
// at flip gates (embedder flips, rerank swaps, consolidation acceptance per
// test/consolidation.md). Dev-split recording normally goes through the gate test
// (EVAL_RECORD=1, see test/eval.md); --split dev here is an escape hatch only.
//
// This file lives under test/ (NOT src/) because it composes test/recall.helper.ts and
// tsconfig rootDir="src" excludes test/ — tsx runs it fine; tsconfig.test.json
// typechecks it.

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  runRecallEval,
  tryLoadRecallEvalV2,
  writeRunArtifact,
} from "./recall.helper.js";
import { config } from "../src/config.js";

export const EVAL_RECORD_COST_NOTE =
  "eval:record is a PAID operator script: it embeds every query of the chosen split " +
  "through live Voyage (the flip-gate run). Nothing was run. Re-run with an explicit " +
  "consent flag:  npm run eval:record -- --yes [--split test|dev] [--label <name>]";

export interface RunEvalArgs {
  yes: boolean;
  split: "dev" | "test";
  label: string;
}

/** Parse CLI args; returns the refusal message instead when --yes is absent. */
export function parseRunEvalArgs(argv: string[]): RunEvalArgs | string {
  if (!argv.includes("--yes")) return EVAL_RECORD_COST_NOTE;
  const opt = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    const v = i !== -1 ? argv[i + 1] : undefined;
    // A flag-shaped "value" means the real value was omitted (`--label --split test`);
    // the label also becomes part of the artifact filename, so never swallow a flag.
    return v !== undefined && v.startsWith("--") ? undefined : v;
  };
  const split = opt("split") ?? "test";
  if (split !== "dev" && split !== "test") {
    return `eval:record: invalid --split "${split}" — use "dev" or "test"`;
  }
  return { yes: true, split, label: opt("label") ?? split };
}

// ── CLI entrypoint (npm run eval:record -- --yes) ─────────────────────────────────────
// Skipped when imported (tests import parseRunEvalArgs directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const args = parseRunEvalArgs(process.argv.slice(2));
  if (typeof args === "string") {
    console.error(args);
    process.exit(1);
  }
  (async () => {
    const evalFile = tryLoadRecallEvalV2();
    if (!evalFile) {
      throw new Error(
        "eval:record: test/fixtures/recall-eval.json is not v2 — migrate/approve gold first",
      );
    }
    const artifact = await runRecallEval(evalFile, {
      projectId: config.defaultProjectId,
      split: args.split,
    });
    const here = dirname(fileURLToPath(import.meta.url));
    const path = writeRunArtifact(join(here, "runs"), args.label, artifact);
    console.error(
      `eval:record — ${artifact.rows} ${args.split}-split rows → ${path}`,
    );
    console.error(JSON.stringify(artifact.aggregates, null, 2));
    const { pool } = await import("../src/db/pool.js");
    await pool.end();
  })().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
