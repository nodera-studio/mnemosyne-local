// AC-105: the CI regression gate for code retrieval (created in wave-2 — the codebase
// twin of memory-mcp's test/recall-gate.test.ts). It recomputes the DEV split of
// test/fixtures/code-eval.json through the live two-layer pipeline and joins it
// per-query against the COMMITTED baseline artifact `test/runs/baseline-dev.json`
// (chosen by the operator at a flip gate, refreshed only there):
//
//   deltas[i] = fresh.ndcg − baseline.ndcg   (joined BY ROW ID; drift fails loudly)
//   FAIL when pairedBootstrapCI(deltas, {seed: 42}) excludes zero on the regression
//   side ("no significant regression + directional win" — at N≈75 only deltas ≥ ~0.05
//   are detectable; by design, see memory-mcp/test/eval.md). Also FAIL when the pool
//   layer's mean Recall@25 drops > 0.05 absolute (the reranker cannot fix recall).
//
// DEV split ONLY — the frozen test split is spent exclusively by flip gates (AC-109).
//
// Graceful-skip ladder (each rung prints the operator action):
//   1. DATABASE_URL unset                        → suite skipped
//   2. no committed test/runs/baseline-dev.json  → skip (record one at a flip gate)
//   3. eval file not yet v2 (no ids/splits)      → skip
//   4. no indexed chunks for the project         → skip (index the corpus first)
//
// The operator ALSO runs this exact test against the LIVE DB — that IS the flip-gate
// run (it embeds every dev query, burning Voyage quota there; CI always lands on a
// skip rung). EVAL_RECORD=1 writes test/runs/<date>-dev.json before gating (copy it to
// baseline-dev.json to freeze).

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import pg from "pg";
import {
  loadCodeEval,
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  runCodeEval,
  writeRunArtifact,
  type RunArtifact,
} from "./code-eval.helper.js";
import { config } from "../src/config.js";

const HAS_DB = !!process.env.DATABASE_URL;
const RECORD = process.env.EVAL_RECORD === "1";

const here = dirname(fileURLToPath(import.meta.url));
const runsDir = join(here, "runs");
const BASELINE_PATH = join(runsDir, "baseline-dev.json");

/** Coarse pool-layer floor: mean Recall@25 must not drop more than this, absolute. */
const POOL_RECALL_MAX_DROP = 0.05;

describe.skipIf(!HAS_DB)(
  "code recall regression gate — dev split (AC-105)",
  () => {
    // Live flip-gate runs embed every dev query through Voyage (~1s/query) — far past
    // vitest's 5s default; mocked/disposable-DB runs finish in ms either way.
    it(
      "per-query nDCG@10 deltas vs baseline-dev.json show no significant regression",
      { timeout: 900_000 },
      async () => {
        // Rung 2: no committed baseline (and not a recording run).
        if (!existsSync(BASELINE_PATH) && !RECORD) {
          console.error(
            `code-recall-gate: no ${BASELINE_PATH} — skipping (record one with EVAL_RECORD=1 at a flip gate)`,
          );
          expect(existsSync(BASELINE_PATH)).toBe(false); // structural assertion
          return;
        }

        // Rung 3: eval file not yet v2.
        const evalFile = loadCodeEval();
        if (evalFile.version !== 2) {
          console.error(
            "code-recall-gate: code-eval.json is not v2 — skipping (add ids/splits first)",
          );
          expect(evalFile.version).toBeUndefined();
          return;
        }

        const pool = new pg.Pool({
          connectionString: process.env.DATABASE_URL,
          max: 2,
        });
        try {
          const projectId = config.defaultProjectId;

          // Rung 4: unindexed corpus (the disposable-CI state).
          const { rows } = await pool.query<{ n: string }>(
            `SELECT count(*) AS n FROM codebase.code_chunks WHERE project_id = $1`,
            [projectId],
          );
          const populated = Number(rows[0].n);
          if (populated === 0) {
            console.error(
              "code-recall-gate: no indexed chunks for the project — skipping (index the corpus first)",
            );
            expect(populated).toBe(0);
            return;
          }

          // Fresh dev-split run through the LIVE two-layer pipeline (embeds each query —
          // Voyage quota on the operator's live run; CI never reaches this rung).
          const fresh = await runCodeEval(evalFile, {
            projectId,
            repo: process.env.CODE_RECALL_REPO, // optional repo scope, as in code-recall.test.ts
            split: "dev",
          });

          if (RECORD) {
            const p = writeRunArtifact(runsDir, "dev", fresh);
            console.error(
              `code-recall-gate: recorded ${p} — copy to baseline-dev.json (commit it) to freeze the baseline`,
            );
            if (!existsSync(BASELINE_PATH)) return; // record-only run, nothing to gate yet
          }

          const baseline = JSON.parse(
            readFileSync(BASELINE_PATH, "utf8"),
          ) as RunArtifact;

          const deltas = perQueryDeltas(baseline, fresh); // throws loudly on split drift
          const ci = pairedBootstrapCI(deltas, { seed: 42 });
          expect(
            regressionExcluded(ci),
            `nDCG@10 regressed: mean delta ${ci.mean.toFixed(4)}, 95% CI [${ci.ciLow.toFixed(4)}, ${ci.ciHigh.toFixed(4)}] excludes zero on the regression side`,
          ).toBe(false);

          // Coarse recall floor at the pool layer — the reranker cannot fix recall.
          expect(
            fresh.aggregates.recallAt25,
            `pool-layer mean Recall@25 dropped > ${POOL_RECALL_MAX_DROP} absolute vs baseline`,
          ).toBeGreaterThanOrEqual(
            baseline.aggregates.recallAt25 - POOL_RECALL_MAX_DROP,
          );
        } finally {
          await pool.end();
        }
      },
    );
  },
);
