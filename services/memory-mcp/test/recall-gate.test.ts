// AC-105: the CI regression gate for memory retrieval, rewritten for the wave-2 eval
// spine. The old scalar mechanism (`V1_RECALL_AT_10_BASELINE`, recall@10 ≥ a recorded
// number) is RETIRED — the gate now recomputes the DEV split and joins it per-query
// against the COMMITTED baseline artifact `test/runs/baseline-dev.json` (a copy of a
// run artifact, chosen by the operator at a flip gate and refreshed only there):
//
//   deltas[i] = fresh.ndcg − baseline.ndcg   (joined BY ROW ID; drift fails loudly)
//   FAIL when pairedBootstrapCI(deltas, {seed: 42}) excludes zero on the regression
//   side ("no significant regression + directional win" — at N≈75 only deltas ≥ ~0.05
//   are detectable; that is by design, see test/eval.md). Also FAIL when the pool
//   layer's mean Recall@25 drops > 0.05 absolute (the reranker cannot fix recall).
//
// The gate recomputes the DEV split ONLY — the frozen test split is spent exclusively
// by flip gates (p-hacking guard, AC-109). Bootstrap seed fixed for reproducibility.
//
// Graceful-skip ladder (each rung prints the operator action):
//   1. DATABASE_URL unset                        → suite skipped
//   2. no committed test/runs/baseline-dev.json  → skip (record one at a flip gate)
//   3. gold file still v1 (not id-keyed)         → skip (gold:migrate → approve → rename)
//   4. embedding_v2 unpopulated for the project  → skip (run backfill:context first)
//
// The operator ALSO runs this exact test against the LIVE DB — that IS the flip-gate
// run (it embeds every dev query, so it burns Voyage quota there; CI always lands on a
// skip rung instead). Recording a fresh artifact: EVAL_RECORD=1 writes
// test/runs/<date>-dev.json before gating (copy it to baseline-dev.json to freeze).

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import pg from "pg";
import {
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  runRecallEval,
  tryLoadRecallEvalV2,
  writeRunArtifact,
  type RunArtifact,
} from "./recall.helper.js";
import { config } from "../src/config.js";

const HAS_DB = !!process.env.DATABASE_URL;
const RECORD = process.env.EVAL_RECORD === "1";

const here = dirname(fileURLToPath(import.meta.url));
const runsDir = join(here, "runs");
const BASELINE_PATH = join(runsDir, "baseline-dev.json");

/** Coarse pool-layer floor: mean Recall@25 must not drop more than this, absolute. */
const POOL_RECALL_MAX_DROP = 0.05;

async function v2PopulatedCount(
  pool: pg.Pool,
  projectId: string,
): Promise<number> {
  const { rows } = await pool.query<{ n: string }>(
    `SELECT count(*) AS n FROM memory.memories
     WHERE project_id = $1 AND archived_at IS NULL
       AND COALESCE(status,'active')='active' AND embedding_v2 IS NOT NULL`,
    [projectId],
  );
  return Number(rows[0].n);
}

describe.skipIf(!HAS_DB)("recall regression gate — dev split (AC-105)", () => {
  // Live flip-gate runs embed every dev query through Voyage (~1s/query) — far past
  // vitest's 5s default; mocked/disposable-DB runs finish in ms either way.
  it(
    "per-query nDCG@10 deltas vs baseline-dev.json show no significant regression",
    { timeout: 900_000 },
    async () => {
      // Rung 2: no committed baseline (and not a recording run).
      if (!existsSync(BASELINE_PATH) && !RECORD) {
        console.error(
          `recall-gate: no ${BASELINE_PATH} — skipping (record one with EVAL_RECORD=1 at a flip gate)`,
        );
        expect(existsSync(BASELINE_PATH)).toBe(false); // structural assertion
        return;
      }

      // Rung 3: gold not yet migrated to stable-id v2.
      const evalFile = tryLoadRecallEvalV2();
      if (!evalFile) {
        console.error(
          "recall-gate: gold file still v1 — skipping (npm run gold:migrate, approve, rename — AC-109)",
        );
        expect(evalFile).toBeNull();
        return;
      }

      const pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 2,
      });
      try {
        const projectId = config.defaultProjectId;

        // Rung 4: pre-backfill corpus (the disposable-CI state).
        const populated = await v2PopulatedCount(pool, projectId);
        if (populated === 0) {
          console.error(
            "recall-gate: embedding_v2 unpopulated — skipping (run after backfill:context)",
          );
          expect(populated).toBe(0);
          return;
        }

        // Fresh dev-split run through the LIVE two-layer pipeline (embeds each query —
        // Voyage quota on the operator's live run; CI never reaches this rung).
        const fresh = await runRecallEval(evalFile, {
          projectId,
          split: "dev",
          pool,
        });

        if (RECORD) {
          const p = writeRunArtifact(runsDir, "dev", fresh);
          console.error(
            `recall-gate: recorded ${p} — copy to baseline-dev.json (commit it) to freeze the baseline`,
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
});
