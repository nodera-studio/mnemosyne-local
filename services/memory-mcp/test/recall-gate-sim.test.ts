// Test-of-the-test for the recall regression gate (wave-2 Step 5 verify): prove the
// gate machinery can actually FAIL, PASS, and SKIP — a gate that cannot fail is not a
// gate. Runs the REAL two-layer runner over a seeded corpus (mocked Voyage, no quota),
// then applies the exact gate math (perQueryDeltas → pairedBootstrapCI seed 42 →
// regressionExcluded) against synthetic baseline artifacts:
//
//   - baseline scores 0.2 ABOVE what the corpus can produce → regression detected (FAIL)
//   - baseline equal to the fresh run                       → no regression (PASS)
//   - v1 gold file / missing baseline                       → the gate's skip rungs
//
// This file mocks ../src/voyage.js, so it is SEPARATE from recall-gate.test.ts — the
// real gate must stay mock-free for the operator's live flip-gate run.

import {
  mkdtempSync,
  rmSync,
  writeFileSync,
  existsSync,
  readFileSync,
  readdirSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  runRecallEval,
  tryLoadRecallEvalV2,
  type RecallEvalFileV2,
  type RunArtifact,
} from "./recall.helper.js";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const H = vi.hoisted(() => {
  function fnv1a(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }
  function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function fakeVec(key: string, dim = 1024): number[] {
    const rnd = mulberry32(fnv1a(key));
    return Array.from({ length: dim }, () =>
      Number((rnd() * 2 - 1).toFixed(3)),
    );
  }
  function fakeRerank(
    query: string,
    docs: string[],
    topK: number,
  ): Array<{ index: number; score: number }> {
    const scored = docs.map((d, i) => ({
      index: i,
      score: Number(
        (0.05 + 0.9 * mulberry32(fnv1a(`${query}|${d.length}|${i}`))()).toFixed(
          6,
        ),
      ),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.min(topK, docs.length));
  }
  return { fakeVec, fakeRerank };
});

vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map((t) => H.fakeVec(`legacy:${t}`)),
  embedContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(t))),
  embedContextualSingle: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(t)),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

const here = dirname(fileURLToPath(import.meta.url));
const PROJ = "gate-sim-mem";
const id = (n: number) => `30000000-0000-4000-8000-00000000000${n}`;

const EVAL: RecallEvalFileV2 = {
  version: 2,
  k: 10,
  changelog: ["gate-sim fixture"],
  rows: [
    {
      id: "m-001",
      query: "postgres connection pool sizing",
      relevantIds: [id(1)],
      split: "dev",
      provenance: "seed-v1",
    },
    {
      id: "m-002",
      query: "rollback a bad deployment",
      relevantIds: [id(2)],
      split: "dev",
      provenance: "seed-v1",
    },
    {
      id: "m-003",
      query: "hnsw index tuning recall latency",
      relevantIds: [id(3)],
      split: "dev",
      provenance: "seed-v1",
    },
  ],
};

describe.skipIf(skip)(
  "recall gate can fail/pass/skip (test-of-the-test)",
  () => {
    let pool: pg.Pool;
    let fresh: RunArtifact;
    const tmp = mkdtempSync(join(tmpdir(), "recall-gate-sim-"));

    beforeAll(async () => {
      pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
      const sqlDir = join(here, "..", "sql");
      for (const f of readdirSync(sqlDir)
        .filter((x) => x.endsWith(".sql"))
        .sort()) {
        const sql = readFileSync(join(sqlDir, f), "utf8");
        if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
        await pool.query(sql);
      }
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        PROJ,
      ]);
      const fixtures = [
        {
          id: id(1),
          title: "Postgres pool sizing",
          content:
            "postgres connection pool sizing guidance: keep max at 10 per service and tune idle timeout",
        },
        {
          id: id(2),
          title: "Deploy rollback runbook",
          content:
            "rollback procedure for a bad deployment: redeploy the previous image tag and verify health",
        },
        {
          id: id(3),
          title: "HNSW index tuning",
          content:
            "hnsw ef_search trades recall for latency; rebuild concurrently after mass re-embeds",
        },
        {
          id: id(4),
          title: "RRF fusion notes",
          content:
            "hybrid search fuses bm25 and vector ranks with reciprocal rank fusion so neither arm dominates",
        },
      ];
      for (const f of fixtures) {
        await pool.query(
          `INSERT INTO memory.memories
           (id, project_id, type, title, content, importance, embedding_v2)
         VALUES ($1, $2, 'semantic'::memory.memory_type, $3, $4, 0.5, $5::halfvec)`,
          [
            f.id,
            PROJ,
            f.title,
            f.content,
            `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
          ],
        );
      }
      // ONE fresh run through the real two-layer runner; all three cases join against it.
      fresh = await runRecallEval(EVAL, {
        projectId: PROJ,
        split: "dev",
        pool,
      });
    });

    afterAll(async () => {
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        PROJ,
      ]);
      await pool.end();
      rmSync(tmp, { recursive: true, force: true });
    });

    function synthBaseline(bump: number): RunArtifact {
      return {
        ...fresh,
        perQuery: fresh.perQuery.map((q) => ({ ...q, ndcg: q.ndcg + bump })),
        aggregates: {
          ...fresh.aggregates,
          ndcgAt10: fresh.aggregates.ndcgAt10 + bump,
        },
      };
    }

    it("FAILS against a baseline 0.2 above what the corpus can produce", () => {
      const baseline = synthBaseline(0.2);
      const deltas = perQueryDeltas(baseline, fresh);
      for (const d of deltas) expect(d).toBeCloseTo(-0.2, 10);
      const ci = pairedBootstrapCI(deltas, { seed: 42 });
      expect(ci.ciHigh).toBeLessThan(0);
      expect(regressionExcluded(ci)).toBe(true); // ← the gate assertion flips red
    });

    it("PASSES against an equal baseline (and the pool floor holds)", () => {
      const baseline = synthBaseline(0);
      const deltas = perQueryDeltas(baseline, fresh);
      const ci = pairedBootstrapCI(deltas, { seed: 42 });
      expect(regressionExcluded(ci)).toBe(false);
      expect(fresh.aggregates.recallAt25).toBeGreaterThanOrEqual(
        baseline.aggregates.recallAt25 - 0.05,
      );
    });

    it("gate math is reproducible: seed 42 twice → identical CI", () => {
      const deltas = perQueryDeltas(synthBaseline(0.2), fresh);
      expect(pairedBootstrapCI(deltas, { seed: 42 })).toEqual(
        pairedBootstrapCI(deltas, { seed: 42 }),
      );
    });

    it("SKIP rung: a v1 gold file is rejected by the v2 loader (gate skips, never gates falsely)", () => {
      const p = join(tmp, "v1-gold.json");
      writeFileSync(
        p,
        JSON.stringify({
          k: 10,
          rows: [{ query: "q", relevantTitles: ["t"] }],
        }),
      );
      expect(tryLoadRecallEvalV2(p)).toBeNull();
    });

    it("SKIP rung: the baseline artifact is simply absent until a flip gate records one", () => {
      expect(existsSync(join(tmp, "baseline-dev.json"))).toBe(false);
    });
  },
);
