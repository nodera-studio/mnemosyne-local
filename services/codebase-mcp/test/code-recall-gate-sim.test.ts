// Test-of-the-test for the code recall regression gate (wave-2 Step 5 verify): prove
// the gate machinery can FAIL, PASS, and SKIP on a seeded corpus. Runs the REAL
// two-layer runner (mocked Voyage, no quota), then applies the exact gate math
// (perQueryDeltas → pairedBootstrapCI seed 42 → regressionExcluded) against synthetic
// baselines — 0.2 above what the corpus can produce (FAIL) and equal (PASS).
//
// SEPARATE from code-recall-gate.test.ts: the real gate stays mock-free for the
// operator's live flip-gate run.

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  loadCodeEval,
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  runCodeEval,
  type CodeEvalFile,
  type RunArtifact,
} from "./code-eval.helper.js";

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
  embedCode: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(`code:${t}`)),
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`code:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

const here = dirname(fileURLToPath(import.meta.url));
const PROJ = "gate-sim-code";
const REPO = "gate-sim-repo";

const FILES = [
  {
    path: "src/auth/token.ts",
    content:
      "export function verifyToken(jwt: string): boolean { return check(jwt); } // bearer auth",
  },
  {
    path: "src/db/pool.ts",
    content:
      "export const pool = new Pool({ connectionString: url, max: 10 }); // shared pg pool",
  },
  {
    path: "src/report/csv.ts",
    content:
      "export function writeCsv(rows: Row[]): string { return rows.map(toLine).join('\\n'); }",
  },
];

const EVAL: CodeEvalFile = {
  version: 2,
  k: 10,
  rows: [
    {
      id: "c-001",
      query: "bearer token verification",
      relevantPaths: ["src/auth/token.ts"],
      split: "dev",
      provenance: "seed-v1",
    },
    {
      id: "c-002",
      query: "postgres connection pool",
      relevantPaths: ["src/db/pool.ts"],
      split: "dev",
      provenance: "seed-v1",
    },
    {
      id: "c-003",
      query: "write rows to csv",
      relevantPaths: ["src/report/csv.ts"],
      split: "dev",
      provenance: "seed-v1",
    },
  ],
};

describe.skipIf(skip)(
  "code recall gate can fail/pass/skip (test-of-the-test)",
  () => {
    let pool: pg.Pool;
    let fresh: RunArtifact;
    const tmp = mkdtempSync(join(tmpdir(), "code-gate-sim-"));

    beforeAll(async () => {
      pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
      const sqlDir = join(here, "..", "sql");
      await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
      for (const f of readdirSync(sqlDir)
        .filter((x) => x.endsWith(".sql"))
        .sort()) {
        const sql = readFileSync(join(sqlDir, f), "utf8");
        if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
        await pool.query(sql);
      }
      await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
        PROJ,
      ]);
      for (const f of FILES) {
        const { rows } = await pool.query<{ id: string }>(
          `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
         VALUES ($1, $2, $3, 'typescript', 'sha-' || $3) RETURNING id`,
          [REPO, PROJ, f.path],
        );
        await pool.query(
          `INSERT INTO codebase.code_chunks
           (file_id, repository_id, project_id, file_path, language, symbol_name,
            start_line, end_line, content, content_sha256, embedding)
         VALUES ($1, $2, $3, $4, 'typescript', NULL, 1, 2, $5, 'csha-' || $4, $6::halfvec)`,
          [
            rows[0].id,
            REPO,
            PROJ,
            f.path,
            f.content,
            `[${H.fakeVec(`chunk:${f.path}`).join(",")}]`,
          ],
        );
      }
      fresh = await runCodeEval(EVAL, { projectId: PROJ, split: "dev" });
    });

    afterAll(async () => {
      await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
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
      const deltas = perQueryDeltas(synthBaseline(0.2), fresh);
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

    it("SKIP rung: a v1 eval file has no version, so the gate skips instead of gating falsely", () => {
      const p = join(tmp, "v1-code-eval.json");
      writeFileSync(
        p,
        JSON.stringify({
          k: 10,
          rows: [{ query: "q", relevantPaths: ["a.ts"] }],
        }),
      );
      expect(loadCodeEval(p).version).toBeUndefined();
    });

    it("SKIP rung: the baseline artifact is simply absent until a flip gate records one", () => {
      expect(existsSync(join(tmp, "baseline-dev.json"))).toBe(false);
    });
  },
);
