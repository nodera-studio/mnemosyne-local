// Wave-2 Step 3 verify: the two-layer v2 runner against a seeded corpus.
//
//  - AC-106: gold row A is SUPERSEDED by B (A.superseded_by = B, status superseded).
//    Search can only ever return B — the runner must credit A's gold via the forward
//    chain at BOTH layers (poolRecall = 1, ndcg reflects B's final rank).
//  - AC-104: the run artifact carries the full retrievalConfig() snapshot + per-query
//    scores for both layers.
//
// Deterministic and network-free: Voyage is module-mocked with seeded-PRNG vectors
// (golden-pin pattern); DB-backed on the disposable :5544 Postgres, skipped without
// DATABASE_URL. NO live Voyage quota is ever burned here.

import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  perQueryDeltas,
  resolveGoldIds,
  runRecallEval,
  writeRunArtifact,
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
const PROJ = "runner-mem";
const id = (n: number) => `20000000-0000-4000-8000-00000000000${n}`;
const A = id(1); // superseded gold row
const B = id(2); // A's successor (what search actually returns)
const C = id(3);
const D = id(4);

describe.skipIf(skip)("runRecallEval v2 (two layers + supersession)", () => {
  let pool: pg.Pool;
  const tmp = mkdtempSync(join(tmpdir(), "recall-runner-"));

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Apply the migration chain idempotently (HOLD files skipped) — golden-pin pattern.
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
        id: A,
        title: "Old pool sizing decision",
        content:
          "postgres pool sizing: max 5 per service (superseded by the newer decision)",
        status: "superseded",
      },
      {
        id: B,
        title: "Pool sizing decision v2",
        content:
          "postgres connection pool sizing: keep max at 10 per service and tune idle timeout",
        status: "active",
      },
      {
        id: C,
        title: "Deploy rollback runbook",
        content:
          "rollback procedure for a bad deployment: redeploy the previous image tag and verify health",
        status: "active",
      },
      {
        id: D,
        title: "HNSW index tuning",
        content:
          "hnsw ef_search trades recall for latency; rebuild concurrently after mass re-embeds",
        status: "active",
      },
    ];
    for (const f of fixtures) {
      await pool.query(
        `INSERT INTO memory.memories
           (id, project_id, type, title, content, importance, status, embedding_v2)
         VALUES ($1, $2, 'semantic'::memory.memory_type, $3, $4, 0.5, $5, $6::halfvec)`,
        [
          f.id,
          PROJ,
          f.title,
          f.content,
          f.status,
          `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
        ],
      );
    }
    // A was consolidated into B: forward pointer set, status superseded (done above).
    await pool.query(
      `UPDATE memory.memories SET superseded_by = $1 WHERE id = $2`,
      [B, A],
    );
  });

  afterAll(async () => {
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
    rmSync(tmp, { recursive: true, force: true });
  });

  it("resolveGoldIds walks the forward chain (and self-resolves unknown ids)", async () => {
    const ghost = "99999999-9999-4999-8999-999999999999";
    const chains = await resolveGoldIds(pool, [A, ghost]);
    expect(chains.get(A)).toEqual(new Set([A, B]));
    // an id absent from the DB still counts for itself
    expect(chains.get(ghost)).toEqual(new Set([ghost]));
  });

  it("resolveGoldIds terminates on a malformed supersession cycle", async () => {
    // temporary cycle: B -> A (A -> B already set)
    await pool.query(
      `UPDATE memory.memories SET superseded_by = $1 WHERE id = $2`,
      [A, B],
    );
    try {
      const chains = await resolveGoldIds(pool, [A]);
      expect(chains.get(A)).toEqual(new Set([A, B]));
    } finally {
      await pool.query(
        `UPDATE memory.memories SET superseded_by = NULL WHERE id = $1`,
        [B],
      );
    }
  });

  it("scores both layers, credits the successor for superseded gold, writes the artifact (AC-104/AC-106)", async () => {
    const evalFile: RecallEvalFileV2 = {
      version: 2,
      k: 10,
      changelog: ["test fixture"],
      rows: [
        {
          id: "m-001",
          query: "postgres connection pool sizing",
          relevantIds: [A], // the SUPERSEDED id — search can only return B
          split: "dev",
          provenance: "seed-v1",
        },
        {
          id: "m-002",
          query: "rollback a bad deployment",
          relevantIds: [C],
          split: "test", // must be EXCLUDED from a dev-split run
          provenance: "seed-v1",
        },
      ],
    };

    const result = await runRecallEval(evalFile, {
      projectId: PROJ,
      split: "dev",
      pool,
    });

    // split filter: only the dev row ran
    expect(result.split).toBe("dev");
    expect(result.rows).toBe(1);
    expect(result.perQuery.map((q) => q.id)).toEqual(["m-001"]);

    const q = result.perQuery[0];
    // POOL layer: B is in the candidate pool and counts for A via the forward chain.
    expect(q.poolRecall).toBe(1);
    // FINAL layer: B's rank credits A — ndcg is exactly the single-relevant discount.
    expect(q.rank).not.toBeNull();
    expect(q.ndcg).toBeCloseTo(1 / Math.log2(q.rank! + 1), 10);
    expect(q.mrr).toBeCloseTo(1 / q.rank!, 10);
    expect(result.aggregates.recallAt25).toBe(1);

    // artifact: full retrievalConfig() snapshot + per-query scores, written to disk
    const path = writeRunArtifact(tmp, "runner-verify", result);
    const parsed = JSON.parse(readFileSync(path, "utf8")) as RunArtifact;
    const { retrievalConfig } = await import("../src/memory.js");
    expect(parsed.retrievalConfig).toEqual(retrievalConfig());
    expect(parsed.evalVersion).toBe(2);
    expect(parsed.perQuery).toEqual(result.perQuery);
    expect(parsed.aggregates).toEqual(result.aggregates);
  });
});

// ── Pure gate-join math (no DB) ──────────────────────────────────────────────────────

function artifact(perQuery: Array<{ id: string; ndcg: number }>): RunArtifact {
  return {
    retrievalConfig: {},
    evalVersion: 2,
    split: "dev",
    rows: perQuery.length,
    perQuery: perQuery.map((q) => ({
      id: q.id,
      query: q.id,
      poolRecall: 1,
      ndcg: q.ndcg,
      mrr: q.ndcg,
      rank: 1,
    })),
    aggregates: { recallAt25: 1, ndcgAt10: 0, mrr10: 0 },
  };
}

describe("perQueryDeltas (gate join)", () => {
  it("joins by row id (id-sorted) and returns fresh − baseline", () => {
    const base = artifact([
      { id: "m-002", ndcg: 0.5 },
      { id: "m-001", ndcg: 1.0 },
    ]);
    const fresh = artifact([
      { id: "m-001", ndcg: 0.8 },
      { id: "m-002", ndcg: 0.7 },
    ]);
    const deltas = perQueryDeltas(base, fresh);
    expect(deltas.length).toBe(2);
    expect(deltas[0]).toBeCloseTo(-0.2, 10); // m-001
    expect(deltas[1]).toBeCloseTo(0.2, 10); // m-002
  });

  it("fails LOUDLY on row-id drift (a changed dev split invalidates the baseline)", () => {
    const base = artifact([{ id: "m-001", ndcg: 1 }]);
    const fresh = artifact([{ id: "m-777", ndcg: 1 }]);
    expect(() => perQueryDeltas(base, fresh)).toThrow(/eval-split drift/);
  });
});
