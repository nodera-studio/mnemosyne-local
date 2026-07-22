// Wave-2 Step 6 (AC-107): memory.search_log + fire-and-forget logging.
//
//   (a) a live searchMemory inserts ONE search_log row whose pool_ids/final_ids match
//       the fused pool and the returned hits (polled briefly — the insert is async by
//       design and must NOT be awaited by the search path);
//   (b) result parity is pinned by the golden suite (search-golden.test.ts) running in
//       the same `npm test` — logging must not perturb results;
//   (c) with the log table DROPPED mid-test, searchMemory still returns normally and
//       no rejection escapes (the fire-and-forget .catch swallows it).
//
// Voyage is fully mocked (PAID-API guard) with the same seeded-PRNG scheme as the
// golden/gate-sim suites; needs the disposable Postgres (describe.skipIf).

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

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
const PROJ = "search-log-mem";
const id = (n: number) => `40000000-0000-4000-8000-00000000000${n}`;

/** Poll until `probe` returns non-null (the insert is fire-and-forget). */
async function poll<T>(
  probe: () => Promise<T | null>,
  timeoutMs = 2000,
): Promise<T | null> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const got = await probe();
    if (got !== null || Date.now() > deadline) return got;
    await new Promise((r) => setTimeout(r, 25));
  }
}

describe.skipIf(skip)("search_log fire-and-forget (AC-107)", () => {
  let pool: pg.Pool;

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
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);
    const fixtures = [
      {
        id: id(1),
        title: "Redis eviction policy",
        content:
          "allkeys-lru evicts the least recently used key across the whole keyspace when maxmemory is hit",
      },
      {
        id: id(2),
        title: "BullMQ retry strategy",
        content:
          "queue jobs retry with exponential backoff; move to failed after the attempts budget",
      },
      {
        id: id(3),
        title: "Halfvec index note",
        content:
          "hnsw over halfvec(1024) keeps recall high while halving index memory",
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
  });

  afterAll(async () => {
    // Re-create the table for later suites (the drop test below removed it), then clean.
    const sql006 = readFileSync(
      join(here, "..", "sql", "006_search_log.sql"),
      "utf8",
    );
    await pool.query(sql006);
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
  });

  it("logs one row per search with matching pool/final id arrays (async, non-blocking)", async () => {
    const { fuseCandidates, searchMemory } = await import("../src/memory.js");

    const query = "redis lru eviction";
    // Expected pool from the same deterministic pipeline (fuseCandidates does NOT log).
    const expectedPool = (await fuseCandidates({ projectId: PROJ, query })).map(
      (c) => c.id,
    );

    const result = await searchMemory({ projectId: PROJ, query, limit: 2 });
    expect(result.hits.length).toBeGreaterThan(0);

    const row = await poll(async () => {
      const { rows } = await pool.query<{
        query: string;
        filters: Record<string, unknown>;
        pool_ids: string[];
        final_ids: string[];
        pool_ms: number | null;
        total_ms: number | null;
      }>(
        `SELECT query, filters, pool_ids, final_ids, pool_ms, total_ms
         FROM memory.search_log WHERE project_id = $1 AND query = $2`,
        [PROJ, query],
      );
      return rows[0] ?? null;
    });

    expect(row).not.toBeNull();
    expect(row!.pool_ids).toEqual(expectedPool);
    expect(row!.final_ids).toEqual(result.hits.map((h) => h.id));
    expect(row!.filters).toEqual({});
    expect(row!.pool_ms).not.toBeNull();
    expect(row!.total_ms).not.toBeNull();
    expect(row!.total_ms!).toBeGreaterThanOrEqual(row!.pool_ms!);
  });

  it("records the type filter in `filters`", async () => {
    const { searchMemory } = await import("../src/memory.js");
    const query = "bullmq retry backoff";
    await searchMemory({ projectId: PROJ, query, type: "semantic", limit: 2 });

    const row = await poll(async () => {
      const { rows } = await pool.query<{ filters: Record<string, unknown> }>(
        `SELECT filters FROM memory.search_log WHERE project_id = $1 AND query = $2`,
        [PROJ, query],
      );
      return rows[0] ?? null;
    });
    expect(row).not.toBeNull();
    expect(row!.filters).toEqual({ type: "semantic" });
  });

  it("search still returns normally when the log table is gone (failure swallowed)", async () => {
    const { searchMemory } = await import("../src/memory.js");
    await pool.query(`DROP TABLE memory.search_log`);

    const result = await searchMemory({
      projectId: PROJ,
      query: "hnsw halfvec index",
      limit: 2,
    });
    expect(result.hits.length).toBeGreaterThan(0);

    // Give the swallowed rejection a couple of ticks to settle INSIDE this test — an
    // unhandled rejection here would fail the vitest run, so surviving this wait IS
    // the AC-107 assertion.
    await new Promise((r) => setTimeout(r, 100));
  });
});
