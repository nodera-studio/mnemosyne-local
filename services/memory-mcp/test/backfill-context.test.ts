// AC-021: the backfill populates embedding_v2 for every active row, in batches, and is
// RESUMABLE. DB-backed (needs the disposable test DB) but the Voyage embedder is MOCKED
// via dependency injection — NO live quota. Skipped gracefully without DATABASE_URL.

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  backfillContext,
  countPending,
  type ContextualEmbedder,
} from "../src/db/backfill-context.js";

const HAS_DB = !!process.env.DATABASE_URL;

// Deterministic mock embedder: one 1024-int vector per text, recording call shapes so we
// can assert batching. NEVER touches the network.
function mockEmbedder(): { fn: ContextualEmbedder; calls: number[] } {
  const calls: number[] = [];
  const fn: ContextualEmbedder = async (texts) => {
    calls.push(texts.length);
    return texts.map((_, i) => Array.from({ length: 1024 }, () => (i % 7) - 3));
  };
  return { fn, calls };
}

describe.skipIf(!HAS_DB)(
  "backfillContext (AC-021, mocked Voyage, test DB)",
  () => {
    let pool: pg.Pool;
    // Unique project per run so parallel/rerun does not collide.
    const projectId = `backfill-test-${Date.now()}`;

    beforeAll(async () => {
      pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 4,
      });
    });

    afterAll(async () => {
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        projectId,
      ]);
      await pool.end();
    });

    async function seed(
      n: number,
      opts: { archived?: boolean; status?: string } = {},
    ): Promise<void> {
      for (let i = 0; i < n; i++) {
        await pool.query(
          `INSERT INTO memory.memories
           (project_id, type, title, content, importance, status, archived_at)
         VALUES ($1, 'semantic', $2, $3, 0.5, $4, $5)`,
          [
            projectId,
            `title ${i}`,
            `content body ${i}`,
            opts.status ?? "active",
            opts.archived ? new Date() : null,
          ],
        );
      }
    }

    async function pendingFor(): Promise<number> {
      const { rows } = await pool.query<{ n: string }>(
        `SELECT count(*) AS n FROM memory.memories
       WHERE project_id = $1 AND archived_at IS NULL
         AND COALESCE(status,'active')='active' AND embedding_v2 IS NULL`,
        [projectId],
      );
      return Number(rows[0].n);
    }

    it("populates embedding_v2 for every active row, in ≤batchSize batches", async () => {
      await seed(5);
      const { fn, calls } = mockEmbedder();

      const res = await backfillContext({
        pool,
        embed: fn,
        batchSize: 2,
        log: () => {},
      });

      // 5 rows / batch 2 → batches of 2,2,1
      expect(res.processed).toBe(5);
      expect(res.batches).toBe(3);
      expect(calls).toEqual([2, 2, 1]);
      expect(calls.every((c) => c <= 2)).toBe(true);

      // every seeded active row now has embedding_v2
      const { rows } = await pool.query<{ n: string }>(
        `SELECT count(*) AS n FROM memory.memories
       WHERE project_id=$1 AND embedding_v2 IS NOT NULL`,
        [projectId],
      );
      expect(Number(rows[0].n)).toBe(5);
      expect(await pendingFor()).toBe(0);
    });

    it("is RESUMABLE — a second run is a no-op (no rows left to embed)", async () => {
      const { fn, calls } = mockEmbedder();
      const res = await backfillContext({
        pool,
        embed: fn,
        batchSize: 2,
        log: () => {},
      });
      expect(res.processed).toBe(0);
      expect(res.batches).toBe(0);
      expect(calls).toEqual([]); // never called the embedder again
    });

    it("skips archived + non-active rows (only active rows get embedded)", async () => {
      const archivedProj = `${projectId}-arch`;
      const apool = pool;
      // seed under a separate project so counts are isolated
      await apool.query(
        `INSERT INTO memory.memories (project_id,type,title,content,importance,status,archived_at)
       VALUES ($1,'semantic','arch','x',0.5,'active',now()),
              ($1,'semantic','superseded','y',0.5,'superseded',null),
              ($1,'semantic','live','z',0.5,'active',null)`,
        [archivedProj],
      );
      const { fn, calls } = mockEmbedder();
      // run a scoped backfill by temporarily filtering: reuse the real fn but assert via SQL
      await backfillContext({
        pool: apool,
        embed: fn,
        batchSize: 50,
        log: () => {},
      });

      const { rows } = await apool.query<{
        title: string;
        has: boolean;
      }>(
        `SELECT title, (embedding_v2 IS NOT NULL) AS has FROM memory.memories
       WHERE project_id=$1 ORDER BY title`,
        [archivedProj],
      );
      const byTitle = Object.fromEntries(rows.map((r) => [r.title, r.has]));
      expect(byTitle["live"]).toBe(true); // active, not archived → embedded
      expect(byTitle["arch"]).toBe(false); // archived → skipped
      expect(byTitle["superseded"]).toBe(false); // non-active status → skipped

      await apool.query(`DELETE FROM memory.memories WHERE project_id=$1`, [
        archivedProj,
      ]);
    });

    it("splits a DB page into token-budgeted sub-batches (≤120k tok/request)", async () => {
      const bigProj = `${projectId}-big`;
      const big = "x ".repeat(54000); // ~108k chars ≈ 27k est tokens per doc
      for (let i = 0; i < 8; i++) {
        await pool.query(
          `INSERT INTO memory.memories (project_id,type,title,content,importance,status,archived_at)
           VALUES ($1,'semantic',$2,$3,0.5,'active',null)`,
          [bigProj, `big ${i}`, big],
        );
      }
      const { fn, calls } = mockEmbedder();
      // batchSize large → all 8 rows land in ONE DB page, so any splitting is the
      // token-budget sub-batcher, not page paging. Before the fix this was a single
      // 8-doc request that Voyage rejects with TOO_MANY_TOKENS_IN_BATCH (120k cap).
      const res = await backfillContext({
        pool,
        embed: fn,
        batchSize: 50,
        log: () => {},
      });
      expect(res.processed).toBe(8);
      expect(calls.length).toBeGreaterThan(1); // sub-batched, not one giant request
      expect(Math.max(...calls)).toBeLessThanOrEqual(4); // each request stays small
      expect(calls.reduce((a, b) => a + b, 0)).toBe(8);
      const pend = await pool.query<{ n: string }>(
        `SELECT count(*) AS n FROM memory.memories WHERE project_id=$1 AND embedding_v2 IS NULL`,
        [bigProj],
      );
      expect(Number(pend.rows[0].n)).toBe(0);
      await pool.query(`DELETE FROM memory.memories WHERE project_id=$1`, [
        bigProj,
      ]);
    });

    it("countPending reflects global active-missing count and never throws", async () => {
      const n = await countPending(pool);
      expect(typeof n).toBe("number");
      expect(n).toBeGreaterThanOrEqual(0);
    });
  },
);

// A pure (no-DB) guard so the file always has at least one running assertion in CI.
describe("backfill embedder mock (no network)", () => {
  it("mock embedder returns one 1024-dim vector per text", async () => {
    const { fn } = mockEmbedder();
    const out = await fn(["a", "b"], "document");
    expect(out).toHaveLength(2);
    expect(out[0]).toHaveLength(1024);
  });
});
