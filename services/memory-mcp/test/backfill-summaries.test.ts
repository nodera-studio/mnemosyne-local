// Wave-2 — PAID resumable summary backfill (AC-811). DB-backed (disposable :5544)
// with the summarizer MOCKED via dependency injection — zero LLM calls, zero network.
// Skipped gracefully without DATABASE_URL. Pure guard tests always run.
//
//   AC-811 — refuses without --yes (cost note, zero LLM calls); with --yes it pages
//            active `summary IS NULL` rows by KEYSET (resumable), covers importer
//            raw-INSERT rows, and a per-row summarizer failure skips that row
//            without aborting the run (the cursor still advances → terminates).

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import {
  BACKFILL_SUMMARY_TIMEOUT_MS,
  COST_NOTE,
  SUMMARY_BACKFILL_BATCH,
  backfillSummaries,
  countPendingSummaries,
  guardPaidRun,
  type RowSummarizer,
} from "../src/db/backfill-summaries.js";

const HAS_DB = !!process.env.DATABASE_URL;
const here = dirname(fileURLToPath(import.meta.url));

describe("guardPaidRun (AC-811 consent gate, pure)", () => {
  it("refuses without --yes, naming the consent command and the model cost", () => {
    const refusal = guardPaidRun([]);
    expect(refusal).toBe(COST_NOTE);
    expect(refusal).toContain("backfill:summaries -- --yes");
    expect(refusal).toContain("PAID");
    expect(refusal).toMatch(/tokens per row/);
  });

  it("passes with --yes", () => {
    expect(guardPaidRun(["--yes"])).toBeNull();
  });

  it("one LLM call per row → small pages", () => {
    expect(SUMMARY_BACKFILL_BATCH).toBe(50);
  });

  it("the CLI summarizer uses the batch 30 s timeout, not the interactive 4 s bound", () => {
    // The 4 s SUMMARY_TIMEOUT_MS protects memory_store latency; a PAID offline row
    // must wait out a slow completion instead of billing and skipping it.
    expect(BACKFILL_SUMMARY_TIMEOUT_MS).toBe(30_000);
    const src = readFileSync(
      join(here, "..", "src", "db", "backfill-summaries.ts"),
      "utf8",
    );
    expect(src).toMatch(/timeoutMs:\s*BACKFILL_SUMMARY_TIMEOUT_MS/);
  });
});

describe.skipIf(!HAS_DB)(
  "backfillSummaries (AC-811, mocked summarizer, test DB)",
  () => {
    let pool: pg.Pool;
    // Unique project per run so rerun/parallel never collides.
    const projectId = `bfsum-test-${Date.now()}`;

    beforeAll(async () => {
      pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 4,
      });
      // Idempotent migration apply (HOLD files skipped) — needs 007's summary column.
      const sqlDir = join(here, "..", "sql");
      for (const f of readdirSync(sqlDir)
        .filter((x) => x.endsWith(".sql"))
        .sort()) {
        const sql = readFileSync(join(sqlDir, f), "utf8");
        if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
        await pool.query(sql);
      }
    });

    afterAll(async () => {
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        projectId,
      ]);
      await pool.end();
    });

    /** Importer-style raw INSERT — deliberately bypasses storeMemory (AC-811's
     *  "covers importer rows" clause: no embedding, no summary, no LLM). */
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
            `bfsum title ${i}`,
            `bfsum content body ${i}`,
            opts.status ?? "active",
            opts.archived ? new Date() : null,
          ],
        );
      }
    }

    async function summariesFor(): Promise<(string | null)[]> {
      const { rows } = await pool.query<{ summary: string | null }>(
        `SELECT summary FROM memory.memories WHERE project_id = $1 ORDER BY title`,
        [projectId],
      );
      return rows.map((r) => r.summary);
    }

    it("keyset-pages pending rows; a null (failing) row is SKIPPED, not fatal", async () => {
      await seed(5);
      // Archived + non-active rows must never be visited.
      await seed(1, { archived: true });
      await seed(1, { status: "superseded" });

      let calls = 0;
      const summarize: RowSummarizer = async (title) => {
        calls += 1;
        // The 3rd VISITED row persistently fails → null → skipped, cursor advances.
        if (calls === 3) return null;
        return `mock summary of ${title}`;
      };

      const res = await backfillSummaries({
        pool,
        summarize,
        batchSize: 2,
        log: () => {},
      });

      // 5 active pending rows / batch 2 → keyset pages of 2,2,1.
      expect(res).toEqual({ processed: 4, skipped: 1, batches: 3 });
      expect(calls).toBe(5); // archived + superseded rows were never visited

      const sums = await summariesFor();
      expect(sums.filter((s) => s !== null)).toHaveLength(4);
      expect(sums.filter((s) => s === null)).toHaveLength(3); // failed + archived + superseded
      for (const s of sums) {
        if (s !== null) expect(s).toMatch(/^mock summary of bfsum title \d$/);
      }
    });

    it("is RESUMABLE — the second run visits ONLY the remaining NULL row", async () => {
      let calls = 0;
      const summarize: RowSummarizer = async (title) => {
        calls += 1;
        return `retry summary of ${title}`;
      };
      const res = await backfillSummaries({
        pool,
        summarize,
        batchSize: 2,
        log: () => {},
      });
      expect(calls).toBe(1); // rows already summarized are naturally skipped
      expect(res).toEqual({ processed: 1, skipped: 0, batches: 1 });
      // Project-scoped pending probe (countPendingSummaries is global by design).
      const { rows } = await pool.query<{ n: string }>(
        `SELECT count(*) AS n FROM memory.memories
         WHERE project_id = $1 AND archived_at IS NULL
           AND COALESCE(status,'active') = 'active' AND summary IS NULL`,
        [projectId],
      );
      expect(Number(rows[0].n)).toBe(0);
    });

    it("a THROWING injected summarizer still only skips the row (never aborts)", async () => {
      await seed(2); // the only pending rows left for this project
      let calls = 0;
      const summarize: RowSummarizer = async () => {
        calls += 1;
        if (calls === 1) throw new Error("summarizer exploded");
        return "recovered";
      };
      const res = await backfillSummaries({
        pool,
        summarize,
        batchSize: 50,
        log: () => {},
      });
      expect(res.processed).toBe(1);
      expect(res.skipped).toBe(1);
    });

    it("countPendingSummaries counts active NULL-summary rows and never throws", async () => {
      const n = await countPendingSummaries(pool);
      expect(typeof n).toBe("number");
      expect(n).toBeGreaterThanOrEqual(0);
    });

    it("visits a row bearing the ALL-ZERO uuid (NULL-seeded keyset floor is inclusive)", async () => {
      // Regression pin: a strict `id > '00000000-…-0'` floor never visits this row
      // while countPendingSummaries still counts it → permanent nonzero exit.
      const ZERO_ID = "00000000-0000-0000-0000-000000000000";
      await pool.query(`DELETE FROM memory.memories WHERE id = $1`, [ZERO_ID]);
      await pool.query(
        `INSERT INTO memory.memories (id, project_id, type, title, content, importance)
         VALUES ($1, $2, 'semantic', 'bfsum zero-floor row', 'zero uuid body', 0.5)`,
        [ZERO_ID, projectId],
      );
      const summarize: RowSummarizer = async (title) =>
        `zero-floor summary of ${title}`;
      const res = await backfillSummaries({
        pool,
        summarize,
        batchSize: 2,
        log: () => {},
      });
      // Pending here: the zero-id row + the row the throwing summarizer skipped in
      // the previous test — both are summarized in one page.
      expect(res).toEqual({ processed: 2, skipped: 0, batches: 1 });
      const { rows } = await pool.query<{ summary: string | null }>(
        `SELECT summary FROM memory.memories WHERE id = $1`,
        [ZERO_ID],
      );
      expect(rows[0].summary).toBe(
        "zero-floor summary of bfsum zero-floor row",
      );
    });
  },
);

// Zero-network paging logic guard (no DB): the keyset loop terminates on an
// all-failing corpus because the cursor advances past skipped rows.
describe("backfillSummaries keyset termination (mock pool, no DB)", () => {
  it("an all-null summarizer terminates with everything skipped", async () => {
    const rows = Array.from({ length: 5 }, (_, i) => ({
      // uuid-sortable fake ids — keyset compares as text in the mock below. The
      // FIRST id is the ALL-ZERO uuid: the NULL-seeded first page must include it
      // (a strict `id > floor` cursor would never visit it).
      id: `00000000-0000-0000-0000-00000000000${i}`,
      title: `t${i}`,
      content: `c${i}`,
    }));
    const queries: string[] = [];
    const mockPool = {
      query: vi.fn(async (text: string, params?: unknown[]) => {
        queries.push(text);
        if (text.includes("SELECT id, title, content")) {
          const [cursor, limit] = params as [string | null, number];
          const page = rows
            .filter((r) => cursor === null || r.id > cursor)
            .slice(0, limit);
          return { rows: page };
        }
        throw new Error(`unexpected write: ${text}`);
      }),
    } as unknown as pg.Pool;

    const visited: string[] = [];
    const res = await backfillSummaries({
      pool: mockPool,
      summarize: async (title) => {
        visited.push(title);
        return null;
      },
      batchSize: 2,
      log: () => {},
    });
    // 5 rows in pages of 2,2,1 — all skipped, ZERO UPDATE statements issued —
    // and the all-zero-uuid row (t0) WAS visited.
    expect(res).toEqual({ processed: 0, skipped: 5, batches: 3 });
    expect(visited).toEqual(["t0", "t1", "t2", "t3", "t4"]);
    expect(queries.some((q) => q.includes("UPDATE"))).toBe(false);
  });
});
