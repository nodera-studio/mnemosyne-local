// Wave-2 — `summary` threading through storeMemory / updateMemory / getMemory
// (AC-808 write-side + the AC-805 memory_get carry). DB-backed (disposable :5544).
// The summarizer is vi.mock'ed at the MODULE level (memory.ts imports it statically)
// and the Voyage embedder is a stubbed fetch (decision-log.test.ts pattern) — zero
// live quota, zero LLM calls.

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import pg from "pg";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

vi.mock("../src/summarize.js", () => ({ summarizeMemory: vi.fn() }));

import { summarizeMemory } from "../src/summarize.js";
import * as mem from "../src/memory.js";

const mockSummarize = vi.mocked(summarizeMemory);

const HAS_DB = !!process.env.DATABASE_URL;
const here = dirname(fileURLToPath(import.meta.url));

// ── Voyage fetch stub (decision-log.test.ts pattern) ─────────────────────────────────
function fakeFlatResponse(n: number, dim = 1024) {
  return {
    data: Array.from({ length: n }, () => ({
      embedding: Array.from({ length: dim }, () => 1),
    })),
  };
}
function fakeContextualResponse(n: number, dim = 1024) {
  return {
    data: Array.from({ length: n }, (_, docIdx) => ({
      index: docIdx,
      data: [{ index: 0, embedding: Array.from({ length: dim }, () => 1) }],
    })),
  };
}
function mockVoyageFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: { body: string }) => {
      const body = JSON.parse(init.body);
      let response: unknown;
      if (url.includes("/contextualizedembeddings")) {
        response = fakeContextualResponse(body.inputs.length);
      } else if (url.includes("/embeddings")) {
        response = fakeFlatResponse(body.input.length);
      } else {
        response = {};
      }
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        json: async () => response,
        text: async () => JSON.stringify(response),
      } as unknown as Response;
    }),
  );
}

describe.skipIf(!HAS_DB)(
  "summary threading: store/update/get (wave-2, test DB)",
  () => {
    let pool: pg.Pool;
    const projectId = `store-sum-test-${Date.now()}`;

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

    beforeEach(() => {
      mockVoyageFetch();
      mockSummarize.mockReset();
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    async function summaryOf(id: string): Promise<string | null> {
      const { rows } = await pool.query<{ summary: string | null }>(
        `SELECT summary FROM memory.memories WHERE id = $1`,
        [id],
      );
      return rows[0].summary;
    }

    it("storeMemory persists the summarizer's non-null result; memory_get carries it", async () => {
      mockSummarize.mockResolvedValue("a summary");
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "with summary",
        content: "content one",
      });
      expect(mockSummarize).toHaveBeenCalledWith("with summary", "content one");
      expect(await summaryOf(id)).toBe("a summary");
      // AC-805 carry: the budgeted memory_get row surfaces the stored summary.
      const row = await mem.getMemory(id);
      expect(row?.summary).toBe("a summary");
    });

    it("storeMemory writes summary = NULL when the summarizer returns null (gate closed / failure)", async () => {
      mockSummarize.mockResolvedValue(null);
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "no summary",
        content: "content two",
      });
      expect(await summaryOf(id)).toBeNull();
      const row = await mem.getMemory(id);
      expect(row?.summary).toBeNull();
    });

    it("the write still resolves when the summarizer settles null after a delay (AC-808)", async () => {
      // A rejecting summarizer is impossible by contract; the failure mode is a
      // DELAYED null (internal timeout) — the store must simply await and succeed.
      mockSummarize.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(null), 40)),
      );
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "slow summarizer",
        content: "content three",
      });
      expect(await summaryOf(id)).toBeNull();
    });

    it("updateMemory recomputes the summary on a content change", async () => {
      mockSummarize.mockResolvedValue("original summary");
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "recompute me",
        content: "old body",
      });
      expect(await summaryOf(id)).toBe("original summary");

      mockSummarize.mockReset();
      mockSummarize.mockResolvedValue("recomputed summary");
      await mem.updateMemory(id, { content: "new body" });
      // Called with the MERGED title/content (existing title + new content).
      expect(mockSummarize).toHaveBeenCalledWith("recompute me", "new body");
      expect(await summaryOf(id)).toBe("recomputed summary");
    });

    it("a null recompute CLEARS the stale summary (never-stale invariant)", async () => {
      mockSummarize.mockResolvedValue("stale summary");
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "goes stale",
        content: "body v1",
      });
      expect(await summaryOf(id)).toBe("stale summary");

      mockSummarize.mockReset();
      mockSummarize.mockResolvedValue(null);
      await mem.updateMemory(id, { content: "body v2" });
      expect(await summaryOf(id)).toBeNull(); // regenerated by the next backfill run
    });

    it("a metadata-only update never touches the summary (no recompute)", async () => {
      mockSummarize.mockResolvedValue("kept summary");
      const { id } = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "kept",
        content: "kept body",
      });
      mockSummarize.mockReset();
      mockSummarize.mockResolvedValue(null);
      await mem.updateMemory(id, { importance: 0.9 });
      expect(mockSummarize).not.toHaveBeenCalled();
      expect(await summaryOf(id)).toBe("kept summary");
    });
  },
);

// Always-running guard so the file has assertions without a DB (CI-without-DB rung).
describe("summarize module mock wiring (no DB)", () => {
  it("memory.ts sees the mocked summarizeMemory", () => {
    expect(vi.isMockFunction(summarizeMemory)).toBe(true);
  });
});
