import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const sampleRepoDir = join(here, "fixtures", "sample-repo");

// Mock the Voyage embedder so the lifecycle test is hermetic (no API key / quota).
// embedCode returns one deterministic 1024-dim vector per input chunk.
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0.001)),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

// Imported AFTER vi.mock so the mocked voyage module is in effect.
const { indexRepo } = await import("../src/indexer.js");
const voyage = await import("../src/voyage.js");

interface RunRow {
  phase: string;
  files_total: number | null;
  files_done: number | null;
  chunks_total: number | null;
  error: string | null;
  finished_at: Date | null;
}

describe.skipIf(skip)("indexer index_runs lifecycle", () => {
  let pool: pg.Pool;

  async function latestRun(repositoryId: string): Promise<RunRow | undefined> {
    const { rows } = await pool.query<RunRow>(
      `SELECT phase, files_total, files_done, chunks_total, error, finished_at
       FROM codebase.index_runs
       WHERE repository_id = $1
       ORDER BY started_at DESC LIMIT 1`,
      [repositoryId],
    );
    return rows[0];
  }

  async function countRuns(repositoryId: string): Promise<number> {
    const { rows } = await pool.query<{ n: string }>(
      "SELECT count(*) AS n FROM codebase.index_runs WHERE repository_id = $1",
      [repositoryId],
    );
    return Number(rows[0].n);
  }

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    // Ensure the schema + index_runs table exist (apply both migrations).
    const { readFileSync, readdirSync } = await import("node:fs");
    const sqlDir = join(here, "..", "sql");
    await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      // Skip HOLD migrations (e.g. 005_drop_bakeoff_scratch.sql) like the real runner.
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
  });

  afterEach(() => {
    vi.mocked(voyage.embedCode).mockClear();
  });

  afterAll(async () => {
    await pool.query(
      "DELETE FROM codebase.index_runs WHERE repository_id IN ('lifecycle-happy','lifecycle-error')",
    );
    await pool.query(
      "DELETE FROM codebase.files WHERE repository_id IN ('lifecycle-happy','lifecycle-error')",
    );
    await pool.end();
  });

  it("writes exactly one terminal 'done' run with counts (AC-001)", async () => {
    await indexRepo(sampleRepoDir, "lifecycle-happy", "test-proj");

    expect(await countRuns("lifecycle-happy")).toBe(1);
    const run = await latestRun("lifecycle-happy");
    expect(run?.phase).toBe("done");
    expect(run?.finished_at).toBeInstanceOf(Date);
    expect(run?.error).toBeNull();
    expect(run?.files_total).toBeGreaterThan(0);
    expect(run?.files_done).toBeGreaterThan(0);
    expect(run?.chunks_total).toBeGreaterThan(0);
  });

  it("records an 'error' run with the message when indexing throws (AC-002)", async () => {
    // Unique-content fixture: the chunk-level embed cache (Wave 3, AC-404) reuses stored
    // vectors by content_sha256 ACROSS repos, so indexing sample-repo again (its chunks
    // already embedded under 'lifecycle-happy') would never call the embedder and the
    // rejection below would go unconsumed. Unseen content guarantees a cache miss.
    const { mkdtempSync, writeFileSync, rmSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const errRepoDir = mkdtempSync(join(tmpdir(), "lifecycle-error-repo-"));
    writeFileSync(
      join(errRepoDir, "unique.ts"),
      "// lifecycle-error unique content\nexport const lifecycleErrorProbe = 1;\n",
      "utf8",
    );

    vi.mocked(voyage.embedCode).mockRejectedValueOnce(
      new Error("boom: voyage unavailable"),
    );

    try {
      await expect(
        indexRepo(errRepoDir, "lifecycle-error", "test-proj"),
      ).rejects.toThrow("boom: voyage unavailable");
    } finally {
      rmSync(errRepoDir, { recursive: true, force: true });
    }

    const run = await latestRun("lifecycle-error");
    expect(run?.phase).toBe("error");
    expect(run?.error).toContain("boom: voyage unavailable");
    expect(run?.finished_at).toBeInstanceOf(Date);
  });
});
