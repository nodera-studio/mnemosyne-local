// Wave 3 — graph-backfill test (AC-007, refined by review Fix 1). Simulates an already-
// indexed corpus that predates the graph feature (files + chunks present, symbols empty),
// then asserts the graph is backfilled for files whose content_sha256 is UNCHANGED.
//
// AC-007 refinement: graph extraction is now DECOUPLED from the sha-skip — `force` is no
// longer the ONLY way to backfill the graph. ANY reindex (forced or not) re-populates
// symbols/edges/imports for every graphable file; the sha-skip gates ONLY the expensive
// embed + chunk rewrite. So a NON-forced reindex backfills the graph WITHOUT re-embedding
// (no Voyage burn on an unchanged corpus), and `force` additionally bypasses the sha-skip
// to re-embed/re-chunk unchanged files (a full rebuild).

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const sampleRepoDir = join(here, "fixtures", "sample-repo");
const REPO = "force-graph-test";

const embedCode = vi.fn(async (texts: string[], _inputType: string) =>
  texts.map(() => Array.from({ length: 1024 }, () => 0.001)),
);
vi.mock("../src/voyage.js", () => ({
  embedCode: (...args: [string[], string]) => embedCode(...args),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

const { indexRepo } = await import("../src/indexer.js");

describe.skipIf(skip)("forced graph backfill (AC-007)", () => {
  let pool: pg.Pool;

  async function count(sql: string, params: unknown[] = []): Promise<number> {
    const { rows } = await pool.query<{ n: string }>(sql, params);
    return Number(rows[0].n);
  }

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
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
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.end();
  });

  it("any reindex backfills the graph on an unchanged-sha corpus (force re-embeds, non-force does not)", async () => {
    // 1. Initial index — files + chunks + symbols written.
    await indexRepo(sampleRepoDir, REPO, "test-proj");
    const filesIndexed = await count(
      "SELECT count(*) AS n FROM codebase.files WHERE repository_id=$1",
      [REPO],
    );
    expect(filesIndexed).toBeGreaterThan(0);

    // 2. Simulate a pre-graph corpus: drop the graph rows but KEEP files + chunks
    //    (so content_sha256 is unchanged on the next run).
    await pool.query(
      "DELETE FROM codebase.symbol_edges WHERE repository_id=$1",
      [REPO],
    );
    await pool.query("DELETE FROM codebase.symbols WHERE repository_id=$1", [
      REPO,
    ]);
    expect(
      await count(
        "SELECT count(*) AS n FROM codebase.symbols WHERE repository_id=$1",
        [REPO],
      ),
    ).toBe(0);

    // 3. AC-007 refinement (Fix 1): a NON-forced reindex now BACKFILLS the graph for an
    //    already-chunked corpus — force is no longer the only way. The sha is unchanged,
    //    so it does this WITHOUT re-embedding (no Voyage quota burned).
    embedCode.mockClear();
    await indexRepo(sampleRepoDir, REPO, "test-proj", false);
    expect(
      await count(
        "SELECT count(*) AS n FROM codebase.symbols WHERE repository_id=$1",
        [REPO],
      ),
    ).toBe(10);
    expect(
      await count(
        "SELECT count(*) AS n FROM codebase.symbol_edges WHERE repository_id=$1",
        [REPO],
      ),
    ).toBeGreaterThan(0);
    // The non-forced backfill path skips embed on every unchanged file.
    expect(embedCode).not.toHaveBeenCalled();

    // 4. A FORCED reindex bypasses the sha-skip entirely — it re-chunks the unchanged
    //    corpus (full rebuild) and keeps the graph populated. Since Wave 3's chunk-level
    //    embed cache (AC-404), re-entering the embed path no longer means re-SPENDING:
    //    every chunk's content_sha256 already has a stored vector, so the forced run
    //    rewrites the chunks with ZERO Voyage calls.
    embedCode.mockClear();
    await indexRepo(sampleRepoDir, REPO, "test-proj", true);
    expect(
      await count(
        "SELECT count(*) AS n FROM codebase.symbols WHERE repository_id=$1",
        [REPO],
      ),
    ).toBe(10);
    expect(
      await count(
        "SELECT count(*) AS n FROM codebase.symbol_edges WHERE repository_id=$1",
        [REPO],
      ),
    ).toBeGreaterThan(0);
    // The forced full rebuild still rewrote every chunk (embeddings all present)...
    expect(
      await count(
        `SELECT count(*) AS n FROM codebase.code_chunks
         WHERE repository_id=$1 AND embedding IS NOT NULL`,
        [REPO],
      ),
    ).toBeGreaterThan(0);
    // ...but the chunk-level cache satisfied all of them (AC-404): zero Voyage calls.
    expect(embedCode).not.toHaveBeenCalled();
  });
});
