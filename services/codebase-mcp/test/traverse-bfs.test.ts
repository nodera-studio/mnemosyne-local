// Wave 3 (graph hardening) — BFS/CTE parity + BFS behavior (AC-401/402).
//
// The sample-repo fixture CONTAINS import edges alongside call edges (route.ts imports
// handleGetUser, handler.ts imports UserRow, ... — see fixtures/graph-fixture.md), so
// asserting BFS ≡ CTE row sets on it proves the import-edge exclusion by construction:
// any import-edge leakage in either engine (e.g. UserRow appearing in a call expansion)
// breaks parity/canary assertions. Also covers: cycle termination via the visited set,
// the injectable per-level cap, router decisions, ambiguity parity, and the 006 covering
// index actually serving the frontier probe. Requires DATABASE_URL (skipIf otherwise).

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const sampleRepoDir = join(here, "fixtures", "sample-repo");
const REPO = "bfs-parity-test";
const PROJ = "bfs-parity-proj";
const STAR_REPO = "bfs-star-test";
const STAR_PROJ = "bfs-star-proj";

// Hermetic embedder (no Voyage key/quota) — one deterministic 1024-dim vector per chunk.
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0.001)),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

const { indexRepo } = await import("../src/indexer.js");
const {
  expandCte,
  expandBfs,
  graphExpand,
  tracePath,
  resolveSeed,
  usesBfs,
  BFS_DEPTH_THRESHOLD,
  MAX_DEPTH,
} = await import("../src/graph/traverse.js");

describe.skipIf(skip)("app-side BFS vs recursive CTE (AC-401/402)", () => {
  let pool: pg.Pool;

  async function seedIdsFor(name: string): Promise<string[]> {
    const rows = await resolveSeed({ projectId: PROJ, repo: REPO, name });
    expect(rows.length).toBeGreaterThan(0);
    return rows.map((r) => r.id);
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
    for (const repo of [REPO, STAR_REPO]) {
      await pool.query(
        "DELETE FROM codebase.index_runs WHERE repository_id=$1",
        [repo],
      );
      await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
        repo,
      ]);
      await pool.query("DELETE FROM codebase.symbols WHERE repository_id=$1", [
        repo,
      ]);
    }
    await indexRepo(sampleRepoDir, REPO, PROJ);

    // Star fixture for the per-level cap: star_root --call--> 6000 generated leaves.
    // Inserted directly (no files rows needed — symbols.file_id is nullable).
    await pool.query(
      `INSERT INTO codebase.symbols (repository_id, project_id, name, kind, file_path, start_line)
       SELECT $1, $2, 'leaf_' || g, 'function', 'star.ts', g + 1
       FROM generate_series(1, 6000) g`,
      [STAR_REPO, STAR_PROJ],
    );
    await pool.query(
      `INSERT INTO codebase.symbols (repository_id, project_id, name, kind, file_path, start_line)
       VALUES ($1, $2, 'star_root', 'function', 'star.ts', 1)`,
      [STAR_REPO, STAR_PROJ],
    );
    await pool.query(
      `INSERT INTO codebase.symbol_edges (repository_id, from_symbol, to_symbol, kind)
       SELECT $1, r.id, l.id, 'call'
       FROM codebase.symbols r, codebase.symbols l
       WHERE r.repository_id = $1 AND r.name = 'star_root'
         AND l.repository_id = $1 AND l.name LIKE 'leaf_%'`,
      [STAR_REPO],
    );
  });

  afterAll(async () => {
    for (const repo of [REPO, STAR_REPO]) {
      await pool.query(
        "DELETE FROM codebase.index_runs WHERE repository_id=$1",
        [repo],
      );
      await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
        repo,
      ]);
      // Star symbols have no files row to cascade from; edges cascade off symbols.
      await pool.query("DELETE FROM codebase.symbols WHERE repository_id=$1", [
        repo,
      ]);
    }
    await pool.end();
  });

  it("the fixture graph really contains import edges (AC-402 precondition)", async () => {
    const { rows } = await pool.query<{ n: string }>(
      `SELECT count(*) AS n FROM codebase.symbol_edges
       WHERE repository_id=$1 AND kind='import'`,
      [REPO],
    );
    expect(Number(rows[0].n)).toBeGreaterThan(0);
  });

  it("BFS ≡ CTE row sets at depths 2/3/4 for every direction (parity, AC-402)", async () => {
    const cases: Array<{
      name: string;
      direction: "callees" | "callers" | "both";
    }> = [
      { name: "route", direction: "callees" },
      { name: "route", direction: "both" },
      { name: "handleGetUser", direction: "callees" },
      { name: "getUser", direction: "callers" },
      { name: "getUser", direction: "both" },
      { name: "useFormat", direction: "callees" },
      { name: "alpha", direction: "both" },
    ];
    for (const c of cases) {
      const ids = await seedIdsFor(c.name);
      for (const depth of [2, 3, 4]) {
        const cte = await expandCte(ids, c.direction, depth, 50);
        const bfs = await expandBfs(ids, c.direction, depth, 50);
        // Identical {id, depth, edge_type, ambiguous} rows in identical order (both
        // engines sort depth, file_path, start_line).
        expect(bfs, `${c.name}/${c.direction}@${depth}`).toEqual(cte);
      }
    }
  });

  it("neither engine follows import edges out of a call expansion (leak canary)", async () => {
    // handler.ts imports UserRow from repo.ts, so the handleGetUser→UserRow IMPORT edge
    // exists; a call expansion that leaked import edges would surface UserRow at depth 1.
    const ids = await seedIdsFor("handleGetUser");
    for (const rows of [
      await expandCte(ids, "callees", 4, 50),
      await expandBfs(ids, "callees", 4, 50),
    ]) {
      expect(rows.map((r) => r.name)).not.toContain("UserRow");
      for (const r of rows) {
        if (r.depth > 0) expect(r.edge_type).toContain("call");
      }
    }
  });

  it("the A↔B cycle terminates at depth 8 via the visited set, min depth per node", async () => {
    const ids = await seedIdsFor("alpha");
    const rows = await expandBfs(ids, "callees", 8, 50);
    expect(rows.map((r) => `${r.name}@${r.depth}`).sort()).toEqual([
      "alpha@0",
      "beta@1",
    ]);
  });

  it("both-direction equal-depth tie keeps the callees label in BOTH engines", async () => {
    // From alpha, beta is reached at depth 1 as callee (call→) AND caller (call←);
    // the deterministic tie-break (callees first) must agree across engines.
    const ids = await seedIdsFor("alpha");
    const cte = await expandCte(ids, "both", 2, 50);
    const bfs = await expandBfs(ids, "both", 2, 50);
    const cteBeta = cte.find((r) => r.name === "beta");
    const bfsBeta = bfs.find((r) => r.name === "beta");
    expect(cteBeta?.edge_type).toBe("call→");
    expect(bfsBeta?.edge_type).toBe("call→");
  });

  it("flags the ambiguous `format` target identically in both engines (AC-010 parity)", async () => {
    const ids = await seedIdsFor("useFormat");
    for (const rows of [
      await expandCte(ids, "callees", 2, 50),
      await expandBfs(ids, "callees", 2, 50),
    ]) {
      const formats = rows.filter((r) => r.name === "format");
      expect(formats.length).toBe(2);
      expect(formats.every((r) => r.ambiguous)).toBe(true);
    }
  });

  it("the injectable per-level cap truncates a hyper-connected frontier without error", async () => {
    const { rows: root } = await pool.query<{ id: string }>(
      "SELECT id FROM codebase.symbols WHERE repository_id=$1 AND name='star_root'",
      [STAR_REPO],
    );
    const out = await expandBfs([root[0].id], "callees", 1, 1000, 10);
    // star_root (depth 0) + a capped frontier of exactly 10 of the 6000 leaves.
    expect(out.length).toBe(11);
    expect(out.filter((r) => r.depth === 1).length).toBe(10);
  });

  it("routes depth ≤ 4 to the CTE and depth ≥ 5 to BFS (AC-401)", async () => {
    expect(BFS_DEPTH_THRESHOLD).toBe(5);
    expect(usesBfs(1)).toBe(false);
    expect(usesBfs(4)).toBe(false);
    expect(usesBfs(5)).toBe(true);
    expect(usesBfs(MAX_DEPTH)).toBe(true);
    expect(usesBfs(99)).toBe(true); // clamps to MAX_DEPTH, still ≥ threshold
    expect(usesBfs(undefined)).toBe(false); // DEFAULT_DEPTH = 4
  });

  it("graphExpand at depth 5 (BFS route) matches the depth-4 CTE result on the fixture", async () => {
    // The route chain bottoms out at depth 3, so the depth-5 BFS expansion must equal
    // the depth-4 CTE expansion row for row.
    const ids = await seedIdsFor("route");
    const viaRouter = await graphExpand({
      seed: { projectId: PROJ, repo: REPO, name: "route" },
      direction: "callees",
      depth: 5,
    });
    const cte = await expandCte(ids, "callees", 4, 50);
    expect(viaRouter).toEqual(cte);
  });

  it("tracePath's no-target fallback inherits BFS routing at depth ≥ 5", async () => {
    const { rows, reachedTo } = await tracePath({
      seed: { projectId: PROJ, repo: REPO, name: "route" },
      depth: 8,
    });
    expect(reachedTo).toBe(false);
    const ids = await seedIdsFor("route");
    const cte = await expandCte(ids, "callees", 4, 50);
    expect(rows).toEqual(cte);
  });

  it("the frontier probe uses the 006 covering index (index-only path exists)", async () => {
    const client = await pool.connect();
    try {
      // The fixture tables are tiny, so the planner would pick a seq scan; disabling it
      // proves the covering index CAN serve the probe (the index-only path exists).
      // ANALYZE gives the planner real stats (the star fixture's 6k edges), so the
      // (from_symbol, kind) index beats a full scan of the (to_symbol, kind) one.
      await client.query("ANALYZE codebase.symbol_edges");
      await client.query("SET enable_seqscan = off");
      const { rows } = await client.query<Record<string, unknown>>(
        `EXPLAIN (FORMAT JSON)
         SELECT to_symbol FROM codebase.symbol_edges
         WHERE from_symbol = '00000000-0000-0000-0000-000000000001'::uuid
           AND kind = 'call'`,
      );
      expect(JSON.stringify(rows)).toContain("symedge_from_kind");
    } finally {
      await client.query("RESET enable_seqscan").catch(() => {});
      client.release();
    }
  });
});
