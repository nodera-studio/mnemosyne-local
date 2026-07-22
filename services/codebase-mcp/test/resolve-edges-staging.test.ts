// Wave 3 (graph hardening) — resolveEdges lock-window shrink (AC-403). Structural check
// via an injected instrumented client: the heavy name-resolution JOINs (INSERT INTO
// resolved_edges ... JOIN codebase.symbols ...) must run BEFORE the transaction opens,
// and the BEGIN..COMMIT window must contain ONLY the DELETE + INSERT-from-staged pair.
// Also pins the behavioral contract: resolved edges land correctly, an empty stage still
// wipes the repo's edges, and the session temp tables never leak past the call.
// Requires DATABASE_URL (skipIf otherwise).

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import pg from "pg";
import { resolveEdges } from "../src/indexer.js";
import type { ByNameEdge } from "../src/graph/extractor.js";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const REPO = "resolve-staging-test";
const PROJ = "resolve-staging-proj";

describe.skipIf(skip)("resolveEdges staged outside the txn (AC-403)", () => {
  let pool: pg.Pool;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    const { readFileSync, readdirSync } = await import("node:fs");
    const sqlDir = join(here, "..", "sql");
    await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      // Skip HOLD migrations like the real runner.
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query("DELETE FROM codebase.symbols WHERE repository_id=$1", [
      REPO,
    ]);
    // Two symbols in different files: a call edge and an import edge can both resolve.
    await pool.query(
      `INSERT INTO codebase.symbols (repository_id, project_id, name, kind, file_path, start_line)
       VALUES ($1, $2, 'callerFn', 'function', 'a.ts', 1),
              ($1, $2, 'calleeFn', 'function', 'b.ts', 1)`,
      [REPO, PROJ],
    );
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.symbols WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.end();
  });

  const edges: ByNameEdge[] = [
    {
      kind: "call",
      fromName: "callerFn",
      fromFile: "a.ts",
      toName: "calleeFn",
      siteLine: 2,
    },
    {
      kind: "import",
      fromName: null,
      fromFile: "a.ts",
      toName: "calleeFn",
      module: "./b.js",
      siteLine: 1,
    },
  ];

  it("runs the resolution JOINs before BEGIN; the txn is DELETE + INSERT-from-staged only", async () => {
    const client = await pool.connect();
    const recorded: string[] = [];
    const raw = client.query.bind(client) as (
      ...args: unknown[]
    ) => Promise<unknown>;
    client.query = ((...args: unknown[]) => {
      if (typeof args[0] === "string") recorded.push(args[0]);
      return raw(...args);
    }) as typeof client.query;

    try {
      const n = await resolveEdges(REPO, edges, client);
      expect(n).toBe(2); // 1 resolved call + 1 resolved import

      const beginIdx = recorded.findIndex((s) => s.trim() === "BEGIN");
      const commitIdx = recorded.findIndex((s) => s.trim() === "COMMIT");
      expect(beginIdx).toBeGreaterThan(-1);
      expect(commitIdx).toBeGreaterThan(beginIdx);

      // Every staging/resolve statement precedes BEGIN (heavy JOINs hold no lock on
      // symbol_edges), including the last INSERT INTO resolved_edges.
      const stagingIdxs = recorded
        .map((s, i) =>
          s.includes("staged_edges") || s.includes("resolved_edges") ? i : -1,
        )
        .filter(
          (i) => i !== -1 && !recorded[i].includes("codebase.symbol_edges"),
        );
      expect(stagingIdxs.length).toBeGreaterThan(0);
      const lastStagingBeforeCommit = stagingIdxs.filter((i) => i < commitIdx);
      expect(Math.max(...lastStagingBeforeCommit)).toBeLessThan(beginIdx);

      // The txn window contains EXACTLY the DELETE + the INSERT-from-staged.
      const inTxn = recorded
        .slice(beginIdx + 1, commitIdx)
        .map((s) => s.trim());
      expect(inTxn).toHaveLength(2);
      expect(inTxn[0]).toContain("DELETE FROM codebase.symbol_edges");
      expect(inTxn[1]).toContain("INSERT INTO codebase.symbol_edges");
      expect(inTxn[1]).toContain("FROM resolved_edges");
      // No JOIN work inside the window.
      for (const s of inTxn) expect(s).not.toContain("JOIN");

      // The session temp tables are dropped before resolveEdges returns.
      const { rows: leftovers } = await raw(
        `SELECT to_regclass('pg_temp.staged_edges') AS a,
                to_regclass('pg_temp.resolved_edges') AS b`,
      ).then(
        (r) => r as pg.QueryResult<{ a: string | null; b: string | null }>,
      );
      expect(leftovers[0].a).toBeNull();
      expect(leftovers[0].b).toBeNull();
    } finally {
      client.release();
    }
  });

  it("lands exactly the resolved call + import rows (behavioral no-op vs the old shape)", async () => {
    const { rows } = await pool.query<{
      from_name: string;
      to_name: string;
      kind: string;
    }>(
      `SELECT sf.name AS from_name, st.name AS to_name, e.kind
       FROM codebase.symbol_edges e
       JOIN codebase.symbols sf ON sf.id = e.from_symbol
       JOIN codebase.symbols st ON st.id = e.to_symbol
       WHERE e.repository_id=$1
       ORDER BY e.kind`,
      [REPO],
    );
    expect(rows).toEqual([
      { from_name: "callerFn", to_name: "calleeFn", kind: "call" },
      { from_name: "callerFn", to_name: "calleeFn", kind: "import" },
    ]);
  });

  it("an empty stage still wipes the repo's edges (empty-input path preserved)", async () => {
    const n = await resolveEdges(REPO, []);
    expect(n).toBe(0);
    const { rows } = await pool.query<{ n: string }>(
      "SELECT count(*) AS n FROM codebase.symbol_edges WHERE repository_id=$1",
      [REPO],
    );
    expect(Number(rows[0].n)).toBe(0);
  });
});
