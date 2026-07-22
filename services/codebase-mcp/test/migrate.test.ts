import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const serviceDir = join(here, "..");
const migrateEntry = join(serviceDir, "src", "db", "migrate.ts");

// Run the REAL migrate runner against the test DB, then inspect the result with a
// fresh pool. Skipped when DATABASE_URL is absent (CI without a Postgres).
describe.skipIf(skip)("codebase-mcp migrate runner", () => {
  let pool: pg.Pool;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    // Start from a clean codebase schema so column assertions are unambiguous.
    await pool.query(
      "DROP SCHEMA IF EXISTS codebase CASCADE; CREATE SCHEMA codebase;",
    );
  });

  afterAll(async () => {
    await pool.end();
  });

  it("applies 001 then 002 cleanly and creates codebase.index_runs", () => {
    const out = execFileSync("npx", ["tsx", migrateEntry], {
      cwd: serviceDir,
      env: { ...process.env, DATABASE_URL },
      encoding: "utf8",
    });
    expect(out).toContain("001_codebase.sql");
    expect(out).toContain("002_index_jobs.sql");
    expect(out).toContain("006_edge_covering_indexes.sql");
    expect(out).toContain("migrations complete");
  });

  it("006 creates the covering symbol_edges indexes; the single-column ones never exist", async () => {
    const { rows } = await pool.query<{ indexname: string }>(
      `SELECT indexname FROM pg_indexes
       WHERE schemaname = 'codebase' AND tablename = 'symbol_edges'`,
    );
    const names = new Set(rows.map((r) => r.indexname));
    expect(names.has("symedge_from_kind")).toBe(true);
    expect(names.has("symedge_to_kind")).toBe(true);
    // 001 no longer creates the single-column prefixes; 006 keeps DROP IF EXISTS only
    // for DBs migrated before that cleanup. Either way they must be absent.
    expect(names.has("symedge_from")).toBe(false);
    expect(names.has("symedge_to")).toBe(false);
  });

  it("is idempotent (re-running is a no-op)", () => {
    const out = execFileSync("npx", ["tsx", migrateEntry], {
      cwd: serviceDir,
      env: { ...process.env, DATABASE_URL },
      encoding: "utf8",
    });
    expect(out).toContain("migrations complete");
  });

  it("adds the bake-off scratch column but SKIPS the HOLD drop (Wave 5)", async () => {
    const out = execFileSync("npx", ["tsx", migrateEntry], {
      cwd: serviceDir,
      env: { ...process.env, DATABASE_URL },
      encoding: "utf8",
    });
    // 004 applies; 005 is HOLD-skipped, not applied.
    expect(out).toContain("004_bakeoff_scratch.sql");
    expect(out).toMatch(/skipping 005_drop_bakeoff_scratch\.sql \(HOLD/);
    // the scratch column exists AND the live `embedding` column is still present.
    const { rows } = await pool.query<{ column_name: string }>(
      `SELECT column_name FROM information_schema.columns
       WHERE table_schema = 'codebase' AND table_name = 'code_chunks'
         AND column_name IN ('embedding', 'embedding_ctx')`,
    );
    const cols = new Set(rows.map((r) => r.column_name));
    expect(cols.has("embedding")).toBe(true);
    expect(cols.has("embedding_ctx")).toBe(true);
  });

  it("index_runs exists with the documented lifecycle columns", async () => {
    const { rows } = await pool.query<{ column_name: string }>(
      `SELECT column_name FROM information_schema.columns
       WHERE table_schema = 'codebase' AND table_name = 'index_runs'`,
    );
    const cols = new Set(rows.map((r) => r.column_name));
    for (const expected of [
      "id",
      "project_id",
      "repository_id",
      "phase",
      "files_total",
      "files_done",
      "chunks_total",
      "symbols_total",
      "edges_total",
      "current_file",
      "error",
      "cancel_requested",
      "started_at",
      "finished_at",
    ]) {
      expect(cols.has(expected), `missing column ${expected}`).toBe(true);
    }
  });

  it("a smoke insert/select round-trips through index_runs", async () => {
    const ins = await pool.query<{ id: string }>(
      `INSERT INTO codebase.index_runs (project_id, repository_id, phase)
       VALUES ('smoke-proj', 'smoke-repo', 'scanning') RETURNING id`,
    );
    const id = ins.rows[0].id;
    const sel = await pool.query<{ phase: string; cancel_requested: boolean }>(
      "SELECT phase, cancel_requested FROM codebase.index_runs WHERE id = $1",
      [id],
    );
    expect(sel.rows[0].phase).toBe("scanning");
    expect(sel.rows[0].cancel_requested).toBe(false);
    await pool.query("DELETE FROM codebase.index_runs WHERE id = $1", [id]);
  });
});
