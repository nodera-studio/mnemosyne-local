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

const REPO = "graph-constraints-test";

// Apply the FULL 001->002->003 chain via the real migrate runner, then assert the
// Wave-2 hardening (project_id, file_id FK, sym_identity unique key) holds against
// a fresh pool. Skipped when DATABASE_URL is absent (CI without a Postgres).
describe.skipIf(skip)("codebase-mcp graph constraints (003)", () => {
  let pool: pg.Pool;

  // Seed a file row so symbols can reference a real file_id.
  async function seedFile(path: string): Promise<string> {
    const { rows } = await pool.query<{ id: string }>(
      `INSERT INTO codebase.files (repository_id, project_id, path, content_sha256)
       VALUES ($1, 'proj', $2, 'sha-' || $2) RETURNING id`,
      [REPO, path],
    );
    return rows[0].id;
  }

  async function insertSymbol(
    fileId: string,
    filePath: string,
    name: string,
    kind: string,
    startLine: number,
  ): Promise<string> {
    const { rows } = await pool.query<{ id: string }>(
      `INSERT INTO codebase.symbols
         (repository_id, project_id, file_id, file_path, name, kind, start_line)
       VALUES ($1, 'proj', $2, $3, $4, $5, $6) RETURNING id`,
      [REPO, fileId, filePath, name, kind, startLine],
    );
    return rows[0].id;
  }

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    // Clean codebase schema so the chain applies from scratch and column/index
    // assertions are unambiguous.
    await pool.query(
      "DROP SCHEMA IF EXISTS codebase CASCADE; CREATE SCHEMA codebase;",
    );
    execFileSync("npx", ["tsx", migrateEntry], {
      cwd: serviceDir,
      env: { ...process.env, DATABASE_URL },
      encoding: "utf8",
    });
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.files WHERE repository_id = $1", [
      REPO,
    ]);
    await pool.end();
  });

  it("adds project_id (NOT NULL) and file_id columns to codebase.symbols", async () => {
    const { rows } = await pool.query<{
      column_name: string;
      is_nullable: string;
      data_type: string;
    }>(
      `SELECT column_name, is_nullable, data_type
       FROM information_schema.columns
       WHERE table_schema = 'codebase' AND table_name = 'symbols'
         AND column_name IN ('project_id', 'file_id')`,
    );
    const byName = new Map(rows.map((r) => [r.column_name, r]));
    expect(byName.get("project_id")?.is_nullable).toBe("NO");
    expect(byName.get("file_id")?.data_type).toBe("uuid");
  });

  it("exposes the sym_identity unique index on the five identity columns", async () => {
    const { rows } = await pool.query<{ indexdef: string }>(
      `SELECT indexdef FROM pg_indexes
       WHERE schemaname = 'codebase' AND tablename = 'symbols'
         AND indexname = 'sym_identity'`,
    );
    expect(rows).toHaveLength(1);
    const def = rows[0].indexdef;
    expect(def).toContain("UNIQUE");
    for (const col of [
      "repository_id",
      "file_path",
      "name",
      "kind",
      "start_line",
    ]) {
      expect(def).toContain(col);
    }
  });

  it("declares the file_id FK to codebase.files with ON DELETE CASCADE", async () => {
    const { rows } = await pool.query<{ confdeltype: string }>(
      `SELECT confdeltype
       FROM pg_constraint
       WHERE conname = 'symbols_file_id_fkey'
         AND conrelid = 'codebase.symbols'::regclass
         AND confrelid = 'codebase.files'::regclass`,
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].confdeltype).toBe("c"); // c = CASCADE
  });

  it("sym_identity rejects a duplicate (repository_id, file_path, name, kind, start_line)", async () => {
    const fileId = await seedFile("dup.ts");
    await insertSymbol(fileId, "dup.ts", "doThing", "function", 10);
    await expect(
      insertSymbol(fileId, "dup.ts", "doThing", "function", 10),
    ).rejects.toThrow(/sym_identity|duplicate key/);
  });

  it("file_id FK rejects an orphan reference to a non-existent file", async () => {
    await expect(
      pool.query(
        `INSERT INTO codebase.symbols
           (repository_id, project_id, file_id, file_path, name, kind, start_line)
         VALUES ($1, 'proj', gen_random_uuid(), 'orphan.ts', 'orphan', 'function', 1)`,
        [REPO],
      ),
    ).rejects.toThrow(/foreign key|symbols_file_id_fkey/);
  });

  it("deleting a file cascades to its symbols and their dependent edges", async () => {
    const fileId = await seedFile("cascade.ts");
    const callerFileId = await seedFile("caller.ts");
    const callee = await insertSymbol(
      fileId,
      "cascade.ts",
      "callee",
      "function",
      5,
    );
    // An edge from a symbol in ANOTHER file pointing INTO the callee — this is the
    // cross-file edge that must also cascade away when the callee's file is deleted.
    const caller = await insertSymbol(
      callerFileId,
      "caller.ts",
      "caller",
      "function",
      3,
    );
    await pool.query(
      `INSERT INTO codebase.symbol_edges (repository_id, from_symbol, to_symbol, kind)
       VALUES ($1, $2, $3, 'call')`,
      [REPO, caller, callee],
    );

    await pool.query("DELETE FROM codebase.files WHERE id = $1", [fileId]);

    const sym = await pool.query(
      "SELECT 1 FROM codebase.symbols WHERE id = $1",
      [callee],
    );
    expect(sym.rowCount).toBe(0); // symbol cascaded away with its file
    const edge = await pool.query(
      "SELECT 1 FROM codebase.symbol_edges WHERE to_symbol = $1",
      [callee],
    );
    expect(edge.rowCount).toBe(0); // dependent edge cascaded too
  });
});
