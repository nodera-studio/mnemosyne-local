// Regression (review Fix 2) — call edges must be scoped to the caller's file. Two files
// each define `foo`; only a.ts's `foo` calls `bar`. The call-edge FROM join is anchored
// on BOTH name AND file_path (se.from_file), so exactly ONE call edge a.foo→bar is
// resolved — NOT a second spurious b.foo→bar from the same-named `foo` in b.ts.
// Requires DATABASE_URL (skipIf otherwise).

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const dupCallerDir = join(here, "fixtures", "dup-caller-repo");
const REPO = "call-edge-scope-test";
const PROJ = "call-edge-scope-proj";

// Hermetic embedder (no Voyage key/quota).
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0.001)),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

const { indexRepo } = await import("../src/indexer.js");

describe.skipIf(skip)("call-edge file scoping (Fix 2)", () => {
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
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
    await indexRepo(dupCallerDir, REPO, PROJ);
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

  it("resolves exactly one foo→bar call edge, originating from a.ts only", async () => {
    const { rows } = await pool.query<{ from_file: string }>(
      `SELECT sf.file_path AS from_file
       FROM codebase.symbol_edges e
       JOIN codebase.symbols sf ON sf.id = e.from_symbol
       JOIN codebase.symbols st ON st.id = e.to_symbol
       WHERE e.repository_id=$1 AND e.kind='call'
         AND sf.name='foo' AND st.name='bar'`,
      [REPO],
    );
    // Before the fix: 2 rows (a.foo→bar AND spurious b.foo→bar). After: exactly 1.
    expect(rows.length).toBe(1);
    expect(rows[0].from_file).toBe("a.ts");
  });

  it("b.ts's foo has NO outgoing call edge to bar", async () => {
    const { rows } = await pool.query<{ n: string }>(
      `SELECT count(*) AS n
       FROM codebase.symbol_edges e
       JOIN codebase.symbols sf ON sf.id = e.from_symbol
       JOIN codebase.symbols st ON st.id = e.to_symbol
       WHERE e.repository_id=$1 AND e.kind='call'
         AND sf.name='foo' AND sf.file_path='b.ts' AND st.name='bar'`,
      [REPO],
    );
    expect(Number(rows[0].n)).toBe(0);
  });
});
