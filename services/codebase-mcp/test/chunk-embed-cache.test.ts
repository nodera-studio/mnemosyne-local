// Wave 3 (graph hardening) — chunk-level embed cache (AC-404). Counts calls on the
// mocked embedder: re-indexing unchanged content with `force` (which bypasses the
// file-level sha-skip, so every chunk goes through the embed path) must make ZERO
// embedCode calls; mutating one chunk's content must embed exactly the changed chunk.
// The fixture is generated into a tmpdir with content unique to this suite so no other
// suite's rows can pre-populate the content_sha256 cache. Requires DATABASE_URL.

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const REPO = "embed-cache-test";
const PROJ = "embed-cache-proj";
const MUTATION_MARKER = "// embed-cache-mutated-line";

// Hermetic call-counting embedder (no Voyage key/quota).
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0.002)),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

const { indexRepo } = await import("../src/indexer.js");
const { embedCode } = await import("../src/voyage.js");
const embedMock = vi.mocked(embedCode);

/** Total number of TEXTS embedded across all embedCode calls since the last reset. */
function embeddedTexts(): string[] {
  return embedMock.mock.calls.flatMap(([texts]) => texts);
}

describe.skipIf(skip)("chunk-level embed cache (AC-404)", () => {
  let pool: pg.Pool;
  let tmpRepo: string;
  let alphaPath: string;

  // alpha.ts: 100 unique comment lines → 2 chunks (60-line windows, 12-line overlap:
  // lines 1–60 and 49–100). beta.ts: 8 lines → 1 chunk. 3 chunks total.
  const alphaLines = Array.from(
    { length: 100 },
    (_, i) => "// embed-cache-alpha-line-" + (i + 1),
  );
  const betaLines = Array.from(
    { length: 8 },
    (_, i) => "// embed-cache-beta-line-" + (i + 1),
  );

  async function chunkCount(): Promise<number> {
    const { rows } = await pool.query<{ n: string }>(
      `SELECT count(*) AS n FROM codebase.code_chunks
       WHERE repository_id=$1 AND embedding IS NOT NULL`,
      [REPO],
    );
    return Number(rows[0].n);
  }

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    const { readFileSync, readdirSync, mkdtempSync, writeFileSync } =
      await import("node:fs");
    const { tmpdir } = await import("node:os");
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
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);

    tmpRepo = mkdtempSync(join(tmpdir(), "embed-cache-repo-"));
    alphaPath = join(tmpRepo, "alpha.ts");
    writeFileSync(alphaPath, alphaLines.join("\n"), "utf8");
    writeFileSync(join(tmpRepo, "beta.ts"), betaLines.join("\n"), "utf8");
  });

  afterAll(async () => {
    const { rmSync } = await import("node:fs");
    rmSync(tmpRepo, { recursive: true, force: true });
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.end();
  });

  it("first index embeds every chunk (cold cache)", async () => {
    embedMock.mockClear();
    await indexRepo(tmpRepo, REPO, PROJ);
    expect(embeddedTexts()).toHaveLength(3);
    expect(await chunkCount()).toBe(3);
  });

  it("a forced reindex of unchanged content makes ZERO embedCode calls (AC-404)", async () => {
    embedMock.mockClear();
    // force=true bypasses the file-level sha-skip, so every chunk re-enters the embed
    // path — the chunk-level content_sha256 cache must satisfy all of them.
    await indexRepo(tmpRepo, REPO, PROJ, true);
    expect(embedMock).not.toHaveBeenCalled();
    expect(await chunkCount()).toBe(3);
  });

  it("mutating one chunk re-embeds exactly the changed chunk", async () => {
    const { writeFileSync } = await import("node:fs");
    // The embedding literal of alpha's UNCHANGED first chunk (start_line 1) must
    // survive the reindex byte-for-byte (reused via ::text, no float round-trip).
    const { rows: before } = await pool.query<{ embedding: string }>(
      `SELECT embedding::text AS embedding FROM codebase.code_chunks
       WHERE repository_id=$1 AND file_path='alpha.ts' AND start_line=1`,
      [REPO],
    );

    // Line 90 lives ONLY in alpha's second chunk (lines 49–100); chunk 1 (1–60) and
    // beta.ts are untouched.
    const mutated = [...alphaLines];
    mutated[89] = MUTATION_MARKER;
    writeFileSync(alphaPath, mutated.join("\n"), "utf8");

    embedMock.mockClear();
    await indexRepo(tmpRepo, REPO, PROJ); // non-forced: beta sha-skips, alpha re-chunks
    const texts = embeddedTexts();
    expect(texts).toHaveLength(1);
    expect(texts[0]).toContain(MUTATION_MARKER);
    expect(await chunkCount()).toBe(3);

    const { rows: after } = await pool.query<{ embedding: string }>(
      `SELECT embedding::text AS embedding FROM codebase.code_chunks
       WHERE repository_id=$1 AND file_path='alpha.ts' AND start_line=1`,
      [REPO],
    );
    expect(after[0].embedding).toBe(before[0].embedding);
  });
});
