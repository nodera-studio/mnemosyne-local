// harvest-eval operator script (wave-2 Step 8) — DB test (skipIf): seeds
// memory.search_log rows, runs the exported function with an injected pool, and
// asserts candidate shape, gold-hint extraction, reason labeling, and the
// already-in-gold exclusion. FREE script — no LLM, no Voyage anywhere.

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readdirSync } from "node:fs";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import pg from "pg";
import { harvestEval, existingGoldQueries } from "../src/db/harvest-eval.js";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const PROJ = "harvest-mem";
const id = (n: number) => `50000000-0000-4000-8000-00000000000${n}`;

describe.skipIf(skip)("harvest-eval (search_log → gold candidates)", () => {
  let pool: pg.Pool;
  const tmp = mkdtempSync(join(tmpdir(), "harvest-eval-"));
  const evalPath = join(tmp, "gold.json");
  const outPath = join(tmp, "candidates.json");

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 2 });
    // Apply the migration chain idempotently (fresh DBs need 006 for search_log).
    const sqlDir = join(here, "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);

    // A tmp gold file whose only query must be EXCLUDED from the harvest.
    writeFileSync(
      evalPath,
      JSON.stringify({
        k: 10,
        rows: [{ query: "Already In Gold", relevantTitles: ["t"] }],
      }),
    );

    const insert = async (
      query: string,
      poolIds: string[],
      finalIds: string[],
      times: number,
    ) => {
      for (let i = 0; i < times; i++) {
        await pool.query(
          `INSERT INTO memory.search_log (project_id, query, filters, pool_ids, final_ids, pool_ms, total_ms)
           VALUES ($1, $2, '{}', $3::uuid[], $4::uuid[], 5, 12)`,
          [PROJ, query, poolIds, finalIds],
        );
      }
    };

    // frequent: 3 hits, final ids sit in the pool's top 10 → high overlap.
    await insert("rotate api keys", [id(1), id(2), id(3)], [id(1), id(2)], 3);
    // zero-hit: pool found candidates, final list came back empty.
    await insert("ghost feature nobody built", [id(4)], [], 2);
    // low-overlap: final id is NOT in the pool's first 10 entries.
    const bigPool = Array.from(
      { length: 11 },
      (_, i) => `60000000-0000-4000-8000-0000000000${String(i + 10)}`,
    );
    await insert("deep pool rescue query", bigPool, [id(9)], 2);
    // already-in-gold (case/whitespace-insensitively) → excluded.
    await insert("already in gold", [id(5)], [id(5)], 5);
    // below min-count and never zero-hit → filtered by HAVING.
    await insert("seen once only", [id(6)], [id(6)], 1);
  });

  afterAll(async () => {
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
    rmSync(tmp, { recursive: true, force: true });
  });

  it("aggregates the log into approved:false candidates with gold hints and reasons", async () => {
    const { candidates, path } = await harvestEval({
      pool,
      projectId: PROJ,
      days: 30,
      minCount: 2,
      limit: 40,
      evalPath,
      outPath,
      log: () => {},
    });

    const byQuery = new Map(candidates.map((c) => [c.query, c]));

    // frequent query: counted, gold hints = most common final ids, right label.
    const freq = byQuery.get("rotate api keys");
    expect(freq).toBeDefined();
    expect(freq!.count).toBe(3);
    expect(freq!.archetype).toBe("frequent");
    expect(freq!.suggestedGold).toEqual([id(1), id(2)]);
    expect(freq!.provenance).toBe("log-harvest");
    expect(freq!.approved).toBe(false);

    // zero-hit surfaces even with empty finals.
    const ghost = byQuery.get("ghost feature nobody built");
    expect(ghost).toBeDefined();
    expect(ghost!.archetype).toBe("zero-hit");
    expect(ghost!.suggestedGold).toEqual([]);

    // low-overlap: final hit lives below the pool's top 10.
    const deep = byQuery.get("deep pool rescue query");
    expect(deep).toBeDefined();
    expect(deep!.archetype).toBe("low-overlap");

    // already-in-gold excluded; sub-min-count filtered.
    expect(byQuery.has("already in gold")).toBe(false);
    expect(byQuery.has("seen once only")).toBe(false);

    // written artifact carries the retention note + the same rows.
    const written = JSON.parse(readFileSync(path, "utf8")) as {
      _note: string;
      candidates: unknown[];
    };
    expect(written._note).toMatch(/90 days/);
    expect(written.candidates).toHaveLength(candidates.length);
  });

  it("gold exclusion normalizes case/whitespace (v1 and v2 files both carry `query`)", () => {
    const gold = existingGoldQueries(evalPath);
    expect(gold.has("already in gold")).toBe(true);
  });
});
