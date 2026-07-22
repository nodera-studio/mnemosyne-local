// Unit tests for the gold v1->v2 migration proposal script (wave-2 Step 1, AC-109).
// The pool is MOCKED (no DB, no network): we inject the resolved title->id rows and
// assert (a) the emitted v2 shape, (b) unresolved-title reporting, and (c) that the
// script is READ-ONLY — it must never issue anything but SELECTs.

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import type { Pool } from "pg";
import { migrateGold } from "../src/db/migrate-gold.js";

const tmp = mkdtempSync(join(tmpdir(), "migrate-gold-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const V1 = {
  k: 10,
  rows: [
    { query: "how is hybrid search fused", relevantTitles: ["RRF fusion"] },
    {
      query: "what embedding model is used",
      relevantTitles: ["Voyage model decision", "A Ghost Title"],
    },
  ],
};

function fakePool(resolved: Array<{ id: string; title: string }>) {
  const queries: { text: string; params: unknown[] }[] = [];
  const pool = {
    query: async (text: string, params: unknown[]) => {
      queries.push({ text, params });
      return { rows: resolved, rowCount: resolved.length };
    },
  } as unknown as Pool;
  return { pool, queries };
}

function writeV1(name: string): string {
  const p = join(tmp, name);
  writeFileSync(p, JSON.stringify(V1));
  return p;
}

describe("migrateGold (gold v1 -> v2 proposal)", () => {
  it("emits the v2 shape: version header, m-NNN ids, relevantIds, split dev, provenance seed-v1", async () => {
    const { pool } = fakePool([
      { id: "11111111-1111-4111-8111-111111111111", title: "RRF fusion" },
      {
        id: "22222222-2222-4222-8222-222222222222",
        title: "Voyage model decision",
      },
      { id: "33333333-3333-4333-8333-333333333333", title: "A Ghost Title" },
    ]);
    const outPath = join(tmp, "out-full.v2.json");
    const logs: string[] = [];
    const { file, unresolvedTitles } = await migrateGold(
      { pool, log: (m) => logs.push(m) },
      { projectId: "proj", evalPath: writeV1("v1-full.json"), outPath },
    );

    expect(file.version).toBe(2);
    expect(file.k).toBe(10);
    expect(Array.isArray(file.changelog)).toBe(true);
    expect(file.changelog[0]).toContain("PENDING HUMAN APPROVAL");
    expect(file.rows).toHaveLength(2);
    expect(file.rows[0]).toEqual({
      id: "m-001",
      query: "how is hybrid search fused",
      relevantIds: ["11111111-1111-4111-8111-111111111111"],
      split: "dev",
      provenance: "seed-v1",
    });
    expect(file.rows[1].id).toBe("m-002");
    expect(file.rows[1].relevantIds).toEqual([
      "22222222-2222-4222-8222-222222222222",
      "33333333-3333-4333-8333-333333333333",
    ]);
    // every row defaults to dev — the operator promotes ~1/3 to test by hand
    for (const r of file.rows) expect(r.split).toBe("dev");
    expect(unresolvedTitles).toEqual([]);

    // the review file is actually written, parseable, and identical to the return
    expect(JSON.parse(readFileSync(outPath, "utf8"))).toEqual(file);
    // the operator reminder (split assignment) is printed
    expect(logs.join("\n")).toContain('assign ~1/3 of rows to split:"test"');
  });

  it("reports unresolved titles to the log and keeps resolved ids on the row", async () => {
    const { pool } = fakePool([
      { id: "11111111-1111-4111-8111-111111111111", title: "RRF fusion" },
      {
        id: "22222222-2222-4222-8222-222222222222",
        title: "Voyage model decision",
      },
      // "A Ghost Title" deliberately NOT in the DB
    ]);
    const logs: string[] = [];
    const { file, unresolvedTitles } = await migrateGold(
      { pool, log: (m) => logs.push(m) },
      {
        projectId: "proj",
        evalPath: writeV1("v1-unres.json"),
        outPath: join(tmp, "out-unres.v2.json"),
      },
    );

    expect(unresolvedTitles).toEqual(["A Ghost Title"]);
    expect(logs.join("\n")).toContain("A Ghost Title");
    // the row survives with the subset that DID resolve
    expect(file.rows[1].relevantIds).toEqual([
      "22222222-2222-4222-8222-222222222222",
    ]);
  });

  it("reports duplicate-title collisions in the log AND the review file's changelog", async () => {
    const { pool } = fakePool([
      { id: "11111111-1111-4111-8111-111111111111", title: "RRF fusion" },
      // TWO memories share the normalized title -> ambiguous resolution
      {
        id: "22222222-2222-4222-8222-222222222222",
        title: "Voyage model decision",
      },
      {
        id: "44444444-4444-4444-8444-444444444444",
        title: "  voyage MODEL decision ",
      },
      { id: "33333333-3333-4333-8333-333333333333", title: "A Ghost Title" },
    ]);
    const logs: string[] = [];
    const { file, duplicateTitles } = await migrateGold(
      { pool, log: (m) => logs.push(m) },
      {
        projectId: "proj",
        evalPath: writeV1("v1-dup.json"),
        outPath: join(tmp, "out-dup.v2.json"),
      },
    );

    expect(duplicateTitles).toEqual([
      {
        title: "voyage model decision",
        ids: [
          "22222222-2222-4222-8222-222222222222",
          "44444444-4444-4444-8444-444444444444",
        ],
      },
    ]);
    // first id wins on the row itself …
    expect(file.rows[1].relevantIds[0]).toBe(
      "22222222-2222-4222-8222-222222222222",
    );
    // … and the ambiguity is visible to the approver in BOTH surfaces
    const changelog = file.changelog.join("\n");
    expect(changelog).toContain('AMBIGUOUS title "voyage model decision"');
    expect(changelog).toContain("44444444-4444-4444-8444-444444444444");
    expect(logs.join("\n")).toContain("AMBIGUOUS title(s)");
    expect(logs.join("\n")).toContain("voyage model decision");
  });

  it("matches titles case-insensitively and trimmed", async () => {
    const { pool, queries } = fakePool([
      { id: "11111111-1111-4111-8111-111111111111", title: "  RRF Fusion  " },
    ]);
    const { file } = await migrateGold(
      { pool },
      {
        projectId: "proj",
        evalPath: writeV1("v1-case.json"),
        outPath: join(tmp, "out-case.v2.json"),
      },
    );
    expect(file.rows[0].relevantIds).toEqual([
      "11111111-1111-4111-8111-111111111111",
    ]);
    // the lookup normalizes on the SQL side too
    expect(queries[0].params[1]).toContain("rrf fusion");
  });

  it("is READ-ONLY: issues exactly one SELECT and nothing else", async () => {
    const { pool, queries } = fakePool([]);
    await migrateGold(
      { pool },
      {
        projectId: "proj",
        evalPath: writeV1("v1-ro.json"),
        outPath: join(tmp, "out-ro.v2.json"),
      },
    );
    expect(queries).toHaveLength(1);
    expect(queries[0].text.trim().toUpperCase().startsWith("SELECT")).toBe(
      true,
    );
    for (const q of queries) {
      expect(q.text).not.toMatch(/\b(INSERT|UPDATE|DELETE|ALTER|DROP)\b/i);
    }
  });

  it("never touches the v1 input file", async () => {
    const evalPath = writeV1("v1-untouched.json");
    const before = readFileSync(evalPath, "utf8");
    const { pool } = fakePool([]);
    await migrateGold(
      { pool },
      { projectId: "proj", evalPath, outPath: join(tmp, "out-x.v2.json") },
    );
    expect(readFileSync(evalPath, "utf8")).toBe(before);
  });
});
