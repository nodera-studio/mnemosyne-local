// Regression (review Fix 3, superseded mechanism → importer write-path unification,
// tech-debt 4761d86c) — every importer row must carry the contextual `embedding_v2`:
// after the contextual flip, searchMemory filters `embedding_v2 IS NOT NULL`, so any
// memory ingested without it is invisible to recall until a manual backfill.
//
// ORIGINALLY this pinned the distill scripts' inline dual-write INSERT. The importers
// now route ALL writes through the shared invariant path (src/db/insert-memory.ts,
// consumed as ../memory-mcp/dist/db/insert-memory.js), which owns the dual-embedding
// write — pinned at DB level by test/insert-memory.test.ts. This STATIC guard (the
// scripts run on the box against live Voyage, so they cannot be unit-tested directly)
// now asserts the routing itself, over all THREE importers:
//   1. no direct `INSERT INTO memory.memories` remains (the bypass this closes);
//   2. every script imports + calls insertMemoryRow from the compiled shared path;
//   3. every script still computes the contextual vector via the contextual endpoint
//      with the correct one-chunk-per-doc / int8 / 1024-dim params (importer.mjs
//      included — it historically lacked the v2 write entirely: the AC-304 hole).

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const importerDir = join(here, "..", "..", "importer");

const SCRIPTS = [
  "importer.mjs",
  "distill-codex.mjs",
  "distill-transcripts.mjs",
];

describe("importers route writes through the shared invariant path", () => {
  for (const name of SCRIPTS) {
    describe(name, () => {
      const src = readFileSync(join(importerDir, name), "utf8");

      it("contains NO direct INSERT INTO memory.memories (the bypass is closed)", () => {
        expect(src).not.toMatch(/INSERT INTO memory\.memories/i);
      });

      it("imports and calls insertMemoryRow from memory-mcp's compiled shared path", () => {
        // Anchor on the import EXPRESSION, not the path string — the path also
        // appears in header comments, which must not satisfy this pin alone.
        expect(src).toMatch(
          /await import\(\s*['"]\.\.\/memory-mcp\/dist\/db\/insert-memory\.js['"]\s*\)/,
        );
        expect(src).toMatch(/insertMemoryRow\s*\(/);
      });

      it("asserts the dist contract version after import (stale-dist guard)", () => {
        expect(src).toMatch(/INSERT_MEMORY_CONTRACT\s*!==\s*1/);
        expect(src).toMatch(
          /typeof\s+helper\.insertMemoryRow\s*!==\s*['"]function['"]/,
        );
      });

      it("computes the contextual vector via /contextualizedembeddings (int8, 1024-dim)", () => {
        expect(src).toContain("/contextualizedembeddings");
        expect(src).toMatch(/output_dtype:\s*['"]int8['"]/);
        expect(src).toMatch(/output_dimension:\s*1024/);
        expect(src).toMatch(/embedContextual\s*\(/);
      });
    });
  }
});
