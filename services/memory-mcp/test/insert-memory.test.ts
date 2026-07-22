// Importer write-path unification (tech-debt 4761d86c, addendum 3) — the shared
// invariant insert in src/db/insert-memory.ts:
//
//   (b) insertMemoryRow writes content_sha256 + metadata.dupCandidates + BOTH
//       embedding columns on every insert (the invariants the direct-SQL importers
//       historically skipped — the AC-304 unembedded-rows hole);
//   (c) an importer-shaped insert through the helper is indistinguishable from a
//       storeMemory row on those invariants (same sha recipe, same vectors stored,
//       same flag structure) — only provenance fields (source_uri/event_date/status)
//       differ, by design;
//   (d) idempotent re-ingest survives: the helper-computed sha is discoverable by
//       the importers' own pre-dedupe recipe (sha256 over content, hex), and
//       source_uri round-trips for the path-key dedupe from importer PR #3.
//
// DB-backed (disposable :5544 Postgres), Voyage fully MOCKED at module level —
// NO live quota. Skipped gracefully without DATABASE_URL.

import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const H = vi.hoisted(() => {
  /** 1024-dim vector with the given (index, value) components; zeros elsewhere. */
  function sparse(components: Array<[number, number]>): number[] {
    const v = new Array<number>(1024).fill(0);
    for (const [i, x] of components) v[i] = x;
    return v;
  }
  // cos(near, unit0) = 0.98 (> 0.9 → flagged); ortho is orthogonal to both.
  const unit0 = sparse([[0, 1]]);
  const near = sparse([
    [0, 0.98],
    [1, 0.199],
  ]);
  const ortho = sparse([[2, 1]]);
  const legacy = sparse([[4, 1]]);
  function vecForText(text: string): number[] {
    if (text.includes("NEARDUP")) return near;
    if (text.includes("ORTHO")) return ortho;
    return sparse([[3, 1]]);
  }
  return { sparse, unit0, near, ortho, legacy, vecForText };
});

vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map(() => H.legacy),
  embedContextualSingle: async (texts: string[]) =>
    texts.map((t) => H.vecForText(t)),
  rerank: async (_q: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_, i) => ({ index: i, score: 1 - i * 0.01 })),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

vi.mock("../src/summarize.js", () => ({
  summarizeMemory: async () => null,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  DUP_CANDIDATE_TOP_S,
  DUP_COSINE_THRESHOLD,
  contentSha256,
  insertMemoryRow,
  toVectorLiteral,
  type DupCandidate,
} from "../src/db/insert-memory.js";
import {
  contentSha256 as reExportedSha,
  DUP_COSINE_THRESHOLD as reExportedThreshold,
  DUP_CANDIDATE_TOP_S as reExportedTopS,
  storeMemory,
} from "../src/memory.js";

const here = dirname(fileURLToPath(import.meta.url));

/** The importers' own recipe (services/importer/*.mjs sha256(content), hex). */
const importerSha256 = (s: string) =>
  createHash("sha256").update(s).digest("hex");

interface InvariantRow {
  content_sha256: string | null;
  has_embedding: boolean;
  has_embedding_v2: boolean;
  embedding_text: string | null;
  embedding_v2_text: string | null;
  status: string;
  source_uri: string | null;
  summary: string | null;
  metadata: {
    tags?: string[];
    dupCandidates?: DupCandidate[];
    dupFlaggedAt?: string;
    [k: string]: unknown;
  };
}

describe.skipIf(skip)("insertMemoryRow — the shared invariant path", () => {
  let pool: pg.Pool;
  const PROJ = `insert-memory-${Date.now()}`;

  const invariants = async (id: string): Promise<InvariantRow> => {
    const { rows } = await pool.query<InvariantRow>(
      `SELECT content_sha256,
              embedding IS NOT NULL AS has_embedding,
              embedding_v2 IS NOT NULL AS has_embedding_v2,
              embedding::text AS embedding_text,
              embedding_v2::text AS embedding_v2_text,
              COALESCE(status,'active') AS status,
              source_uri, summary, metadata
       FROM memory.memories WHERE id = $1`,
      [id],
    );
    return rows[0];
  };

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Apply the migration chain idempotently (HOLD files skipped) — golden-suite pattern.
    const sqlDir = join(here, "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
  });

  afterAll(async () => {
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
  });

  it("(b) writes sha + BOTH embeddings; no dup flags when nothing clears the threshold", async () => {
    const content = "Importer row: ORTHO content unlike everything else.";
    const r = await insertMemoryRow(
      pool,
      {
        projectId: PROJ,
        type: "semantic",
        title: "orthogonal import",
        content,
        sourceKind: "docs",
        metadata: { tags: ["migrated", "docs"] },
      },
      { vec: H.legacy, vec2: H.ortho },
    );
    expect(r.dupCandidates).toEqual([]);

    const row = await invariants(r.id);
    expect(row.content_sha256).toBe(importerSha256(content));
    expect(row.has_embedding).toBe(true);
    expect(row.has_embedding_v2).toBe(true); // the AC-304 hole, closed
    expect(row.metadata.tags).toEqual(["migrated", "docs"]);
    expect(row.metadata.dupCandidates).toBeUndefined();
    expect(row.metadata.dupFlaggedAt).toBeUndefined();
    expect(row.status).toBe("active"); // default, same as the column default
  });

  it("(b) flags >threshold cosine neighbors into metadata.dupCandidates", async () => {
    // Anchor at unit0 (seeded raw); a NEARDUP-vec insert must flag it (cos 0.98).
    const anchor = await pool.query<{ id: string }>(
      `INSERT INTO memory.memories (project_id, type, title, content, embedding_v2)
       VALUES ($1, 'semantic', 'anchor', 'anchor content', $2::halfvec)
       RETURNING id`,
      [PROJ, toVectorLiteral(H.unit0)],
    );
    const r = await insertMemoryRow(
      pool,
      {
        projectId: PROJ,
        type: "episodic",
        title: "near-dup import",
        content: "Importer row that lands NEAR the anchor.",
        sourceKind: "codex-session",
      },
      { vec: H.legacy, vec2: H.near },
    );
    expect(r.dupCandidates.map((c) => c.id)).toContain(anchor.rows[0].id);

    const row = await invariants(r.id);
    expect(row.metadata.dupCandidates).toBeDefined();
    const flagged = row.metadata.dupCandidates!;
    expect(flagged.map((c) => c.id)).toContain(anchor.rows[0].id);
    for (const c of flagged)
      expect(c.sim).toBeGreaterThan(DUP_COSINE_THRESHOLD);
    expect(flagged.length).toBeLessThanOrEqual(DUP_CANDIDATE_TOP_S);
    expect(typeof row.metadata.dupFlaggedAt).toBe("string");
  });

  it("(b) non-active rows SKIP the neighbor pass — no flags, no ANN query (importer 'superseded' stubs)", async () => {
    // Same NEAR geometry that flags in the test above — but status:'superseded'
    // must not buy the ANN query: the flags prioritize consolidation of LIVE rows.
    const queries: string[] = [];
    const spyPool = {
      query: (text: string, values?: unknown[]) => {
        queries.push(text);
        return pool.query(text, values);
      },
    };
    const r = await insertMemoryRow(
      spyPool,
      {
        projectId: PROJ,
        type: "episodic",
        title: "superseded stub",
        content: "Importer stub that would otherwise flag the anchor.",
        sourceKind: "research",
        status: "superseded",
      },
      { vec: H.legacy, vec2: H.near },
    );
    expect(r.dupCandidates).toEqual([]);
    expect(queries).toHaveLength(1); // the INSERT only — no neighbor SELECT
    expect(queries[0]).toMatch(/INSERT INTO memory\.memories/);

    const row = await invariants(r.id);
    expect(row.metadata.dupCandidates).toBeUndefined();
    expect(row.metadata.dupFlaggedAt).toBeUndefined();
  });

  it("(c) an importer-shaped insert is indistinguishable from a storeMemory row on the invariants", async () => {
    // Row S via the canonical path (mocked embeds: legacy + NEARDUP contextual).
    const contentS = "NEARDUP store-path row for the parity check.";
    const s = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "store-path row",
      content: contentS,
    });

    // Row I via the helper with the SAME injected vectors, importer-shaped extras.
    const contentI =
      "Importer-batch row for the parity check (distinct content).";
    const i = await insertMemoryRow(
      pool,
      {
        projectId: PROJ,
        type: "semantic",
        title: "importer-path row",
        content: contentI,
        importance: 0.4,
        sourceKind: "claude-session",
        sourceUri:
          "/opt/mnemosyne/import-src/Developer/Apps/ExampleProject/docs/x.md",
        eventDate: "2026-07-01",
        status: "active",
        metadata: { tags: ["claude-session", "distilled"] },
      },
      { vec: H.legacy, vec2: H.near },
    );

    const rowS = await invariants(s.id);
    const rowI = await invariants(i.id);

    // Same sha recipe, each over its OWN content.
    expect(rowS.content_sha256).toBe(importerSha256(contentS));
    expect(rowI.content_sha256).toBe(importerSha256(contentI));

    // Identical stored vectors (same injected embeddings ⇒ byte-identical columns).
    expect(rowI.embedding_text).toBe(rowS.embedding_text);
    expect(rowI.embedding_v2_text).toBe(rowS.embedding_v2_text);
    expect(rowS.has_embedding && rowS.has_embedding_v2).toBe(true);
    expect(rowI.has_embedding && rowI.has_embedding_v2).toBe(true);

    // Same dup-flag structure: both sit near the anchor from the previous test, so
    // both carry dupCandidates with >threshold sims and a dupFlaggedAt timestamp.
    for (const row of [rowS, rowI]) {
      expect(row.metadata.dupCandidates).toBeDefined();
      for (const c of row.metadata.dupCandidates!) {
        expect(c.sim).toBeGreaterThan(DUP_COSINE_THRESHOLD);
        expect(c.sim).toBe(Number(c.sim.toFixed(4)));
      }
      expect(typeof row.metadata.dupFlaggedAt).toBe("string");
      expect(row.status).toBe("active");
      expect(row.summary).toBeNull();
    }
    // Provenance is the ONLY intended divergence.
    expect(rowS.source_uri).toBeNull();
    expect(rowI.source_uri).toBe(
      "/opt/mnemosyne/import-src/Developer/Apps/ExampleProject/docs/x.md",
    );

    // The importer row participates in the store-path exact-dup short-circuit:
    // storeMemory of the SAME content must dedupe onto the helper-inserted row.
    const dup = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "retitled restatement",
      content: contentI,
    });
    expect(dup).toEqual({
      id: i.id,
      title: "importer-path row",
      duplicate: true,
    });
  });

  it("(d) idempotent re-ingest: helper-written rows are discoverable by the importers' pre-dedupe reads", async () => {
    const content = "Re-ingest guard content, hashed by both recipes.";
    const uri =
      "/opt/mnemosyne/import-src/Developer/Apps/ExampleProject/.claude/plans/y.md";
    const r = await insertMemoryRow(
      pool,
      {
        projectId: PROJ,
        type: "procedural",
        title: "re-ingest guard",
        content,
        sourceKind: "plan",
        sourceUri: uri,
        eventDate: new Date("2026-06-15"),
      },
      { vec: H.legacy, vec2: H.ortho },
    );

    // Importer pre-pass 1 (content hash): SELECT content_sha256 … must surface the
    // helper-computed sha under the importers' own createHash recipe.
    const shas = new Set(
      (
        await pool.query<{ content_sha256: string }>(
          `SELECT content_sha256 FROM memory.memories WHERE content_sha256 IS NOT NULL`,
        )
      ).rows.map((x) => x.content_sha256),
    );
    expect(shas.has(importerSha256(content))).toBe(true);

    // Importer pre-pass 2 (path key): source_uri round-trips for pathKey() dedupe.
    const { rows } = await pool.query<{
      source_uri: string;
      event_date: string;
    }>(
      `SELECT source_uri, event_date::date::text AS event_date
       FROM memory.memories WHERE id = $1`,
      [r.id],
    );
    expect(rows[0].source_uri).toBe(uri);
    expect(rows[0].event_date).toBe("2026-06-15");
  });
});

// Pure guards — always run, no DB.
describe("insert-memory invariants (pure)", () => {
  it("contentSha256 is the importer recipe and memory.ts re-exports the SAME function", () => {
    expect(contentSha256("abc")).toBe(importerSha256("abc"));
    expect(contentSha256("abc")).toMatch(/^[0-9a-f]{64}$/);
    expect(reExportedSha).toBe(contentSha256);
  });

  it("thresholds are single-sourced: memory.ts re-exports the insert-memory constants", () => {
    expect(reExportedThreshold).toBe(DUP_COSINE_THRESHOLD);
    expect(reExportedThreshold).toBe(0.9);
    expect(reExportedTopS).toBe(DUP_CANDIDATE_TOP_S);
    expect(reExportedTopS).toBe(3);
  });

  it("toVectorLiteral serializes the pgvector literal shape — and stays byte-identical to voyage.ts's copy", async () => {
    expect(toVectorLiteral([1, 0, 0.5])).toBe("[1,0,0.5]");
    // The duplication is deliberate (keeps this module a config-free leaf); this
    // pin is what keeps the two copies from silently diverging. importActual
    // bypasses this file's voyage mock — we need the REAL implementation.
    const { toVectorLiteral: voyageCopy } =
      await vi.importActual<typeof import("../src/voyage.js")>(
        "../src/voyage.js",
      );
    const probe = [0.123456789, -1, 0, 42.5, 1e-7];
    expect(voyageCopy(probe)).toBe(toVectorLiteral(probe));
  });

  it("INSERT_MEMORY_CONTRACT is exported for the importers' stale-dist assert", async () => {
    const mod = await import("../src/db/insert-memory.js");
    expect(mod.INSERT_MEMORY_CONTRACT).toBe(1);
  });

  it("insert-memory stays a dependency-free leaf (node:crypto only) — safe for plain-node importer consumption", () => {
    const src = readFileSync(
      join(here, "..", "src", "db", "insert-memory.ts"),
      "utf8",
    );
    const valueImports = [
      ...src.matchAll(/^import\s+(?!type\s)[^;]+from\s+["']([^"']+)["']/gm),
    ].map((m) => m[1]);
    expect(valueImports).toEqual(["node:crypto"]);
  });
});
