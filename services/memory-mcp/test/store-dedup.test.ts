// Wave 5 (WS3), Steps 1–2 — the CHEAP write-path checks in storeMemory:
//
//   AC-301 — content_sha256 is populated on insert (importers' exact recipe:
//            sha256 hex over CONTENT ONLY); an active same-hash row in the project
//            short-circuits to the existing id with NO insert and NO embed spend.
//   AC-302 — top-s cosine neighbors above DUP_COSINE_THRESHOLD are recorded into
//            metadata.dupCandidates (ids + sims) — a flag for the batch job, with
//            NO LLM call anywhere in the write path.
//
// DB-backed (disposable :5544 Postgres), Voyage fully MOCKED at module level with
// call counters — NO live quota. Skipped gracefully without DATABASE_URL.
//
// Wave-3 delta (retrieval-token-efficiency plan): the summarizer is mocked with a
// call counter too — the exact-dup short-circuit must skip the summarize spend as
// well as the embed spend (dup check runs BEFORE the Promise.all in storeMemory).

import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Hoisted so the vi.mock factory (hoisted above imports) can reference them.
const H = vi.hoisted(() => {
  const counters = { embed: 0, embedContextualSingle: 0, summarize: 0 };
  /** 1024-dim vector with the given (index, value) components; zeros elsewhere. */
  function sparse(components: Array<[number, number]>): number[] {
    const v = new Array<number>(1024).fill(0);
    for (const [i, x] of components) v[i] = x;
    return v;
  }
  // cos(near, unit0) = 0.98 (> 0.9 → flagged); cos(near, mid) ≈ 0.828 (< 0.9).
  const unit0 = sparse([[0, 1]]);
  const mid = sparse([
    [0, 0.7],
    [1, 0.7141],
  ]);
  const near = sparse([
    [0, 0.98],
    [1, 0.199],
  ]);
  const ortho = sparse([[2, 1]]);
  /** The mocked contextual embedder picks the vector from a marker in the text. */
  function vecForText(text: string): number[] {
    if (text.includes("NEARDUP")) return near;
    if (text.includes("ORTHO")) return ortho;
    return sparse([[3, 1]]);
  }
  return { counters, sparse, unit0, mid, near, ortho, vecForText };
});

vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => {
    H.counters.embed += texts.length;
    return texts.map(() => H.sparse([[4, 1]]));
  },
  embedContextualSingle: async (texts: string[]) => {
    H.counters.embedContextualSingle += texts.length;
    return texts.map((t) => H.vecForText(t));
  },
  rerank: async (_q: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_, i) => ({ index: i, score: 1 - i * 0.01 })),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

// Counter-only summarizer mock: resolves null (same as the closed SUMMARIZE_ON_STORE
// gate) so every non-dup insert still writes summary = NULL — behavior unchanged,
// invocations countable.
vi.mock("../src/summarize.js", () => ({
  summarizeMemory: async () => {
    H.counters.summarize += 1;
    return null;
  },
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  DUP_CANDIDATE_TOP_S,
  DUP_COSINE_THRESHOLD,
  contentSha256,
  storeMemory,
  updateMemory,
  type DupCandidate,
} from "../src/memory.js";

const here = dirname(fileURLToPath(import.meta.url));

const importerSha256 = (s: string) =>
  createHash("sha256").update(s).digest("hex");

describe.skipIf(skip)("storeMemory write-path dedup (AC-301/AC-302)", () => {
  let pool: pg.Pool;
  const PROJ = `store-dedup-${Date.now()}`;
  const OTHER_PROJ = `${PROJ}-other`;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Apply the migration chain idempotently (HOLD files skipped) so the suite is
    // self-sufficient on a fresh disposable DB (golden-suite pattern).
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
    await pool.query(
      `DELETE FROM memory.memories WHERE project_id IN ($1, $2)`,
      [PROJ, OTHER_PROJ],
    );
    await pool.end();
  });

  it("populates content_sha256 on insert with the importers' exact recipe", async () => {
    const content = "Postgres RLS isolates tenants per organization.";
    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "RLS tenancy",
      content,
    });
    expect(r.duplicate).toBeUndefined();
    const { rows } = await pool.query<{ content_sha256: string }>(
      `SELECT content_sha256 FROM memory.memories WHERE id = $1`,
      [r.id],
    );
    // Fixture cross-check: the importer scripts hash CONTENT ONLY
    // (services/importer/distill-transcripts.mjs — sha256(content), hex).
    expect(rows[0].content_sha256).toBe(importerSha256(content));
    expect(rows[0].content_sha256).toBe(contentSha256(content));
  });

  it("same content twice → ONE row, second call returns the first id, duplicate: true, ZERO embed + ZERO summarize calls", async () => {
    const content = "The disposable eval Postgres listens on port 5544.";
    const first = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "Eval DB port",
      content,
    });
    const before = { ...H.counters };
    // Mock-wiring guard: the non-dup store above MUST have gone through the mocked
    // summarizer — otherwise the zero-call assertion below would pass vacuously.
    expect(before.summarize).toBeGreaterThan(0);

    // Different TITLE on purpose: the dedup key is content-only (importer recipe).
    const second = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "A retitled restatement",
      content,
    });

    expect(second).toEqual({
      id: first.id,
      title: first.title,
      duplicate: true,
    });
    // No embed quota spent on the dup path — neither endpoint was called again —
    // and the short-circuit fires BEFORE the Promise.all, so the summarizer was
    // never invoked either (wave-3 delta: zero LLM spend on the dup path).
    expect(H.counters.embed).toBe(before.embed);
    expect(H.counters.embedContextualSingle).toBe(before.embedContextualSingle);
    expect(H.counters.summarize).toBe(before.summarize);

    const { rows } = await pool.query<{ n: string }>(
      `SELECT count(*) AS n FROM memory.memories
       WHERE project_id = $1 AND content_sha256 = $2`,
      [PROJ, contentSha256(content)],
    );
    expect(Number(rows[0].n)).toBe(1);
  });

  it("the dup check is scoped to the project — same content in another project inserts", async () => {
    const content = "The disposable eval Postgres listens on port 5544.";
    const r = await storeMemory({
      projectId: OTHER_PROJ,
      type: "semantic",
      title: "Eval DB port",
      content,
    });
    expect(r.duplicate).toBeUndefined();
    const { rows } = await pool.query<{ n: string }>(
      `SELECT count(*) AS n FROM memory.memories WHERE project_id = $1`,
      [OTHER_PROJ],
    );
    expect(Number(rows[0].n)).toBe(1);
  });

  it("a superseded/archived same-hash row does NOT block a re-store", async () => {
    const content = "A retired fact that comes back.";
    await pool.query(
      `INSERT INTO memory.memories
         (project_id, type, title, content, status, content_sha256)
       VALUES ($1, 'semantic', 'retired', $2, 'superseded', $3),
              ($1, 'semantic', 'archived twin', $2, 'active', $3)`,
      [PROJ, content, contentSha256(content)],
    );
    // Make the 'active' twin archived (archived_at set) so BOTH seeded rows are
    // non-candidates: one by status, one by archived_at.
    await pool.query(
      `UPDATE memory.memories SET archived_at = now()
       WHERE project_id = $1 AND title = 'archived twin'`,
      [PROJ],
    );
    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "restored",
      content,
    });
    expect(r.duplicate).toBeUndefined();
    const { rows } = await pool.query<{ status: string }>(
      `SELECT COALESCE(status,'active') AS status FROM memory.memories WHERE id = $1`,
      [r.id],
    );
    expect(rows[0].status).toBe("active");
  });

  it("flags >0.9-cosine neighbors into metadata.dupCandidates; sub-threshold neighbors stay out (AC-302)", async () => {
    // Seed neighbors directly with literal vectors: N1 at unit0 (cos 0.98 to the new
    // row), N2 mid-similarity (cos ≈ 0.828 to the new row — below threshold).
    const n1 = await pool.query<{ id: string }>(
      `INSERT INTO memory.memories (project_id, type, title, content, embedding_v2)
       VALUES ($1, 'semantic', 'anchor', 'anchor content', $2::halfvec)
       RETURNING id`,
      [PROJ, `[${H.unit0.join(",")}]`],
    );
    await pool.query(
      `INSERT INTO memory.memories (project_id, type, title, content, embedding_v2)
       VALUES ($1, 'semantic', 'mid neighbor', 'mid content', $2::halfvec)`,
      [PROJ, `[${H.mid.join(",")}]`],
    );

    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "near dup",
      content: "NEARDUP of the anchor row.",
      metadata: { tags: ["kept"] },
    });

    const { rows } = await pool.query<{
      metadata: {
        tags?: string[];
        dupCandidates?: DupCandidate[];
        dupFlaggedAt?: string;
      };
    }>(`SELECT metadata FROM memory.memories WHERE id = $1`, [r.id]);
    const meta = rows[0].metadata;
    expect(meta.tags).toEqual(["kept"]); // caller metadata survives the merge
    expect(meta.dupCandidates).toBeDefined();
    const ids = meta.dupCandidates!.map((c) => c.id);
    expect(ids).toContain(n1.rows[0].id);
    for (const c of meta.dupCandidates!) {
      expect(c.sim).toBeGreaterThan(DUP_COSINE_THRESHOLD);
    }
    expect(meta.dupCandidates!.length).toBeLessThanOrEqual(DUP_CANDIDATE_TOP_S);
    expect(typeof meta.dupFlaggedAt).toBe("string");
  });

  it("no dupCandidates key at all when nothing clears the threshold", async () => {
    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "orthogonal",
      content: "ORTHO row unlike everything else.",
    });
    const { rows } = await pool.query<{
      metadata: Record<string, unknown>;
    }>(`SELECT metadata FROM memory.memories WHERE id = $1`, [r.id]);
    expect(rows[0].metadata.dupCandidates).toBeUndefined();
    expect(rows[0].metadata.dupFlaggedAt).toBeUndefined();
  });

  it("updateMemory refreshes content_sha256 when content changes (and only then)", async () => {
    const content = "Original content before the edit.";
    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "editable",
      content,
    });

    // Title-only edit: hash must still match the ORIGINAL content.
    await updateMemory(r.id, { title: "renamed" });
    let row = await pool.query<{ content_sha256: string }>(
      `SELECT content_sha256 FROM memory.memories WHERE id = $1`,
      [r.id],
    );
    expect(row.rows[0].content_sha256).toBe(contentSha256(content));

    // Content edit: hash follows the new content (no stale dedup key).
    const edited = "Edited content after the edit.";
    await updateMemory(r.id, { content: edited });
    row = await pool.query(
      `SELECT content_sha256 FROM memory.memories WHERE id = $1`,
      [r.id],
    );
    expect(row.rows[0].content_sha256).toBe(contentSha256(edited));
  });

  it("a decision store NEVER short-circuits — duplicate content still inserts and flips its supersedesId target", async () => {
    const content = "Decision: rerank-2.5 is the pinned reranker.";
    // An active NON-decision row already carries the exact hash.
    const plain = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "plain twin",
      content,
    });
    expect(plain.duplicate).toBeUndefined();

    const prior = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "old decision",
      content: "Decision: rerank-2.5-lite is the pinned reranker.",
      sourceKind: "decision",
    });

    // The decision re-assertion byte-matches the plain row — it must STILL insert
    // and still run its user-authored supersession bookkeeping (AC-041), never
    // silently return duplicate:true.
    const d = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "new decision",
      content,
      sourceKind: "decision",
      supersedesId: prior.id,
    });
    expect(d.duplicate).toBeUndefined();
    expect(d.id).not.toBe(plain.id);

    const flipped = await pool.query<{
      decision_status: string;
      superseded_by: string;
    }>(
      `SELECT decision_status, superseded_by FROM memory.memories WHERE id = $1`,
      [prior.id],
    );
    expect(flipped.rows[0].decision_status).toBe("superseded");
    expect(flipped.rows[0].superseded_by).toBe(d.id);
  });

  it("updateMemory clears stale dupCandidates flags on content edits (title edits keep them)", async () => {
    // Anchor at unit0, then a NEARDUP store that flags it (same geometry as the
    // AC-302 test above).
    await pool.query(
      `INSERT INTO memory.memories (project_id, type, title, content, embedding_v2)
       VALUES ($1, 'semantic', 'flag anchor', 'flag anchor content', $2::halfvec)`,
      [PROJ, `[${H.unit0.join(",")}]`],
    );
    const r = await storeMemory({
      projectId: PROJ,
      type: "semantic",
      title: "flagged row",
      content: "NEARDUP of the flag anchor.",
    });
    const meta = async () =>
      (
        await pool.query<{ metadata: Record<string, unknown> }>(
          `SELECT metadata FROM memory.memories WHERE id = $1`,
          [r.id],
        )
      ).rows[0].metadata;
    expect((await meta()).dupCandidates).toBeDefined();

    // Title-only edit: the write-time flags still describe the content — kept.
    await updateMemory(r.id, { title: "flagged row, renamed" });
    expect((await meta()).dupCandidates).toBeDefined();

    // Content edit: the recorded sims describe the OLD content — cleared, so the
    // consolidation judge is never fed stale write-time similarity.
    await updateMemory(r.id, { content: "Entirely different content now." });
    const after = await meta();
    expect(after.dupCandidates).toBeUndefined();
    expect(after.dupFlaggedAt).toBeUndefined();
  });
});

// Pure guards — always run, no DB.
describe("write-path dedup (pure)", () => {
  it("contentSha256 is the importer recipe: sha256 hex over content only", () => {
    expect(contentSha256("abc")).toBe(importerSha256("abc"));
    expect(contentSha256("abc")).toMatch(/^[0-9a-f]{64}$/);
  });

  it("src/memory.ts never imports the LLM client — the write path stays LLM-free (AC-302)", () => {
    const src = readFileSync(join(here, "..", "src", "memory.ts"), "utf8");
    expect(src).not.toMatch(/from\s+["'][^"']*llm\.js["']/);
  });

  it("threshold constant is the program's 0.90 lock", () => {
    expect(DUP_COSINE_THRESHOLD).toBe(0.9);
  });
});
