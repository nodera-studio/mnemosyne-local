// Golden-output pin for searchMemory (AC-101). Recorded BEFORE the two-phase
// fuseCandidates/rerankAndBlend split so the split can prove "no behavior change":
// the exact ordered id arrays below MUST stay green, unchanged, after the refactor.
//
// Everything is deterministic: Voyage is module-mocked (query vectors + rerank scores
// derived from seeded PRNGs over the input strings), fixture embedding_v2 vectors are
// inserted as literals, created_at/importance are fixed, and Date is faked to a fixed
// instant so the recency term of the blend is exact. With ≤10 rows Postgres seq-scans,
// and the fixture vectors produce strictly distinct cosine orderings, so HNSW-vs-seqscan
// plan changes cannot flip ranks.
//
// DB-backed (disposable :5544 Postgres); skipped gracefully without DATABASE_URL.
// NO live Voyage quota is ever burned here.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic helpers, hoisted so the vi.mock factory (which is hoisted above all
// imports) can reference them too.
const H = vi.hoisted(() => {
  /** FNV-1a 32-bit string hash. */
  function fnv1a(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }
  /** Tiny seeded PRNG (same algorithm the eval math layer uses). */
  function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  /** Deterministic 1024-dim unit-range vector for a string key (3-decimal values so
   *  the halfvec quantization is stable and the SQL literals stay small). */
  function fakeVec(key: string, dim = 1024): number[] {
    const rnd = mulberry32(fnv1a(key));
    return Array.from({ length: dim }, () =>
      Number((rnd() * 2 - 1).toFixed(3)),
    );
  }
  /** Deterministic rerank: score from (query, doc length, index) — strictly distinct
   *  with overwhelming probability — returned sorted desc like the real Voyage API. */
  function fakeRerank(
    query: string,
    docs: string[],
    topK: number,
  ): Array<{ index: number; score: number }> {
    const scored = docs.map((d, i) => ({
      index: i,
      score: Number(
        (0.05 + 0.9 * mulberry32(fnv1a(`${query}|${d.length}|${i}`))()).toFixed(
          6,
        ),
      ),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, Math.min(topK, docs.length));
  }
  return { fnv1a, mulberry32, fakeVec, fakeRerank };
});

// Module-mock Voyage: searchMemory imports voyage.js directly, so the whole module is
// replaced (deterministic, network-free). toVectorLiteral mirrors the real one.
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map((t) => H.fakeVec(`legacy:${t}`)),
  embedContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(t))),
  embedContextualSingle: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(t)),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import { searchMemory, type MemoryType } from "../src/memory.js";

const here = dirname(fileURLToPath(import.meta.url));

const PROJ = "golden-mem";
const EMPTY_PROJ = "golden-mem-empty";
// Fixed instant AFTER every fixture date; Date is faked to it so the recency term is
// exact and the pinned ordering can never drift with wall-clock time.
const NOW = new Date("2026-07-01T12:00:00Z");

const id = (n: number) => `00000000-0000-4000-8000-00000000000${n}`;

interface Fixture {
  id: string;
  type: MemoryType;
  title: string;
  content: string;
  importance: number;
  createdAt: string;
  eventDate: string | null;
}

// 8 fixtures: distinct content lengths (rerank inputs), distinct importance and
// created_at values so the recency/importance blend produces strictly ordered finals.
const FIXTURES: Fixture[] = [
  {
    id: id(1),
    type: "semantic",
    title: "Postgres pool sizing",
    content:
      "postgres connection pool sizing guidance: keep max at 10 per service, watch for pool exhaustion under load and tune idle timeout",
    importance: 0.9,
    createdAt: "2026-06-10T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(2),
    type: "procedural",
    title: "Deploy rollback runbook",
    content:
      "rollback procedure for a bad deployment: redeploy the previous image tag, verify health checks, then announce in the channel",
    importance: 0.7,
    createdAt: "2026-06-28T00:00:00Z",
    eventDate: "2026-06-30T00:00:00Z",
  },
  {
    id: id(3),
    type: "episodic",
    title: "Incident: connection storm",
    content:
      "incident report: postgres connections exhausted after a deploy loop opened a new pool per request; fixed by sharing the pool singleton",
    importance: 0.6,
    createdAt: "2026-06-25T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(4),
    type: "semantic",
    title: "RRF fusion notes",
    content:
      "hybrid search fuses bm25 and vector ranks with reciprocal rank fusion; k of 60 flattens the tail so neither arm dominates the pool",
    importance: 0.8,
    createdAt: "2026-06-20T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(5),
    type: "semantic",
    title: "Voyage embedding models",
    content:
      "voyage-context-4 embeds each chunk in the context of its document and is the corpus embedder for memory",
    importance: 0.4,
    createdAt: "2026-06-27T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(6),
    type: "procedural",
    title: "Backup restore procedure",
    content:
      "restore the nightly backup: stop writers, pg_restore into a scratch database, verify counts, then swap the connection string over",
    importance: 0.55,
    createdAt: "2026-06-18T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(7),
    type: "episodic",
    title: "Migration 005 shipped",
    content:
      "the decision log migration landed as 005 after the wave-p renumbering; the batch runner applies it in filename order",
    importance: 0.3,
    createdAt: "2026-06-29T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(8),
    type: "semantic",
    title: "HNSW index tuning",
    content:
      "hnsw index tuning: ef_search trades recall for latency; rebuild concurrently after mass re-embeds so the graph stays dense",
    importance: 0.65,
    createdAt: "2026-06-15T00:00:00Z",
    eventDate: null,
  },
];

const byId = new Map(FIXTURES.map((f) => [f.id, f]));

function iso(s: string): string {
  return new Date(s).toISOString();
}

/** The deterministic FALLBACK snippet (180 chars, whitespace-collapsed prefix) —
 *  what a hit gets when the query does not lexically match its content (AC-802). */
function expectedSnippet(fixtureId: string): string {
  return byId.get(fixtureId)!.content.replace(/\s+/g, " ").slice(0, 180);
}

/** Pinned snippet for a hit of a pinned query: the recorded ts_headline when the
 *  query matches lexically, the 180-char prefix otherwise. */
function expectedQuerySnippet(
  queryLabel: QueryLabel,
  fixtureId: string,
): string {
  return GOLDEN_SNIPPETS[queryLabel][fixtureId] ?? expectedSnippet(fixtureId);
}

describe.skipIf(skip)("searchMemory golden pins (AC-101)", () => {
  let pool: pg.Pool;

  beforeAll(async () => {
    vi.useFakeTimers({ toFake: ["Date"], now: NOW });
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Apply the migration chain idempotently (HOLD files skipped) so the test is
    // self-sufficient on a fresh disposable DB — same pattern as codebase-mcp tests.
    const sqlDir = join(here, "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    for (const f of FIXTURES) {
      await pool.query(
        `INSERT INTO memory.memories
           (id, project_id, type, title, content, importance, created_at, event_date, embedding_v2)
         VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6, $7, $8, $9::halfvec)`,
        [
          f.id,
          PROJ,
          f.type,
          f.title,
          f.content,
          f.importance,
          f.createdAt,
          f.eventDate,
          `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
        ],
      );
    }
  });

  afterAll(async () => {
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
    vi.useRealTimers();
  });

  /** Shared shape assertions for every pinned query. */
  function assertShape(
    hits: Awaited<ReturnType<typeof searchMemory>>["hits"],
    limit: number,
    queryLabel: QueryLabel,
  ): void {
    expect(hits.length).toBeLessThanOrEqual(limit);
    for (const h of hits) {
      expect(h.snippet).toBe(expectedQuerySnippet(queryLabel, h.id));
      // Headline snippets are word-bounded (MaxWords), not char-bounded — the
      // 180-char cap applies only to the prefix-fallback form.
      if (!(h.id in GOLDEN_SNIPPETS[queryLabel])) {
        expect(h.snippet.length).toBeLessThanOrEqual(180);
      }
      expect(h.title).toBe(byId.get(h.id)!.title);
      expect(h.type).toBe(byId.get(h.id)!.type);
      expect(h.importance).toBeCloseTo(byId.get(h.id)!.importance, 6);
      expect("score" in h).toBe(false);
      expect(h.createdAt).toBe(iso(byId.get(h.id)!.createdAt));
      const expectedEvent = byId.get(h.id)!.eventDate;
      expect(h.eventDate).toBe(expectedEvent ? iso(expectedEvent) : null);
      expect(h.effectiveDate).toBe(h.eventDate ?? h.createdAt);
      expect(h.status).toBe("active");
      // Wave-2: fixtures are never summarized — pins the NULL pass-through.
      expect(h.summary).toBeNull();
    }
  }

  it("pins query 1 (plain): 'postgres connection pool'", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "postgres connection pool",
      limit: 5,
    });
    const { hits } = result;
    assertShape(hits, 5, "q1");
    expect(hits.map((h) => h.id)).toEqual(GOLDEN_Q1);
  });

  it("pins query 2 (plain): 'rollback deployment procedure'", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "rollback deployment procedure",
      limit: 5,
    });
    const { hits } = result;
    assertShape(hits, 5, "q2");
    expect(hits.map((h) => h.id)).toEqual(GOLDEN_Q2);
  });

  it("pins query 3 (type-filtered semantic): 'embedding index tuning'", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "embedding index tuning",
      type: "semantic",
      limit: 5,
    });
    const { hits } = result;
    assertShape(hits, 5, "q3");
    for (const h of hits) expect(h.type).toBe("semantic");
    expect(hits.map((h) => h.id)).toEqual(GOLDEN_Q3);
  });

  it("pins the empty-corpus early return: []", async () => {
    const result = await searchMemory({
      projectId: EMPTY_PROJ,
      query: "anything at all",
      limit: 5,
    });
    expect(result.hits).toEqual([]);
  });
});

// ── Golden id orders (recorded 2026-07-03 against the pre-split searchMemory) ────────
// Do NOT regenerate casually: these pins are the AC-101 no-behavior-change guard for
// the fuseCandidates/rerankAndBlend split. If a deliberate behavior change lands later
// (e.g. wave-4 token shaping), update them in the SAME commit as the change.
const GOLDEN_Q1: string[] = [id(5), id(6), id(7), id(1), id(2)];
const GOLDEN_Q2: string[] = [id(2), id(7), id(5), id(4), id(1)];
// Only 4 of the 8 fixtures are type=semantic, so the filtered pin has 4 hits.
const GOLDEN_Q3: string[] = [id(4), id(5), id(8), id(1)];

// ── Golden query-aware snippets (recorded 2026-07-04 on pgvector/pgvector:pg17 —
// the SAME Postgres major the live stack runs, so ts_headline output is identical).
// Only hits whose content lexically matches the query get a `**`-marked ts_headline
// fragment (AC-801); every other hit keeps the deterministic 180-char prefix and is
// deliberately ABSENT from this map (AC-802 — expectedQuerySnippet falls back).
// ts_headline is deterministic given (config, text, query, options), so these are
// stable pins, re-recordable only on a Postgres major bump.
type QueryLabel = "q1" | "q2" | "q3";
const GOLDEN_SNIPPETS: Record<QueryLabel, Record<string, string>> = {
  q1: {
    [id(1)]:
      "**postgres** **connection** **pool** sizing guidance: keep max at 10 per service, watch for **pool** exhaustion under load and tune idle",
  },
  q2: {
    [id(2)]:
      "**rollback** **procedure** for a bad **deployment**: redeploy the previous image tag, verify health checks, then announce in the channel",
  },
  q3: {
    [id(8)]: "hnsw **index** **tuning**: ef_search trades recall for",
  },
};
