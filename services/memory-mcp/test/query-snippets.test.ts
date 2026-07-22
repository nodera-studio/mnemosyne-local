// Wave 1 (retrieval token efficiency) — query-aware snippet behavior. Covers:
//   AC-801 — lexical match → ts_headline snippet with ** markers (private-use
//            sentinels internally, rendered as ** for display — sentinels never
//            leak); two match regions → the " … " fragment delimiter; whitespace
//            collapsed
//   AC-802 — vector-only hit / stop-words-only query / headline-SQL failure →
//            the deterministic 180-char prefix, and the search never rejects;
//            literal ** markdown in content must NOT masquerade as a match
//   AC-803 — getRecent issues NO ts_headline SQL and keeps prefix snippets
//
// Deterministic: Voyage is module-mocked (seeded PRNG vectors/rerank), fixture
// embedding_v2 vectors are inserted as literals. Headlines themselves are pure
// Postgres (ts_headline is deterministic given config/text/query/options).
// DB-backed (disposable :5544); skipped without DATABASE_URL. NO live quota.

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const H = vi.hoisted(() => {
  function fnv1a(s: string): number {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return h >>> 0;
  }
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
  function fakeVec(key: string, dim = 1024): number[] {
    const rnd = mulberry32(fnv1a(key));
    return Array.from({ length: dim }, () =>
      Number((rnd() * 2 - 1).toFixed(3)),
    );
  }
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

import {
  applyQuerySnippets,
  getRecent,
  searchMemory,
  type MemoryType,
} from "../src/memory.js";
import { pool as livePool } from "../src/db/pool.js";

const here = dirname(fileURLToPath(import.meta.url));

const PROJ = "w1-query-snips";

const id = (n: number) => `30000000-0000-4000-8000-00000000000${n}`;

interface Fixture {
  id: string;
  type: MemoryType;
  title: string;
  content: string;
}

// Fixture 1: the query lexemes appear in TWO regions separated by >20 words of
// unrelated filler, so MaxFragments=2 yields two fragments joined by " … ".
// Fixture 2: the match region carries raw newlines + runs of spaces, so the
// whitespace-collapse of the headline is exercised for real.
// Fixture 3: shares no lexeme with any query below — reachable only through the
// vector arm, i.e. the AC-802 prefix-fallback hit.
// Fixture 4: literal `**` markdown in the opening words and no lexeme shared with
// any query — ts_headline's no-match fallback headline for it CONTAINS `**`, so a
// naive `includes("**")` match check false-positives (the sentinel regression pin).
const FIXTURES: Fixture[] = [
  {
    id: id(1),
    type: "semantic",
    title: "Vector index rebuild cadence",
    content:
      "vector index rebuild after a mass re-embed keeps the hnsw graph dense and the recall stable. " +
      "meanwhile the unrelated middle section talks about queue depth alerts, retry budgets, connection churn, " +
      "dashboard tiles, paging policies, weekly reports, capacity reviews, on-call handoffs, and the long tail of " +
      "operational chores that fill a sprint. schedule the vector index rebuild nightly so the graph stays balanced.",
  },
  {
    id: id(2),
    type: "procedural",
    title: "Ingress gateway timeouts",
    content:
      "kubernetes ingress gateway\n  timeout tuning:   raise the upstream\n\nread timeout before the retry storm hits the backend pods",
  },
  {
    id: id(3),
    type: "episodic",
    title: "Saga compensation order",
    content:
      "saga compensation steps unwind bookings in reverse order when a leg fails midway",
  },
  {
    id: id(4),
    type: "semantic",
    title: "Markdown emphasis note",
    content:
      "release notes keep **bold** emphasis markers and **callout** styling in the opening words so rendered changelogs stay readable. " +
      "a trailing filler sentence pads this fixture comfortably past one hundred eighty characters to keep the raw body strictly longer than the collapsed prefix form.",
  },
];

const byId = new Map(FIXTURES.map((f) => [f.id, f]));

/** The deterministic prefix fallback (AC-802). */
function prefixOf(fixtureId: string): string {
  return byId.get(fixtureId)!.content.replace(/\s+/g, " ").slice(0, 180);
}

describe.skipIf(skip)("query-aware snippets (AC-801/802/803)", () => {
  let pool: pg.Pool;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
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
           (id, project_id, type, title, content, embedding_v2)
         VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6::halfvec)`,
        [
          f.id,
          PROJ,
          f.type,
          f.title,
          f.content,
          `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
        ],
      );
    }
  });

  afterAll(async () => {
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
  });

  it("AC-801: lexical query → **-marked fragments, two regions joined by the … delimiter", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "vector index rebuild",
      limit: 10,
    });
    const hit = result.hits.find((h) => h.id === id(1))!;
    expect(hit).toBeTruthy();
    expect(hit.snippet).toContain("**vector**");
    expect(hit.snippet).toContain("**rebuild**");
    expect(hit.snippet).toContain(" … ");
    expect(hit.snippet).not.toBe(prefixOf(id(1)));
    expect(hit.snippet).not.toMatch(/\n| {2}/);
    // The private-use match sentinels are internal only — never in display output.
    expect(hit.snippet).not.toMatch(/[\uE000\uE001]/);
  });

  it("AC-801: headline whitespace is collapsed (newlines / space runs in the match region)", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "ingress gateway timeout",
      limit: 10,
    });
    const hit = result.hits.find((h) => h.id === id(2))!;
    expect(hit).toBeTruthy();
    expect(hit.snippet).toContain("**ingress**");
    expect(hit.snippet).not.toMatch(/\n| {2}/);
  });

  it("AC-802: vector-only hit (no shared lexeme) keeps the 180-char prefix", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "vector index rebuild",
      limit: 10,
    });
    const hit = result.hits.find((h) => h.id === id(3))!;
    expect(hit).toBeTruthy();
    expect(hit.snippet).toBe(prefixOf(id(3)));
  });

  it("AC-802 regression: literal ** markdown in content must NOT masquerade as a lexical match", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "vector index rebuild",
      limit: 10,
    });
    // Fixture 4 shares no lexeme with the query, so its headline is the no-match
    // document-prefix fallback — which CONTAINS the row's own literal `**`. Only
    // sentinel presence may accept a headline: the 180-char prefix must survive.
    const hit = result.hits.find((h) => h.id === id(4))!;
    expect(hit).toBeTruthy();
    expect(hit.snippet).toBe(prefixOf(id(4)));
    expect(hit.snippet).toContain("**bold**"); // the prefix keeps the raw markdown
  });

  it("AC-802: stop-words-only query → every snippet is the prefix", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "the of and",
      limit: 10,
    });
    expect(result.hits.length).toBeGreaterThan(0);
    for (const h of result.hits) expect(h.snippet).toBe(prefixOf(h.id));
  });

  it("AC-802: headline SQL failure → search still succeeds with prefix snippets", async () => {
    const orig = livePool.query.bind(livePool) as (
      ...args: unknown[]
    ) => unknown;
    const spy = vi
      .spyOn(livePool as unknown as Record<"query", typeof orig>, "query")
      .mockImplementation((...args: unknown[]) => {
        if (typeof args[0] === "string" && args[0].includes("ts_headline")) {
          throw new Error("headline SQL down");
        }
        return orig(...args);
      });
    try {
      const result = await searchMemory({
        projectId: PROJ,
        query: "vector index rebuild",
        limit: 10,
      });
      expect(result.hits.length).toBeGreaterThan(0);
      for (const h of result.hits) expect(h.snippet).toBe(prefixOf(h.id));
    } finally {
      spy.mockRestore();
    }
  });

  it("AC-803: getRecent issues NO ts_headline SQL and keeps prefix snippets", async () => {
    const orig = livePool.query.bind(livePool) as (
      ...args: unknown[]
    ) => unknown;
    const texts: string[] = [];
    const spy = vi
      .spyOn(livePool as unknown as Record<"query", typeof orig>, "query")
      .mockImplementation((...args: unknown[]) => {
        if (typeof args[0] === "string") texts.push(args[0]);
        return orig(...args);
      });
    try {
      const hits = await getRecent({ projectId: PROJ, limit: 10 });
      expect(hits.length).toBe(FIXTURES.length);
      for (const h of hits) expect(h.snippet).toBe(prefixOf(h.id));
      expect(texts.some((t) => t.includes("ts_headline"))).toBe(false);
    } finally {
      spy.mockRestore();
    }
  });

  it("applyQuerySnippets early-returns on zero hits without issuing SQL", async () => {
    const orig = livePool.query.bind(livePool) as (
      ...args: unknown[]
    ) => unknown;
    const spy = vi
      .spyOn(livePool as unknown as Record<"query", typeof orig>, "query")
      .mockImplementation((...args: unknown[]) => orig(...args));
    try {
      await expect(applyQuerySnippets("anything", [])).resolves.toEqual([]);
      // Filtered to headline SQL: a late fire-and-forget search_log insert from
      // an earlier test may legitimately land on this spy.
      const texts = spy.mock.calls.map((c) =>
        typeof c[0] === "string" ? c[0] : "",
      );
      expect(texts.some((t) => t.includes("ts_headline"))).toBe(false);
    } finally {
      spy.mockRestore();
    }
  });
});
