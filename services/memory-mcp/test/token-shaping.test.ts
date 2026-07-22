// Wave 4 (WS2) — token shaping, memory side. Covers:
//   AC-201 — tags/after applied as WHERE clauses in BOTH RRF arms of fuseCandidates
//   AC-202 — zero-pool filtered search retries once unfiltered + prepends a notice
//   AC-203 — formatted output is rank-only (no raw relevance/blend scores)
//   AC-204 — truncation appends the steering line naming optional filters
//   AC-207 — public hits carry createdAt/eventDate/effectiveDate/status; NULL
//            event_date falls back to created_at; getRecent has field parity
//   AC-107/wave-4 — search_log.filters records tags/after (+ retried)
//
// Deterministic: Voyage module-mocked (seeded PRNG vectors/rerank), fixtures with
// literal embedding_v2 vectors. DB-backed (disposable :5544); skipped without
// DATABASE_URL. NO live Voyage quota is ever burned here.

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
  // Probe for the query-embed boundary: `calls` counts embedContextualSingle
  // invocations (MEDIUM-001 — a retried search must embed exactly once); `delayMs`
  // slows the embed so timing assertions can tell the retried pool window from the
  // first-attempt window (LOW-006).
  const embedProbe = { calls: 0, delayMs: 0 };
  return { fnv1a, mulberry32, fakeVec, fakeRerank, embedProbe };
});

vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map((t) => H.fakeVec(`legacy:${t}`)),
  embedContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(t))),
  embedContextualSingle: async (texts: string[]) => {
    H.embedProbe.calls += 1;
    if (H.embedProbe.delayMs > 0) {
      await new Promise((r) => setTimeout(r, H.embedProbe.delayMs));
    }
    return texts.map((t) => H.fakeVec(t));
  },
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  budgetMemoryBody,
  formatHits,
  fuseCandidates,
  getRecent,
  searchMemory,
  MEMORY_GET_DEFAULT_MAX_CHARS,
  type MemoryType,
} from "../src/memory.js";
import { TOOL_DESCRIPTIONS, buildServer } from "../src/server.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

const here = dirname(fileURLToPath(import.meta.url));

const PROJ = "ws2-mem-shaping";
const EMPTY_PROJ = "ws2-mem-shaping-empty";

const id = (n: number) => `20000000-0000-4000-8000-00000000000${n}`;

interface Fixture {
  id: string;
  type: MemoryType;
  title: string;
  content: string;
  importance: number;
  tags: string[];
  createdAt: string;
  eventDate: string | null;
}

// 8 rows sharing the token "gateway" so an unfiltered query pools all of them.
// Content deliberately avoids the substrings the AC-203 guard regexes hunt for.
const FIXTURES: Fixture[] = [
  {
    id: id(1),
    type: "semantic",
    title: "Ingress routing rules",
    content:
      "ingress controller routing rules for the cluster gateway and its default backend",
    importance: 0.9,
    tags: ["infra", "network"],
    createdAt: "2026-05-01T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(2),
    type: "procedural",
    title: "Gateway cert rotation",
    content:
      "rotate the gateway certificates: renew the issuer, roll the pods, verify the handshake",
    importance: 0.7,
    tags: ["infra"],
    createdAt: "2026-06-01T00:00:00Z",
    eventDate: "2026-06-20T00:00:00Z",
  },
  {
    id: id(3),
    type: "semantic",
    title: "Gateway retrieval notes",
    content:
      "the retrieval pipeline treats the gateway service as one indexing unit with a single owner",
    importance: 0.6,
    tags: ["retrieval"],
    createdAt: "2026-06-25T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(4),
    type: "episodic",
    title: "Gateway outage postmortem",
    content:
      "the march gateway outage traced back to an expired certificate on the internal listener",
    importance: 0.8,
    tags: [],
    createdAt: "2026-03-01T00:00:00Z",
    eventDate: "2026-03-10T00:00:00Z",
  },
  {
    id: id(5),
    type: "semantic",
    title: "Gateway rate limits",
    content:
      "gateway rate limiting uses a token bucket per client with a shared burst allowance",
    importance: 0.5,
    tags: ["network"],
    createdAt: "2026-06-10T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(6),
    type: "procedural",
    title: "Gateway failover drill",
    content:
      "quarterly gateway failover drill: drain the primary, promote the standby, watch the probes",
    importance: 0.4,
    tags: [],
    createdAt: "2026-04-15T00:00:00Z",
    eventDate: null,
  },
  {
    id: id(7),
    type: "episodic",
    title: "Gateway upgrade shipped",
    content:
      "the gateway upgrade to the newer proxy release shipped without a rollback window",
    importance: 0.3,
    tags: ["infra"],
    createdAt: "2026-06-28T00:00:00Z",
    eventDate: "2026-06-29T00:00:00Z",
  },
  {
    id: id(8),
    type: "semantic",
    title: "Gateway naming convention",
    content:
      "every gateway route name follows the service dash environment pattern for discoverability",
    importance: 0.2,
    tags: [],
    createdAt: "2026-02-01T00:00:00Z",
    eventDate: null,
  },
];

const byId = new Map(FIXTURES.map((f) => [f.id, f]));
const iso = (s: string): string => new Date(s).toISOString();

async function connectClient() {
  const server = buildServer(PROJ);
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test", version: "1.0.0" });
  await Promise.all([
    client.connect(clientTransport),
    server.connect(serverTransport),
  ]);
  return client;
}

async function callTool(
  client: Awaited<ReturnType<typeof connectClient>>,
  name: string,
  args: Record<string, unknown>,
): Promise<{ text: string; isError: boolean }> {
  const res = (await client.callTool({ name, arguments: args })) as {
    isError?: boolean;
    content: { type: string; text: string }[];
  };
  return {
    text: res.content.map((c) => c.text).join("\n"),
    isError: res.isError === true,
  };
}

/** Poll until fn returns non-null (fire-and-forget log landing). */
async function until<T>(
  fn: () => Promise<T | null>,
  ms = 2000,
): Promise<T | null> {
  const t0 = Date.now();
  for (;;) {
    const v = await fn();
    if (v !== null) return v;
    if (Date.now() - t0 > ms) return null;
    await new Promise((r) => setTimeout(r, 50));
  }
}

describe("memory tool descriptions (AC-205)", () => {
  it("every description has disjoint Use when and Do NOT use clauses", () => {
    expect(Object.keys(TOOL_DESCRIPTIONS).sort()).toEqual(
      [
        "memory_decision_chain",
        "memory_delete",
        "memory_get",
        "memory_get_entity",
        "memory_get_recent",
        "memory_list",
        "memory_search",
        "memory_store",
        "memory_update",
      ].sort(),
    );
    for (const [name, description] of Object.entries(TOOL_DESCRIPTIONS)) {
      expect(description, name).toContain("Use when:");
      expect(description, name).toContain("Do NOT use");
      expect(description, name).toContain("Example:");
    }
    expect(TOOL_DESCRIPTIONS.memory_search).toContain("optionally narrowing");
    expect(TOOL_DESCRIPTIONS.memory_search).toContain("ANY-of");
  });

  it("LOW-002: type is strict scoping; tags/after are droppable refinements", async () => {
    const client = await connectClient();
    try {
      const { tools } = await client.listTools();
      const search = tools.find((t) => t.name === "memory_search")!;
      const props = (
        search.inputSchema as {
          properties: Record<string, { description?: string }>;
        }
      ).properties;
      // `type` scopes strictly: kept on the zero-result retry, never auto-dropped.
      expect(props.type.description).toContain("strict scoping");
      expect(props.type.description).toContain("never auto-dropped");
      // `tags`/`after` are the droppable refinements the retry sheds.
      for (const droppable of ["tags", "after"]) {
        expect(props[droppable].description, droppable).toContain(
          "auto-dropped (with a notice)",
        );
        expect(props[droppable].description, droppable).not.toContain(
          "strict scoping",
        );
      }
    } finally {
      await client.close();
    }
  });
});

describe("budgetMemoryBody (AC-805/806, pure)", () => {
  const row = (content: unknown): Record<string, unknown> => ({
    id: "x",
    content,
    metadata: { keep: true },
  });

  it("content exactly at the budget passes through unchanged (no truncation fields)", () => {
    const r = row("a".repeat(6000));
    const shaped = budgetMemoryBody(r, MEMORY_GET_DEFAULT_MAX_CHARS);
    expect(shaped).toBe(r);
    expect("truncated" in shaped).toBe(false);
    expect("totalChars" in shaped).toBe(false);
    expect("note" in shaped).toBe(false);
  });

  it("one char over the budget → truncated content + markers + pinned note", () => {
    const shaped = budgetMemoryBody(row("a".repeat(6001)), 6000);
    expect(shaped.truncated).toBe(true);
    expect(shaped.totalChars).toBe(6001);
    expect((shaped.content as string).length).toBe(6000);
    expect(shaped.note).toBe(
      "content truncated to 6000 of 6001 chars — call memory_get again with full=true (or maxChars=0) for the complete body",
    );
    // Only content is budgeted — structured fields stay intact.
    expect(shaped.metadata).toEqual({ keep: true });
    expect(shaped.id).toBe("x");
  });

  it("maxChars=0 means unlimited — oversized content passes through unchanged", () => {
    const r = row("a".repeat(50000));
    expect(budgetMemoryBody(r, 0)).toBe(r);
  });

  it("non-string content passes through unchanged", () => {
    const r = row(null);
    expect(budgetMemoryBody(r, 10)).toBe(r);
  });
});

describe("memory_search input validation", () => {
  it("rejects an invalid after timestamp as a tool error before search runs", async () => {
    const client = await connectClient();
    try {
      const got = await callTool(client, "memory_search", {
        query: "gateway",
        after: "not-a-date",
      });

      expect(got.isError).toBe(true);
      expect(got.text).toMatch(/invalid.*datetime|iso.*datetime/i);
    } finally {
      await client.close();
    }
  });
});

describe.skipIf(skip)("memory token shaping (AC-201/202/203/204/207)", () => {
  let pool: pg.Pool;
  let client: Awaited<ReturnType<typeof connectClient>>;

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
           (id, project_id, type, title, content, importance, metadata,
            created_at, event_date, embedding_v2)
         VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6, $7, $8, $9, $10::halfvec)`,
        [
          f.id,
          PROJ,
          f.type,
          f.title,
          f.content,
          f.importance,
          JSON.stringify({ tags: f.tags }),
          f.createdAt,
          f.eventDate,
          `[${H.fakeVec(`mem:${f.id}`).join(",")}]`,
        ],
      );
    }
    client = await connectClient();
  });

  afterAll(async () => {
    await client?.close();
    await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.query(`DELETE FROM memory.search_log WHERE project_id = $1`, [
      PROJ,
    ]);
    await pool.end();
  });

  // ── AC-201: filters are pool-level WHERE clauses in both arms ──────────────

  it("AC-201: tags filter (ANY-of) excludes untagged rows from the pool", async () => {
    const pool_ = await fuseCandidates({
      projectId: PROJ,
      query: "gateway",
      tags: ["infra"],
    });
    expect(pool_.length).toBeGreaterThan(0);
    const expected = new Set(
      FIXTURES.filter((f) => f.tags.includes("infra")).map((f) => f.id),
    );
    for (const c of pool_) expect(expected.has(c.id)).toBe(true);
    expect(new Set(pool_.map((c) => c.id))).toEqual(expected);
  });

  it("AC-201: tags is ANY-of — two tags widen to the union", async () => {
    const pool_ = await fuseCandidates({
      projectId: PROJ,
      query: "gateway",
      tags: ["infra", "retrieval"],
    });
    const expected = new Set(
      FIXTURES.filter((f) =>
        f.tags.some((t) => t === "infra" || t === "retrieval"),
      ).map((f) => f.id),
    );
    expect(new Set(pool_.map((c) => c.id))).toEqual(expected);
  });

  it("AC-201: after cuts by COALESCE(event_date, created_at)", async () => {
    const pool_ = await fuseCandidates({
      projectId: PROJ,
      query: "gateway",
      after: "2026-06-15T00:00:00Z",
    });
    const cutoff = Date.parse("2026-06-15T00:00:00Z");
    const expected = new Set(
      FIXTURES.filter(
        (f) => Date.parse(f.eventDate ?? f.createdAt) >= cutoff,
      ).map((f) => f.id),
    );
    // Sanity: the fixture set exercises both sides of the COALESCE — an old row
    // with a NULL event_date is excluded, a June event_date on an old created_at
    // is included.
    expect(expected.has(id(2))).toBe(true);
    expect(expected.has(id(1))).toBe(false);
    expect(new Set(pool_.map((c) => c.id))).toEqual(expected);
  });

  it("AC-201: tags + after + type compose", async () => {
    const pool_ = await fuseCandidates({
      projectId: PROJ,
      query: "gateway",
      type: "procedural",
      tags: ["infra"],
      after: "2026-06-15T00:00:00Z",
    });
    expect(pool_.map((c) => c.id)).toEqual([id(2)]);
  });

  // ── AC-202: zero-pool filtered search retries unfiltered with a notice ─────

  it("AC-202: impossible tag → one unfiltered retry, notice first line", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      tags: ["no-such-tag"],
      limit: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.droppedFilters).toEqual({ tags: "no-such-tag" });
    expect(result.hits.length).toBeGreaterThan(0);
    const text = formatHits(result.hits, result);
    expect(text.split("\n")[0]).toBe(
      "Note: no results matched filters {tags=no-such-tag}; showing unfiltered results. Filters are optional — retry with different values to narrow.",
    );
  });

  it("AC-202: the retry drops tags/after but KEEPS the type scoping filter", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      type: "semantic",
      tags: ["no-such-tag"],
      after: "2026-01-01T00:00:00Z",
      limit: 10,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.droppedFilters).toEqual({
      tags: "no-such-tag",
      after: "2026-01-01T00:00:00Z",
    });
    expect(result.hits.length).toBeGreaterThan(0);
    for (const h of result.hits) expect(h.type).toBe("semantic");
  });

  it("AC-202: unfiltered zero-corpus query does NOT retry (no loop)", async () => {
    const result = await searchMemory({
      projectId: EMPTY_PROJ,
      query: "anything at all",
      limit: 5,
    });
    expect(result.hits).toEqual([]);
    expect(result.retriedWithoutFilters).toBe(false);
    expect(formatHits(result.hits, result)).toBe("No matching memories.");
  });

  it("AC-202: filtered zero-corpus retry suppresses the unfiltered-results notice", async () => {
    const result = await searchMemory({
      projectId: EMPTY_PROJ,
      query: "anything at all",
      tags: ["no-such-tag"],
      limit: 5,
    });
    expect(result.hits).toEqual([]);
    expect(result.retriedWithoutFilters).toBe(true);
    expect(formatHits(result.hits, result)).toBe("No matching memories.");
  });

  it("MEDIUM-001: the zero-pool retry reuses the query embedding — exactly ONE embed call", async () => {
    H.embedProbe.calls = 0;
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      tags: ["no-such-tag"],
      limit: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(H.embedProbe.calls).toBe(1);
  });

  // ── AC-203: rank-only output ────────────────────────────────────────────────

  it("AC-203: formatted output carries ranks + ids but no raw scores", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 5,
    });
    const text = formatHits(result.hits, result);
    expect(text).toMatch(/^1\. \[/); // rank numbering present
    expect(text).toContain(result.hits[0].id); // identifiers present
    expect(text).not.toMatch(/score/i);
    expect(text).not.toMatch(/\b0\.\d{4}\b/);
    for (const h of result.hits) expect("score" in h).toBe(false);
  });

  // ── AC-204: truncation steering line ────────────────────────────────────────

  it("AC-204: limit=3 over an 8-row pool appends the steering line", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 3,
    });
    expect(result.poolSize).toBe(8);
    expect(result.truncated).toBe(true);
    const lines = formatHits(result.hits, result).split("\n");
    expect(lines[lines.length - 1]).toBe(
      "Showing top 3 of 8 candidates. Prefer several small targeted searches over one broad one — optionally narrow with type, tags, or after.",
    );
  });

  it("AC-204: limit ≥ pool size → no steering line", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 20,
    });
    expect(result.truncated).toBe(false);
    expect(formatHits(result.hits, result)).not.toContain("Showing top");
  });

  // ── AC-207: public date/status surface ──────────────────────────────────────

  it("AC-207: eventDate row → effectiveDate = eventDate; rendered in the header", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 20,
    });
    const h = result.hits.find((x) => x.id === id(2))!;
    expect(h).toBeTruthy();
    expect(h.createdAt).toBe(iso("2026-06-01T00:00:00Z"));
    expect(h.eventDate).toBe(iso("2026-06-20T00:00:00Z"));
    expect(h.effectiveDate).toBe(h.eventDate);
    expect(h.status).toBe("active");
    expect(formatHits([h])).toContain(`(2026-06-20, id: ${h.id})`);
  });

  it("LOW-001: status renders in the header ONLY when it deviates from active", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 20,
    });
    const h = result.hits.find((x) => x.id === id(2))!;
    expect(h.status).toBe("active");
    // Active (the only status live search/recent can return) stays out of the header…
    expect(formatHits([h])).not.toContain("status:");
    // …while a non-default status still renders.
    expect(formatHits([{ ...h, status: "superseded" }])).toContain(
      `(2026-06-20, status: superseded, id: ${h.id})`,
    );
  });

  it("AC-207: NULL event_date row → effectiveDate falls back to createdAt", async () => {
    const result = await searchMemory({
      projectId: PROJ,
      query: "gateway",
      limit: 20,
    });
    const h = result.hits.find((x) => x.id === id(1))!;
    expect(h).toBeTruthy();
    expect(h.eventDate).toBeNull();
    expect(h.effectiveDate).toBe(h.createdAt);
    expect(formatHits([h])).toContain("(2026-05-01, id: ");
  });

  it("AC-207: memory_get_recent exposes the identical public hit shape", async () => {
    const [search, recent] = [
      await searchMemory({ projectId: PROJ, query: "gateway", limit: 20 }),
      await getRecent({ projectId: PROJ, limit: 20 }),
    ];
    for (const probe of [id(1), id(2)]) {
      const s = search.hits.find((x) => x.id === probe)!;
      const r = recent.find((x) => x.id === probe)!;
      // Field parity holds, but snippet TEXT deliberately differs since wave 1
      // (AC-801/803): search snippets may be query-aware headlines, while
      // getRecent always keeps the deterministic 180-char prefix.
      const prefix = byId
        .get(probe)!
        .content.replace(/\s+/g, " ")
        .slice(0, 180);
      expect(r).toEqual({ ...s, snippet: prefix });
      expect(r.snippet).toBe(prefix);
    }
    for (const r of recent) {
      expect("score" in r).toBe(false);
      expect(r.effectiveDate).toBe(r.eventDate ?? r.createdAt);
      expect(byId.has(r.id)).toBe(true);
    }
  });

  // ── search_log filter recording (wave-2 extensibility contract) ────────────

  it("search_log records tags/after in filters; retried flag on the retry row", async () => {
    const q1 = "gateway cert rotation probe";
    await searchMemory({
      projectId: PROJ,
      query: q1,
      tags: ["infra"],
      after: "2026-06-15T00:00:00Z",
      limit: 3,
    });
    const row1 = await until(async () => {
      const { rows } = await pool.query<{ filters: Record<string, unknown> }>(
        `SELECT filters FROM memory.search_log
         WHERE project_id = $1 AND query = $2
         ORDER BY created_at DESC LIMIT 1`,
        [PROJ, q1],
      );
      return rows[0] ?? null;
    });
    expect(row1).not.toBeNull();
    expect(row1!.filters).toEqual({
      tags: ["infra"],
      after: "2026-06-15T00:00:00Z",
    });

    const q2 = "gateway retried probe";
    // LOW-006 probe: slow the (single, up-front) query embed. The embed precedes the
    // first attempt's pool window, so a pool_ms frozen before the retry would come
    // out ≥ the delay, while a re-measured retried pool (SQL only) stays far below it.
    H.embedProbe.delayMs = 400;
    try {
      await searchMemory({
        projectId: PROJ,
        query: q2,
        tags: ["no-such-tag"],
        limit: 3,
      });
    } finally {
      H.embedProbe.delayMs = 0;
    }
    const row2 = await until(async () => {
      const { rows } = await pool.query<{
        filters: Record<string, unknown>;
        pool_ms: number;
        total_ms: number;
      }>(
        `SELECT filters, pool_ms, total_ms FROM memory.search_log
         WHERE project_id = $1 AND query = $2
         ORDER BY created_at DESC LIMIT 1`,
        [PROJ, q2],
      );
      return rows[0] ?? null;
    });
    expect(row2).not.toBeNull();
    expect(row2!.filters).toEqual({ tags: ["no-such-tag"], retried: true });
    // LOW-006: pool_ms covers the RETRIED pool — the one whose ids were logged.
    expect(row2!.pool_ms).toBeGreaterThanOrEqual(0);
    expect(row2!.pool_ms).toBeLessThan(400);
    expect(row2!.total_ms).toBeGreaterThanOrEqual(400);
    expect(row2!.total_ms).toBeGreaterThanOrEqual(row2!.pool_ms);
  });

  // ── AC-208: memory_get status/date/supersession banner ────────────────────

  it("AC-208: active memory_get returns JSON fields with no banner", async () => {
    const got = await callTool(client, "memory_get", { id: id(1) });
    expect(got.isError).toBe(false);
    expect(got.text).not.toMatch(/^status=|^superseded/);
    const row = JSON.parse(got.text) as Record<string, unknown>;
    expect(row.status).toBe("active");
    expect(row.event_date).toBeNull();
    expect(row.superseded_by).toBeNull();
  });

  it("AC-208: superseded memory_get prepends successor banner and includes fields", async () => {
    await pool.query(
      `UPDATE memory.memories
         SET status = 'superseded', superseded_by = $2
       WHERE id = $1`,
      [id(4), id(7)],
    );
    const got = await callTool(client, "memory_get", { id: id(4) });
    expect(got.isError).toBe(false);
    expect(got.text.split("\n")[0]).toBe(`superseded — see ${id(7)}`);
    const row = JSON.parse(got.text.split("\n").slice(1).join("\n")) as Record<
      string,
      unknown
    >;
    expect(row.status).toBe("superseded");
    expect(row.event_date).toBe(iso("2026-03-10T00:00:00Z"));
    expect(row.superseded_by).toBe(id(7));
  });

  it("AC-208: archived memory_get prepends status banner, never excludes", async () => {
    await pool.query(
      `UPDATE memory.memories SET status = 'archived' WHERE id = $1`,
      [id(8)],
    );
    const got = await callTool(client, "memory_get", { id: id(8) });
    expect(got.isError).toBe(false);
    expect(got.text.split("\n")[0]).toBe("status=archived");
    const row = JSON.parse(got.text.split("\n").slice(1).join("\n")) as Record<
      string,
      unknown
    >;
    expect(row.status).toBe("archived");
    expect(row.id).toBe(id(8));
  });

  // ── AC-805/806: memory_get content budget ──────────────────────────────────

  const BIG_ID = id(9);
  const BIG_CONTENT = "lorem ".repeat(1200); // 7200 chars > the 6000 default

  /** Idempotent oversized fixture (no embedding — memory_get never needs one). */
  async function insertBigRow(): Promise<void> {
    await pool.query(
      `INSERT INTO memory.memories (id, project_id, type, title, content)
       VALUES ($1, $2, 'semantic', 'Oversized body', $3)
       ON CONFLICT (id) DO NOTHING`,
      [BIG_ID, PROJ, BIG_CONTENT],
    );
  }

  it("AC-805: default memory_get truncates >6000-char content with explicit markers", async () => {
    await insertBigRow();
    const got = await callTool(client, "memory_get", { id: BIG_ID });
    expect(got.isError).toBe(false);
    const row = JSON.parse(got.text) as Record<string, unknown>;
    expect((row.content as string).length).toBe(6000);
    expect(row.content).toBe(BIG_CONTENT.slice(0, 6000));
    expect(row.truncated).toBe(true);
    expect(row.totalChars).toBe(7200);
    expect(row.note).toContain("full=true");
    // metadata is never budgeted, and the JSON stayed parseable by construction.
    expect(row.metadata).toEqual({});
  });

  it("AC-806: full=true returns the complete content with no truncation fields", async () => {
    await insertBigRow();
    const got = await callTool(client, "memory_get", {
      id: BIG_ID,
      full: true,
    });
    expect(got.isError).toBe(false);
    const row = JSON.parse(got.text) as Record<string, unknown>;
    expect(row.content).toBe(BIG_CONTENT);
    expect("truncated" in row).toBe(false);
    expect("totalChars" in row).toBe(false);
    expect("note" in row).toBe(false);
  });

  it("AC-806: maxChars=0 is the same escape hatch as full=true", async () => {
    await insertBigRow();
    const got = await callTool(client, "memory_get", {
      id: BIG_ID,
      maxChars: 0,
    });
    expect(got.isError).toBe(false);
    const row = JSON.parse(got.text) as Record<string, unknown>;
    expect(row.content).toBe(BIG_CONTENT);
    expect("truncated" in row).toBe(false);
  });
});

// ── Wave-2: formatHits summary line (pure — no DB) ───────────────────────────

describe("formatHits summary line (wave-2)", () => {
  const base = {
    id: "00000000-0000-4000-8000-00000000fee1",
    title: "A memory",
    type: "semantic" as const,
    snippet: "the snippet text",
    summary: null as string | null,
    importance: 0.5,
    createdAt: "2026-06-20T00:00:00.000Z",
    eventDate: null,
    effectiveDate: "2026-06-20T00:00:00.000Z",
    status: "active",
  };

  it("a NULL-summary hit renders EXACTLY the pre-wave-2 two-line form (pin)", () => {
    expect(formatHits([base])).toBe(
      `1. [semantic] A memory  (2026-06-20, id: ${base.id})\n   the snippet text`,
    );
  });

  it("a summarized hit appends the third indented summary line", () => {
    const hit = { ...base, summary: "Dense stored summary." };
    expect(formatHits([hit])).toBe(
      `1. [semantic] A memory  (2026-06-20, id: ${base.id})\n   the snippet text\n   summary: Dense stored summary.`,
    );
  });

  it("the summary line composes with a non-default status note", () => {
    const hit = { ...base, summary: "S.", status: "superseded" };
    expect(formatHits([hit])).toBe(
      `1. [semantic] A memory  (2026-06-20, status: superseded, id: ${base.id})\n   the snippet text\n   summary: S.`,
    );
  });
});
