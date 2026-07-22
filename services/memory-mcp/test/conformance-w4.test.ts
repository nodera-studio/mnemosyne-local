// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Wave 4 / WS2 token shaping (memory-mcp half)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/
// wave-4-token-shaping.md + index.md ACs). Every assertion below was drafted from
// the EARS acceptance criteria BEFORE locating the exported symbols; this suite is
// the independent anti-reward-hacking gate, additive to the Implementer's tests.
//
// AC map (memory-mcp scope):
//   AC-201 — `tags` (ANY-of on metadata) + `after` (COALESCE(event_date,
//            created_at) >= after) filter the fused pool in BOTH RRF arms:
//            a BM25-only row (no vector) and a vector-only row (no lexical
//            overlap) are BOTH excluded when they fail the filter.
//   AC-202 — zero-pool filtered search retries once unfiltered and prepends a
//            notice line naming the dropped filter(s); scoping filter `type`
//            is KEPT on the retry; zero-corpus unfiltered query stays plain.
//   AC-203 — responses carry ranks + identifiers but NO raw relevance/blend
//            scores (no `score` key on public hits, no score text).
//   AC-204 — truncation at k (pool larger) appends a steering line naming
//            type/tags/after as OPTIONAL refinements; k ≥ pool does not.
//   AC-205 — every registered tool description has "Use when" + "Do NOT use"
//            clauses; tags/after are optional refinements, never required.
//   AC-207 — public hits (memory_search + memory_get_recent) carry createdAt /
//            eventDate (ISO|null) / effectiveDate (= eventDate ?? createdAt) /
//            status; NULL event_date supported; formatHits renders the
//            effective date (YYYY-MM-DD) in the hit header for BOTH tools.
//   AC-208 — memory_get JSON includes status/event_date/superseded_by;
//            superseded rows get the `superseded — see <id>` banner, other
//            non-active statuses `status=<status>` — banner, never exclusion.
//   plan §Step 1/3 — search_log records the filters jsonb (present keys only);
//            a zero-pool retry logs ONE row carrying a retried:true marker.
//
// Deterministic throughout: Voyage is module-mocked (seeded PRNG vectors and
// rerank scores); fixture vectors are literals. DB tests are self-contained
// under conf-w4-mem-* project ids and clean up after themselves. NO live
// Voyage/Anthropic quota is ever spent here.
// ─────────────────────────────────────────────────────────────────────────────

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import type { MemoryType } from "../src/memory.js";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic helpers, hoisted so the vi.mock factory can reference them.
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

// HARD RULE: never call live Voyage — module-mock the boundary.
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

// Dynamic imports AFTER env + mocks are in place (config reads env at load).
const mem = await import("../src/memory.js");
const { buildServer } = await import("../src/server.js");
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } =
  await import("@modelcontextprotocol/sdk/inMemory.js");

// Unique fixture namespace — never collides with other suites' project ids.
const P_FILT = "conf-w4-mem-filt";
const P_STEER = "conf-w4-mem-steer";
const P_DATE = "conf-w4-mem-date";
const P_GET = "conf-w4-mem-get";
const P_EMPTY = "conf-w4-mem-empty";
const ALL_PROJECTS = [P_FILT, P_STEER, P_DATE, P_GET, P_EMPTY];

const mid = (n: number) =>
  `40000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// Filter-project fixture ids (query below reaches every arm deliberately).
const Q = "alpha rotation policy vault";
const F_TAG = mid(1); // tagged, embedded, recent
const F_BM25 = mid(2); // UNtagged, old, NO vector — reachable via BM25 arm only
const F_VEC = mid(3); // UNtagged, old, vector ≡ query vector — vector arm only
const F_EVT = mid(4); // old created_at, RECENT event_date (COALESCE probe)
const F_PROC = mid(5); // procedural type (scoping-filter-kept probe)

const D_NULL = mid(40); // event_date NULL → effectiveDate = createdAt
const D_EVT = mid(41); // event_date ≠ created_at → effectiveDate = eventDate
const D_NULL_CREATED = "2026-06-10T00:00:00Z";
const D_EVT_CREATED = "2026-06-20T00:00:00Z";
const D_EVT_EVENT = "2026-03-05T00:00:00Z";

const G_ACT = mid(60);
const G_SUCC = mid(61);
const G_SUP = mid(62);
const G_ARCH = mid(63);

type McpClient = InstanceType<typeof Client>;

async function connectClient(projectId: string): Promise<McpClient> {
  const server = buildServer(projectId);
  const [clientTransport, serverTransport] =
    InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "conf-w4", version: "1.0.0" });
  await Promise.all([
    client.connect(clientTransport),
    server.connect(serverTransport),
  ]);
  return client;
}

async function callTool(
  client: McpClient,
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

// ── Pure: AC-205 — tool descriptions + optional filter params (no DB) ────────

describe("AC-205: tool descriptions — disjoint Use-when / Do-NOT-use, optional filters", () => {
  let client: McpClient;

  beforeAll(async () => {
    client = await connectClient("conf-w4-mem-desc");
  });

  afterAll(async () => {
    await client.close();
  });

  it("every registered tool description contains both a 'Use when' and a 'Do NOT use' clause", async () => {
    const { tools } = await client.listTools();
    expect(tools.length).toBeGreaterThanOrEqual(8);
    for (const t of tools) {
      const d = t.description ?? "";
      expect(d.toLowerCase(), `${t.name} needs a "Use when" clause`).toContain(
        "use when",
      );
      expect(
        d.toLowerCase(),
        `${t.name} needs a "Do NOT use" clause`,
      ).toContain("do not use");
    }
  });

  it("memory_search exposes tags + after as optional refinements, never required", async () => {
    const { tools } = await client.listTools();
    const search = tools.find((t) => t.name === "memory_search");
    expect(search).toBeTruthy();
    const schema = search!.inputSchema as {
      properties: Record<string, { description?: string }>;
      required?: string[];
    };
    expect(schema.properties.tags).toBeTruthy();
    expect(schema.properties.after).toBeTruthy();
    const required = schema.required ?? [];
    expect(required).not.toContain("tags");
    expect(required).not.toContain("after");
    expect(schema.properties.tags.description ?? "").toMatch(/optional/i);
    expect(schema.properties.after.description ?? "").toMatch(/optional/i);
  });

  it("the core tool set is registered under its documented names", async () => {
    const { tools } = await client.listTools();
    const names = tools.map((t) => t.name);
    for (const expected of [
      "memory_search",
      "memory_store",
      "memory_get",
      "memory_get_recent",
      "memory_update",
      "memory_list",
    ]) {
      expect(names).toContain(expected);
    }
  });
});

// ── DB-backed conformance (skipped cleanly without DATABASE_URL) ─────────────

describe.skipIf(skip)("W4 conformance — DB-backed (memory-mcp)", () => {
  let db: pg.Pool;
  let filtClient: McpClient;
  let steerClient: McpClient;
  let dateClient: McpClient;
  let getClient: McpClient;

  async function insertMemory(row: {
    id: string;
    projectId: string;
    type?: MemoryType;
    title: string;
    content: string;
    importance?: number;
    createdAt?: string;
    eventDate?: string | null;
    status?: string;
    tags?: string[];
    embedKey?: string | null;
    supersededBy?: string | null;
  }): Promise<void> {
    await db.query(
      `INSERT INTO memory.memories
         (id, project_id, type, title, content, importance, created_at,
          event_date, status, metadata, superseded_by, embedding_v2)
       VALUES ($1, $2, $3::memory.memory_type, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12::halfvec)`,
      [
        row.id,
        row.projectId,
        row.type ?? "semantic",
        row.title,
        row.content,
        row.importance ?? 0.5,
        row.createdAt ?? "2026-06-20T00:00:00Z",
        row.eventDate ?? null,
        row.status ?? "active",
        JSON.stringify(row.tags ? { tags: row.tags } : {}),
        row.supersededBy ?? null,
        row.embedKey ? `[${H.fakeVec(row.embedKey).join(",")}]` : null,
      ],
    );
  }

  async function cleanup(): Promise<void> {
    await db.query(
      `UPDATE memory.memories SET superseded_by = NULL WHERE project_id = ANY($1)`,
      [ALL_PROJECTS],
    );
    await db.query(`DELETE FROM memory.memories WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
    await db.query(`DELETE FROM memory.search_log WHERE project_id = ANY($1)`, [
      ALL_PROJECTS,
    ]);
  }

  /** Poll a predicate on a deadline (search_log inserts are fire-and-forget). */
  async function until(
    fn: () => Promise<boolean>,
    ms = 4000,
    step = 50,
  ): Promise<boolean> {
    const deadline = Date.now() + ms;
    while (Date.now() < deadline) {
      if (await fn()) return true;
      await new Promise((r) => setTimeout(r, step));
    }
    return fn();
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Idempotent migration apply (HOLD files skipped) — self-sufficient on a fresh DB.
    const { readFileSync, readdirSync } = await import("node:fs");
    const { dirname, join } = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    const sqlDir = join(dirname(fileURLToPath(import.meta.url)), "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();

    // ── P_FILT: per-arm filter probes ────────────────────────────────────────
    await insertMemory({
      id: F_TAG,
      projectId: P_FILT,
      title: "Vault rotation policy",
      content: "alpha rotation policy for the vault secrets and keys",
      tags: ["vault", "policy"],
      createdAt: "2026-06-15T00:00:00Z",
      embedKey: "w4:tag",
    });
    await insertMemory({
      id: F_BM25,
      projectId: P_FILT,
      title: "Untagged lexical twin",
      content: "alpha rotation policy vault untagged lexical twin row",
      createdAt: "2026-05-01T00:00:00Z",
      embedKey: null, // BM25 arm ONLY
    });
    await insertMemory({
      id: F_VEC,
      projectId: P_FILT,
      title: "Untagged vector twin",
      content: "zebra quantum prose with no lexical overlap whatsoever",
      createdAt: "2026-05-01T00:00:00Z",
      embedKey: Q, // identical to the query vector → vector arm ONLY
    });
    await insertMemory({
      id: F_EVT,
      projectId: P_FILT,
      title: "Old row with recent event date",
      content: "alpha rotation policy vault event dated row",
      createdAt: "2026-01-06T00:00:00Z",
      eventDate: "2026-06-20T00:00:00Z",
      embedKey: "w4:evt",
    });
    await insertMemory({
      id: F_PROC,
      projectId: P_FILT,
      type: "procedural",
      title: "Procedural vault runbook",
      content: "alpha rotation policy vault procedural runbook steps",
      createdAt: "2026-06-10T00:00:00Z",
      embedKey: "w4:proc",
    });

    // ── P_STEER: 8-row pool for the truncation-steering boundary ────────────
    for (let i = 1; i <= 8; i++) {
      await insertMemory({
        id: mid(20 + i),
        projectId: P_STEER,
        title: `Steering row ${i}`,
        content: `steering conformance shared beacon token row ${i} ${"pad ".repeat(i)}`,
        importance: 0.3 + i * 0.05,
        createdAt: `2026-06-${String(10 + i).padStart(2, "0")}T00:00:00Z`,
        embedKey: `steer:${i}`,
      });
    }

    // ── P_DATE: AC-207 date/status fixtures ──────────────────────────────────
    await insertMemory({
      id: D_NULL,
      projectId: P_DATE,
      title: "Temporal row without event date",
      content: "temporal conformance golden date row null variant",
      createdAt: D_NULL_CREATED,
      eventDate: null,
      embedKey: "w4:dnull",
    });
    await insertMemory({
      id: D_EVT,
      projectId: P_DATE,
      title: "Temporal row with event date",
      content: "temporal conformance golden date row event variant",
      createdAt: D_EVT_CREATED,
      eventDate: D_EVT_EVENT,
      embedKey: "w4:devt",
    });

    // ── P_GET: AC-208 status/supersession fixtures (successor first — FK) ───
    await insertMemory({
      id: G_SUCC,
      projectId: P_GET,
      title: "Successor row",
      content: "the successor of the superseded fixture row",
      embedKey: "w4:gsucc",
    });
    await insertMemory({
      id: G_ACT,
      projectId: P_GET,
      title: "Active fixture row",
      content: "plain active row for the no-banner case",
      eventDate: "2026-02-01T00:00:00Z",
      embedKey: "w4:gact",
    });
    await insertMemory({
      id: G_SUP,
      projectId: P_GET,
      title: "Superseded fixture row",
      content: "superseded row whose successor must be pointed at",
      status: "superseded",
      supersededBy: G_SUCC,
      embedKey: "w4:gsup",
    });
    await insertMemory({
      id: G_ARCH,
      projectId: P_GET,
      title: "Archived fixture row",
      content: "archived row for the status banner case",
      status: "archived",
      embedKey: "w4:garch",
    });

    filtClient = await connectClient(P_FILT);
    steerClient = await connectClient(P_STEER);
    dateClient = await connectClient(P_DATE);
    getClient = await connectClient(P_GET);
  });

  afterAll(async () => {
    await filtClient?.close();
    await steerClient?.close();
    await dateClient?.close();
    await getClient?.close();
    await cleanup();
    await db.end();
  });

  // ── AC-201: filters hit BOTH RRF arms of the fused pool ────────────────────

  it("AC-201 sanity: the unfiltered pool reaches all three probe rows", async () => {
    const pool = await mem.fuseCandidates({ projectId: P_FILT, query: Q });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(F_TAG);
    expect(ids).toContain(F_BM25); // BM25 arm works without a vector
    expect(ids).toContain(F_VEC); // vector arm works without lexical overlap
  });

  it("AC-201: tags filter excludes non-tagged rows from BOTH arms", async () => {
    const pool = await mem.fuseCandidates({
      projectId: P_FILT,
      query: Q,
      tags: ["vault"],
    });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(F_TAG);
    expect(ids).not.toContain(F_BM25); // would leak if only the vector arm were filtered
    expect(ids).not.toContain(F_VEC); // would leak if only the BM25 arm were filtered
    expect(ids).toHaveLength(1);
  });

  it("AC-201: tags match is ANY-of", async () => {
    const pool = await mem.fuseCandidates({
      projectId: P_FILT,
      query: Q,
      tags: ["vault", "zzz-absent-tag"],
    });
    expect(pool.map((c) => c.id)).toContain(F_TAG);
  });

  it("AC-201: after cuts old rows in BOTH arms and honors COALESCE(event_date, created_at)", async () => {
    const pool = await mem.fuseCandidates({
      projectId: P_FILT,
      query: Q,
      after: "2026-06-01T00:00:00Z",
    });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(F_TAG); // created 2026-06-15
    expect(ids).toContain(F_EVT); // created January, event_date 2026-06-20
    expect(ids).toContain(F_PROC); // created 2026-06-10
    expect(ids).not.toContain(F_BM25); // old, BM25-arm probe
    expect(ids).not.toContain(F_VEC); // old, vector-arm probe
  });

  it("AC-201: filters compose (tags + after + type)", async () => {
    const both = await mem.fuseCandidates({
      projectId: P_FILT,
      query: Q,
      tags: ["vault"],
      after: "2026-06-01T00:00:00Z",
    });
    expect(both.map((c) => c.id)).toEqual([F_TAG]);

    const withType = await mem.fuseCandidates({
      projectId: P_FILT,
      query: Q,
      type: "semantic",
      tags: ["vault"],
    });
    expect(withType.map((c) => c.id)).toEqual([F_TAG]);
  });

  // ── AC-202: zero-pool filtered search → one unfiltered retry + notice ──────

  it("AC-202: impossible filter retries unfiltered and reports it", async () => {
    const result = await mem.searchMemory({
      projectId: P_FILT,
      query: Q,
      tags: ["zzz-impossible-tag"],
      limit: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(Object.keys(result.droppedFilters)).toContain("tags");
  });

  it("AC-202: scoping filter `type` is KEPT on the retry", async () => {
    const result = await mem.searchMemory({
      projectId: P_FILT,
      query: Q,
      type: "procedural",
      tags: ["zzz-impossible-tag"],
      limit: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    for (const h of result.hits) expect(h.type).toBe("procedural");
  });

  it("AC-202: zero-corpus unfiltered query returns plain no-results, no retry", async () => {
    const result = await mem.searchMemory({
      projectId: P_EMPTY,
      query: "anything at all",
      limit: 5,
    });
    expect(result.hits).toEqual([]);
    expect(result.retriedWithoutFilters).toBe(false);
    const { text } = await callTool(
      await connectClient(P_EMPTY),
      "memory_search",
      { query: "anything at all", limit: 5 },
    );
    expect(text).toContain("No matching memories.");
    expect(text).not.toContain("Note: no results matched filters");
  });

  it("AC-202 (MCP): the notice line is prepended and names every dropped filter", async () => {
    const { text, isError } = await callTool(filtClient, "memory_search", {
      query: Q,
      tags: ["zzz-impossible-tag"],
      after: "2026-06-01T00:00:00Z",
      limit: 10,
    });
    expect(isError).toBe(false);
    const first = text.split("\n")[0];
    expect(first.startsWith("Note: no results matched filters")).toBe(true);
    expect(first).toContain("tags");
    expect(first).toContain("zzz-impossible-tag");
    expect(first).toContain("after");
    expect(first).toContain("showing unfiltered results");
    expect(first).toMatch(/filters are optional/i);
    // Unfiltered results follow the notice.
    expect(text).toMatch(/\n1\. /);
  });

  // ── AC-203: rank-only output — no raw scores anywhere ──────────────────────

  it("AC-203: public hits carry no `score` key", async () => {
    const result = await mem.searchMemory({
      projectId: P_FILT,
      query: Q,
      limit: 5,
    });
    expect(result.hits.length).toBeGreaterThan(0);
    for (const h of result.hits) {
      expect("score" in h).toBe(false);
      expect(typeof h.id).toBe("string");
    }
  });

  it("AC-203 (MCP): response text has ranks + ids but no score text", async () => {
    const { text } = await callTool(filtClient, "memory_search", {
      query: Q,
      limit: 10,
    });
    expect(text).toMatch(/^1\. /m); // ranks
    expect(text).toContain(F_TAG); // identifiers
    expect(text).not.toMatch(/score/i); // no score labels
    expect(text).not.toMatch(/0\.\d{4}/); // no raw score values
  });

  // ── AC-204: truncation-steering boundary ───────────────────────────────────

  it("AC-204: k=3 over an 8-row pool ends with the steering line", async () => {
    const direct = await mem.searchMemory({
      projectId: P_STEER,
      query: "steering conformance shared beacon token",
      limit: 3,
    });
    expect(direct.hits).toHaveLength(3);
    expect(direct.poolSize).toBe(8);
    expect(direct.truncated).toBe(true);

    const { text } = await callTool(steerClient, "memory_search", {
      query: "steering conformance shared beacon token",
      limit: 3,
    });
    const last = text.trimEnd().split("\n").at(-1) ?? "";
    expect(last).toContain("Showing top 3 of 8 candidates");
    expect(last).toMatch(/optionally/i);
    expect(last).toContain("type");
    expect(last).toContain("tags");
    expect(last).toContain("after");
  });

  it("AC-204: k ≥ pool size appends no steering line", async () => {
    const direct = await mem.searchMemory({
      projectId: P_STEER,
      query: "steering conformance shared beacon token",
      limit: 12,
    });
    expect(direct.truncated).toBe(false);

    const { text } = await callTool(steerClient, "memory_search", {
      query: "steering conformance shared beacon token",
      limit: 12,
    });
    expect(text).not.toContain("Showing top");
  });

  // ── AC-207: public date/status surface + rendered effective dates ──────────

  it("AC-207: hits carry createdAt/eventDate/effectiveDate/status; NULL event_date falls back", async () => {
    const result = await mem.searchMemory({
      projectId: P_DATE,
      query: "temporal conformance golden date row",
      limit: 5,
    });
    const byId = new Map(result.hits.map((h) => [h.id, h]));
    const nullHit = byId.get(D_NULL);
    const evtHit = byId.get(D_EVT);
    expect(nullHit).toBeTruthy();
    expect(evtHit).toBeTruthy();

    expect(nullHit!.createdAt).toBe(new Date(D_NULL_CREATED).toISOString());
    expect(nullHit!.eventDate).toBeNull();
    expect(nullHit!.effectiveDate).toBe(nullHit!.createdAt);
    expect(nullHit!.status).toBe("active");

    expect(evtHit!.createdAt).toBe(new Date(D_EVT_CREATED).toISOString());
    expect(evtHit!.eventDate).toBe(new Date(D_EVT_EVENT).toISOString());
    expect(evtHit!.effectiveDate).toBe(evtHit!.eventDate);
    expect(evtHit!.effectiveDate).not.toBe(evtHit!.createdAt);
    expect(evtHit!.status).toBe("active");
  });

  it("AC-207: memory_get_recent exposes the identical date/status fields", async () => {
    const search = await mem.searchMemory({
      projectId: P_DATE,
      query: "temporal conformance golden date row",
      limit: 5,
    });
    const recent = await mem.getRecent({ projectId: P_DATE, limit: 10 });
    expect(recent.length).toBe(2);
    const recentById = new Map(recent.map((h) => [h.id, h]));
    for (const s of search.hits) {
      const r = recentById.get(s.id);
      expect(r).toBeTruthy();
      expect(r!.createdAt).toBe(s.createdAt);
      expect(r!.eventDate).toBe(s.eventDate);
      expect(r!.effectiveDate).toBe(s.effectiveDate);
      expect(r!.status).toBe(s.status);
    }
  });

  it("AC-207 (MCP): both tools render the effective date (YYYY-MM-DD) in the hit header", async () => {
    for (const toolName of ["memory_search", "memory_get_recent"]) {
      const args =
        toolName === "memory_search"
          ? { query: "temporal conformance golden date row", limit: 5 }
          : { limit: 10 };
      const { text } = await callTool(dateClient, toolName, args);
      const evtLine = text
        .split("\n")
        .find((l) => l.includes("Temporal row with event date"));
      const nullLine = text
        .split("\n")
        .find((l) => l.includes("Temporal row without event date"));
      expect(evtLine, `${toolName} must list the event-date row`).toBeTruthy();
      expect(nullLine, `${toolName} must list the null-date row`).toBeTruthy();
      expect(evtLine!).toContain("2026-03-05"); // eventDate, NOT createdAt
      expect(evtLine!).not.toContain("2026-06-20");
      expect(nullLine!).toContain("2026-06-10"); // createdAt fallback
    }
  });

  // ── AC-208: memory_get banners — banner, never exclusion ───────────────────

  it("AC-208: superseded row → `superseded — see <successor>` banner + full JSON", async () => {
    const { text, isError } = await callTool(getClient, "memory_get", {
      id: G_SUP,
    });
    expect(isError).toBe(false);
    const nl = text.indexOf("\n");
    expect(nl).toBeGreaterThan(0);
    expect(text.slice(0, nl)).toBe(`superseded — see ${G_SUCC}`);
    const row = JSON.parse(text.slice(nl + 1)) as Record<string, unknown>;
    expect(row.id).toBe(G_SUP);
    expect(row.status).toBe("superseded");
    expect(row.superseded_by).toBe(G_SUCC);
    expect("event_date" in row).toBe(true);
  });

  it("AC-208: archived row → `status=archived` banner + full JSON", async () => {
    const { text, isError } = await callTool(getClient, "memory_get", {
      id: G_ARCH,
    });
    expect(isError).toBe(false);
    const nl = text.indexOf("\n");
    expect(text.slice(0, nl)).toBe("status=archived");
    const row = JSON.parse(text.slice(nl + 1)) as Record<string, unknown>;
    expect(row.id).toBe(G_ARCH);
    expect(row.status).toBe("archived");
  });

  it("AC-208: active row → no banner; JSON includes status/event_date/superseded_by", async () => {
    const { text, isError } = await callTool(getClient, "memory_get", {
      id: G_ACT,
    });
    expect(isError).toBe(false);
    const row = JSON.parse(text) as Record<string, unknown>;
    expect(row.id).toBe(G_ACT);
    expect(row.status).toBe("active");
    expect(row.superseded_by).toBeNull();
    expect(row.event_date).toBeTruthy();

    const direct = await mem.getMemory(G_ACT);
    expect(direct).toBeTruthy();
    expect("status" in direct!).toBe(true);
    expect("event_date" in direct!).toBe(true);
    expect("superseded_by" in direct!).toBe(true);
  });

  // ── plan Step 1 detail: bad `after` input fails cleanly ────────────────────

  it("bad `after` value is rejected as a clean tool error (never a pg crash)", async () => {
    let errored = false;
    try {
      const res = await callTool(filtClient, "memory_search", {
        query: Q,
        after: "not-a-date",
        limit: 5,
      });
      errored = res.isError;
    } catch {
      errored = true; // schema-level rejection is equally acceptable
    }
    expect(errored).toBe(true);
  });

  // ── plan Steps 1+3: search_log records filters; retry logs ONE row ─────────

  it("search_log records the supplied filters (present keys only)", async () => {
    const probe = `${Q} log probe filters`;
    await mem.searchMemory({
      projectId: P_FILT,
      query: probe,
      tags: ["vault"],
      after: "2026-06-01T00:00:00Z",
      limit: 5,
    });
    const landed = await until(async () => {
      const { rows } = await db.query(
        `SELECT filters FROM memory.search_log WHERE project_id = $1 AND query = $2`,
        [P_FILT, probe],
      );
      return rows.length === 1;
    });
    expect(landed).toBe(true);
    const { rows } = await db.query<{ filters: Record<string, unknown> }>(
      `SELECT filters FROM memory.search_log WHERE project_id = $1 AND query = $2`,
      [P_FILT, probe],
    );
    expect(rows[0].filters.tags).toBeTruthy();
    expect(rows[0].filters.after).toBeTruthy();
    expect("type" in rows[0].filters).toBe(false); // only present keys
  });

  it("a zero-pool retry is logged as ONE row carrying a retried marker", async () => {
    const probe = `${Q} log probe retry`;
    await mem.searchMemory({
      projectId: P_FILT,
      query: probe,
      tags: ["zzz-impossible-tag"],
      limit: 5,
    });
    const landed = await until(async () => {
      const { rows } = await db.query(
        `SELECT 1 FROM memory.search_log WHERE project_id = $1 AND query = $2`,
        [P_FILT, probe],
      );
      return rows.length >= 1;
    });
    expect(landed).toBe(true);
    // Give a possible (non-conformant) second insert time to land before counting.
    await new Promise((r) => setTimeout(r, 300));
    const { rows } = await db.query(
      `SELECT * FROM memory.search_log WHERE project_id = $1 AND query = $2`,
      [P_FILT, probe],
    );
    expect(rows).toHaveLength(1);
    expect(JSON.stringify(rows[0])).toMatch(/"retried"\s*:\s*true/);
  });
});
