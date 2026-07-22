// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Wave 4 / WS2 token shaping (codebase-mcp half)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/
// wave-4-token-shaping.md + index.md ACs). Assertions drafted from the EARS
// acceptance criteria BEFORE locating exported symbols; independent of the
// Implementer's own tests.
//
// AC map (codebase-mcp scope):
//   AC-201 — `path` (escaped substring) + `extension` (dot-normalized) filter
//            the fused pool in BOTH RRF arms: a BM25-only chunk (no vector)
//            and a vector-only chunk (no lexical overlap) are BOTH excluded
//            when they fail the filter; LIKE wildcards in user input are
//            escaped (no `%`/`_` injection); composes with `repo`.
//   AC-202 — zero-pool filtered search retries once unfiltered with a notice
//            naming the dropped filter(s); scoping filters (`repo`) are KEPT
//            on the retry; zero-corpus unfiltered query stays plain.
//   AC-203 — rank-only output: no `score` key on public hits, no score text.
//   AC-204 — truncation at k (pool larger) ends with a steering line naming
//            repo/path/extension/language as OPTIONAL refinements plus the
//            code_get_file follow-up; k ≥ pool does not.
//   AC-205 — every tool description has "Use when" + "Do NOT use" clauses;
//            path/extension are optional refinements, never required.
//   AC-206 — overlapping/adjacent same-file top-k chunks merge into one hit
//            with the interval-union line range, hard-capped at maxSpan
//            (MAX_MERGED_LINES); merged hits keep the best constituent's rank
//            position; different files never merge; no backfill.
//
// Deterministic throughout: Voyage is module-mocked (seeded PRNG vectors and
// rerank scores); fixture vectors are literals. DB tests are self-contained
// under conf-w4-code-* project/repo ids and clean up. NO live Voyage/Anthropic
// quota is ever spent here.
// ─────────────────────────────────────────────────────────────────────────────

import { tmpdir } from "node:os";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";
import type { MergeableCodeHit } from "../src/merge-hits.js";

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
  embedCode: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(`code:${t}`)),
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`code:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

// Env BEFORE the src imports — config.ts reads process.env at module load.
process.env.VOYAGE_API_KEY ??= "test-key";
process.env.REPOS_ROOT ??= tmpdir();
process.env.DEFAULT_PROJECT_ID ??= "conf-w4-code-default";

const search = await import("../src/search.js");
const { mergeAdjacentHits } = await import("../src/merge-hits.js");
const { buildServer } = await import("../src/server.js");
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } =
  await import("@modelcontextprotocol/sdk/inMemory.js");

// Unique fixture namespace.
const P_CFILT = "conf-w4-code-filt";
const P_STEER = "conf-w4-code-steer";
const P_MERGE = "conf-w4-code-merge";
const P_EMPTY = "conf-w4-code-empty";
const ALL_PROJECTS = [P_CFILT, P_STEER, P_MERGE, P_EMPTY];
const R_FILT = "conf-w4-repo-filt";
const R_OTHER = "conf-w4-repo-other";
const R_STEER = "conf-w4-repo-steer";
const R_MERGE = "conf-w4-repo-merge";
const ALL_REPOS = [R_FILT, R_OTHER, R_STEER, R_MERGE];

const cid = (n: number) =>
  `50000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// Filter-project fixtures (the query reaches every arm deliberately).
const CQ = "frontier traversal batching queue";
const K_TS = cid(1); // .ts under src/graph/, embedded, lexical match
const K_MD_BM25 = cid(2); // .md, NO vector — BM25 arm only
const K_MD_VEC = cid(3); // .md, vector ≡ query vector, no lexical overlap
const K_PCT = cid(4); // path contains a literal `%`
const K_WILD = cid(5); // path `noXmatch` — wildcard-injection probe
const K_OTH = cid(6); // same project, OTHER repo (repo-scoping probe)

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

// ── Pure: AC-206 — adjacent-chunk merge (no DB) ──────────────────────────────

let hitSeq = 0;
function hit(
  filePath: string,
  startLine: number,
  endLine: number,
): MergeableCodeHit {
  hitSeq += 1;
  return {
    chunkId: `conf-w4-hit-${hitSeq}`,
    repositoryId: "conf-w4-repo-pure",
    filePath,
    startLine,
    endLine,
    symbolName: `sym${hitSeq}`,
    language: "typescript",
    snippet: `snippet ${hitSeq}`,
  };
}

describe("AC-206: mergeAdjacentHits (pure)", () => {
  it("two overlapping same-file chunks merge to the interval union with the leader's identity", () => {
    const a = hit("src/a.ts", 10, 70);
    const b = hit("src/a.ts", 58, 118);
    const out = mergeAdjacentHits([a, b], 120);
    expect(out).toHaveLength(1);
    expect(out[0].startLine).toBe(10);
    expect(out[0].endLine).toBe(118);
    expect(out[0].mergedCount).toBe(2);
    // Leader = best-ranked constituent: id/snippet/symbol come from it.
    expect(out[0].chunkId).toBe(a.chunkId);
    expect(out[0].snippet).toBe(a.snippet);
    expect(out[0].symbolName).toBe(a.symbolName);
    expect(out[0].filePath).toBe("src/a.ts");
  });

  it("merged hits keep the best constituent's rank position; k shrinks, no backfill", () => {
    const a = hit("src/a.ts", 10, 70);
    const other = hit("src/other.ts", 5, 40);
    const b = hit("src/a.ts", 58, 118);
    const out = mergeAdjacentHits([a, other, b], 120);
    expect(out).toHaveLength(2);
    expect(out[0].chunkId).toBe(a.chunkId); // merged hit at rank 1 (a's position)
    expect(out[0].startLine).toBe(10);
    expect(out[0].endLine).toBe(118);
    expect(out[1].chunkId).toBe(other.chunkId); // survivor keeps its order
    expect(out[1].mergedCount ?? 1).toBe(1);
  });

  it("touching chunks merge (plan: overlap OR touch)", () => {
    const out = mergeAdjacentHits(
      [hit("src/t.ts", 1, 50), hit("src/t.ts", 51, 90)],
      120,
    );
    expect(out).toHaveLength(1);
    expect(out[0].startLine).toBe(1);
    expect(out[0].endLine).toBe(90);
  });

  it("chunks separated by a gap never merge", () => {
    const out = mergeAdjacentHits(
      [hit("src/g.ts", 1, 50), hit("src/g.ts", 60, 100)],
      400,
    );
    expect(out).toHaveLength(2);
  });

  it("different files never merge, even with identical ranges", () => {
    const out = mergeAdjacentHits(
      [hit("src/x.ts", 1, 60), hit("src/y.ts", 1, 60)],
      400,
    );
    expect(out).toHaveLength(2);
  });

  it("span cap: a union of exactly maxSpan lines merges; clearly above it does not", () => {
    // Union 1..120 = 120 lines — within the cap under either span convention.
    const atCap = mergeAdjacentHits(
      [hit("src/cap.ts", 1, 60), hit("src/cap.ts", 49, 120)],
      120,
    );
    expect(atCap).toHaveLength(1);
    expect(atCap[0].startLine).toBe(1);
    expect(atCap[0].endLine).toBe(120);

    // Union 1..122 = 122 lines — beyond the cap under either span convention.
    const overCap = mergeAdjacentHits(
      [hit("src/cap.ts", 1, 60), hit("src/cap.ts", 51, 122)],
      120,
    );
    expect(overCap).toHaveLength(2);
  });

  it("a chunker-shaped chain (60-line windows, step 48) splits at the cap and chains without one", () => {
    const chain = () => [
      hit("src/chain.ts", 1, 60),
      hit("src/chain.ts", 49, 108),
      hit("src/chain.ts", 97, 156),
    ];
    const capped = mergeAdjacentHits(chain(), 120);
    expect(capped).toHaveLength(2);
    // Fairness note: the plan's literal "output ranges never overlap" property is
    // unsatisfiable when the cap splits a chain of by-construction-overlapping
    // chunks (any grouping of [1-60],[49-108],[97-156] under a 120 cap keeps one
    // 12-line chunker overlap). The satisfiable invariant: merging never CREATES
    // overlap beyond the inputs' pre-existing 12-line overlap.
    const sorted = [...capped].sort((a, b) => a.startLine - b.startLine);
    expect(sorted[0].endLine - sorted[1].startLine).toBeLessThanOrEqual(12 - 1);
    for (const h of capped) {
      if ((h.mergedCount ?? 1) > 1) {
        expect(h.endLine - h.startLine).toBeLessThanOrEqual(120);
      }
    }

    const uncapped = mergeAdjacentHits(chain(), 400);
    expect(uncapped).toHaveLength(1);
    expect(uncapped[0].startLine).toBe(1);
    expect(uncapped[0].endLine).toBe(156);
    expect(uncapped[0].mergedCount).toBe(3);
  });

  it("empty input and single hits pass through unchanged", () => {
    expect(mergeAdjacentHits([], 120)).toEqual([]);
    const single = hit("src/s.ts", 1, 60);
    const out = mergeAdjacentHits([single], 120);
    expect(out).toHaveLength(1);
    expect(out[0].startLine).toBe(1);
    expect(out[0].endLine).toBe(60);
    expect(out[0].mergedCount ?? 1).toBe(1);
  });
});

// ── Pure: AC-205 — tool descriptions + optional filter params (no DB) ────────

describe("AC-205: tool descriptions — disjoint Use-when / Do-NOT-use, optional filters", () => {
  let client: McpClient;

  beforeAll(async () => {
    client = await connectClient("conf-w4-code-desc");
  });

  afterAll(async () => {
    await client.close();
  });

  it("every registered tool description contains both a 'Use when' and a 'Do NOT use' clause", async () => {
    const { tools } = await client.listTools();
    expect(tools.length).toBeGreaterThanOrEqual(7);
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

  it("code_search exposes path + extension as optional refinements, never required", async () => {
    const { tools } = await client.listTools();
    const cs = tools.find((t) => t.name === "code_search");
    expect(cs).toBeTruthy();
    const schema = cs!.inputSchema as {
      properties: Record<string, { description?: string }>;
      required?: string[];
    };
    expect(schema.properties.path).toBeTruthy();
    expect(schema.properties.extension).toBeTruthy();
    const required = schema.required ?? [];
    expect(required).not.toContain("path");
    expect(required).not.toContain("extension");
    expect(schema.properties.path.description ?? "").toMatch(/optional/i);
    expect(schema.properties.extension.description ?? "").toMatch(/optional/i);
  });
});

// ── DB-backed conformance (skipped cleanly without DATABASE_URL) ─────────────

describe.skipIf(skip)("W4 conformance — DB-backed (codebase-mcp)", () => {
  let db: pg.Pool;
  let filtClient: McpClient;
  let steerClient: McpClient;
  let mergeClient: McpClient;

  async function insertFile(
    repo: string,
    projectId: string,
    path: string,
    language: string | null,
  ): Promise<string> {
    const { rows } = await db.query<{ id: string }>(
      `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (repository_id, path)
         DO UPDATE SET content_sha256 = EXCLUDED.content_sha256
       RETURNING id`,
      [repo, projectId, path, language, `conf-w4-sha-${repo}-${path}`],
    );
    return rows[0].id;
  }

  async function insertChunk(c: {
    id: string;
    repo: string;
    projectId: string;
    path: string;
    language: string;
    symbolName: string | null;
    startLine: number;
    endLine: number;
    content: string;
    embedKey?: string | null;
  }): Promise<void> {
    const fileId = await insertFile(c.repo, c.projectId, c.path, c.language);
    await db.query(
      `INSERT INTO codebase.code_chunks
         (id, file_id, repository_id, project_id, file_path, language,
          symbol_name, start_line, end_line, content, content_sha256, embedding)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::halfvec)`,
      [
        c.id,
        fileId,
        c.repo,
        c.projectId,
        c.path,
        c.language,
        c.symbolName,
        c.startLine,
        c.endLine,
        c.content,
        `conf-w4-chunk-sha-${c.id}`,
        c.embedKey ? `[${H.fakeVec(c.embedKey).join(",")}]` : null,
      ],
    );
  }

  async function cleanup(): Promise<void> {
    await db.query(
      `DELETE FROM codebase.code_chunks WHERE project_id = ANY($1) OR repository_id = ANY($2)`,
      [ALL_PROJECTS, ALL_REPOS],
    );
    await db.query(
      `DELETE FROM codebase.files WHERE project_id = ANY($1) OR repository_id = ANY($2)`,
      [ALL_PROJECTS, ALL_REPOS],
    );
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Idempotent migration apply (HOLD files skipped) — self-sufficient on a fresh DB.
    const { readFileSync, readdirSync } = await import("node:fs");
    const { dirname, join } = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    await db.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    const sqlDir = join(dirname(fileURLToPath(import.meta.url)), "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();

    // ── P_CFILT: per-arm filter probes ───────────────────────────────────────
    await insertChunk({
      id: K_TS,
      repo: R_FILT,
      projectId: P_CFILT,
      path: "src/graph/traverse.ts",
      language: "typescript",
      symbolName: "frontierTraversal",
      startLine: 1,
      endLine: 40,
      content:
        "export function frontierTraversal() {\n  // frontier traversal batching queue for graph expansion\n}",
      embedKey: "w4c:ts",
    });
    await insertChunk({
      id: K_MD_BM25,
      repo: R_FILT,
      projectId: P_CFILT,
      path: "docs/frontier.md",
      language: "markdown",
      symbolName: null,
      startLine: 1,
      endLine: 20,
      content:
        "# Frontier notes\nfrontier traversal batching queue documentation notes",
      embedKey: null, // BM25 arm ONLY
    });
    await insertChunk({
      id: K_MD_VEC,
      repo: R_FILT,
      projectId: P_CFILT,
      path: "docs/other.md",
      language: "markdown",
      symbolName: null,
      startLine: 1,
      endLine: 20,
      content: "completely unrelated zebra prose with different words",
      embedKey: `code:${CQ}`, // identical to the query vector → vector arm ONLY
    });
    await insertChunk({
      id: K_PCT,
      repo: R_FILT,
      projectId: P_CFILT,
      path: "src/pct/100%file.ts",
      language: "typescript",
      symbolName: "pctProbe",
      startLine: 1,
      endLine: 30,
      content: "// frontier traversal batching queue percent path probe",
      embedKey: "w4c:pct",
    });
    await insertChunk({
      id: K_WILD,
      repo: R_FILT,
      projectId: P_CFILT,
      path: "src/noXmatch/wild.ts",
      language: "typescript",
      symbolName: "wildProbe",
      startLine: 1,
      endLine: 30,
      content: "// frontier traversal batching queue wildcard path probe",
      embedKey: "w4c:wild",
    });
    await insertChunk({
      id: K_OTH,
      repo: R_OTHER,
      projectId: P_CFILT,
      path: "src/graph/other-repo.ts",
      language: "typescript",
      symbolName: "otherRepoProbe",
      startLine: 1,
      endLine: 30,
      content: "// frontier traversal batching queue in the other repository",
      embedKey: "w4c:oth",
    });

    // ── P_STEER: 8 distinct-file chunks for the steering boundary ───────────
    for (let i = 1; i <= 8; i++) {
      await insertChunk({
        id: cid(20 + i),
        repo: R_STEER,
        projectId: P_STEER,
        path: `src/steer/file${i}.ts`,
        language: "typescript",
        symbolName: `steer${i}`,
        startLine: 1,
        endLine: 30,
        content: `// steering conformance shared beacon token file ${i} ${"pad ".repeat(i)}`,
        embedKey: `w4c:steer:${i}`,
      });
    }

    // ── P_MERGE: 3 chunker-shaped overlapping chunks of ONE file ────────────
    const mergeRanges: Array<[number, number]> = [
      [1, 60],
      [49, 108],
      [97, 156],
    ];
    for (let i = 0; i < mergeRanges.length; i++) {
      await insertChunk({
        id: cid(40 + i),
        repo: R_MERGE,
        projectId: P_MERGE,
        path: "src/big/module.ts",
        language: "typescript",
        symbolName: `segment${i}`,
        startLine: mergeRanges[i][0],
        endLine: mergeRanges[i][1],
        content: `// gigantic merged module beacon segment ${i} ${"pad ".repeat(i + 1)}`,
        embedKey: `w4c:merge:${i}`,
      });
    }

    filtClient = await connectClient(P_CFILT);
    steerClient = await connectClient(P_STEER);
    mergeClient = await connectClient(P_MERGE);
  });

  afterAll(async () => {
    await filtClient?.close();
    await steerClient?.close();
    await mergeClient?.close();
    await cleanup();
    await db.end();
  });

  // ── AC-201: filters hit BOTH RRF arms of the fused pool ────────────────────

  it("AC-201 sanity: the unfiltered pool reaches all three probe chunks", async () => {
    const pool = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
    });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(K_TS);
    expect(ids).toContain(K_MD_BM25); // BM25 arm works without a vector
    expect(ids).toContain(K_MD_VEC); // vector arm works without lexical overlap
  });

  it("AC-201: extension filter excludes non-matching chunks from BOTH arms", async () => {
    const pool = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      extension: "ts",
    });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(K_TS);
    expect(ids).not.toContain(K_MD_BM25); // would leak if only the vector arm were filtered
    expect(ids).not.toContain(K_MD_VEC); // would leak if only the BM25 arm were filtered
    for (const c of pool) expect(c.file_path.endsWith(".ts")).toBe(true);
  });

  it("AC-201: a leading dot on extension is normalized; both arms still contribute", async () => {
    const dotted = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      extension: ".md",
    });
    const bare = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      extension: "md",
    });
    const dottedIds = dotted.map((c) => c.id).sort();
    expect(dottedIds).toEqual(bare.map((c) => c.id).sort());
    expect(dottedIds).toEqual([K_MD_BM25, K_MD_VEC].sort()); // one per arm
  });

  it("AC-201: path is a substring filter and composes with the repo scoping filter", async () => {
    const pool = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      path: "src/graph",
    });
    const ids = pool.map((c) => c.id);
    expect(ids).toContain(K_TS);
    expect(ids).toContain(K_OTH); // same substring, other repo, same project
    for (const c of pool) expect(c.file_path).toContain("src/graph");

    const scoped = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      repo: R_FILT,
      path: "src/graph",
    });
    expect(scoped.map((c) => c.id)).toEqual([K_TS]);
  });

  it("AC-201: LIKE wildcards in path input are escaped (no injection)", async () => {
    // A literal `%` in the stored path is matchable…
    const pct = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      path: "100%file",
    });
    expect(pct.map((c) => c.id)).toEqual([K_PCT]);
    // …but `%` in user input must NOT act as a wildcard ("no%match" ≠ "noXmatch")…
    const injected = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      path: "no%match",
    });
    expect(injected).toEqual([]);
    // …and neither must `_` ("no_match" ≠ "noXmatch" — wait, `_` matches ONE char:
    // "noXmatch" would match "no_match" if unescaped).
    const underscore = await search.fuseCodeCandidates({
      projectId: P_CFILT,
      query: CQ,
      path: "no_match",
    });
    expect(underscore.map((c) => c.id)).not.toContain(K_WILD);
  });

  // ── AC-202: zero-pool filtered search → one unfiltered retry + notice ──────

  it("AC-202: impossible filter retries unfiltered and reports it", async () => {
    const result = await search.searchCode({
      projectId: P_CFILT,
      query: CQ,
      path: "zzz/absent-path",
      k: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(Object.keys(result.droppedFilters)).toContain("path");
  });

  it("AC-202: scoping filter `repo` is KEPT on the retry", async () => {
    const result = await search.searchCode({
      projectId: P_CFILT,
      query: CQ,
      repo: R_FILT,
      extension: "zzz",
      k: 10,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(result.hits.map((h) => h.chunkId)).not.toContain(K_OTH);
  });

  it("AC-202: zero-corpus unfiltered query returns plain no-results, no retry", async () => {
    const result = await search.searchCode({
      projectId: P_EMPTY,
      query: "anything at all",
      k: 5,
    });
    expect(result.hits).toEqual([]);
    expect(result.retriedWithoutFilters).toBe(false);
    const emptyClient = await connectClient(P_EMPTY);
    const { text } = await callTool(emptyClient, "code_search", {
      query: "anything at all",
      k: 5,
    });
    await emptyClient.close();
    expect(text).not.toContain("Note: no results matched filters");
    expect(text).not.toMatch(/^1\. /m);
  });

  it("AC-202 (MCP): the notice line is prepended and names the dropped filter(s)", async () => {
    const { text, isError } = await callTool(filtClient, "code_search", {
      query: CQ,
      path: "zzz/absent-path",
      k: 5,
    });
    expect(isError).toBe(false);
    const first = text.split("\n")[0];
    expect(first.startsWith("Note: no results matched filters")).toBe(true);
    expect(first).toContain("path");
    expect(first).toContain("zzz/absent-path");
    expect(first).toContain("showing unfiltered results");
    expect(first).toMatch(/filters are optional/i);
  });

  // ── AC-203: rank-only output — no raw scores anywhere ──────────────────────

  it("AC-203: public hits carry no `score` key", async () => {
    const result = await search.searchCode({
      projectId: P_CFILT,
      query: CQ,
      k: 5,
    });
    expect(result.hits.length).toBeGreaterThan(0);
    for (const h of result.hits) {
      expect("score" in h).toBe(false);
      expect(typeof h.chunkId).toBe("string");
    }
  });

  it("AC-203 (MCP): response text has ranks + identifiers but no score text", async () => {
    // k=10 ≥ pool size (6) so every fixture id is listed regardless of the
    // deterministic rerank ordering.
    const { text } = await callTool(filtClient, "code_search", {
      query: CQ,
      k: 10,
    });
    expect(text).toMatch(/^1\. /m); // ranks
    expect(text).toContain("src/graph/traverse.ts"); // identifiers
    expect(text).toMatch(/\S+:\d+-\d+/); // path:start-end identifier shape
    expect(text).not.toMatch(/score/i);
    expect(text).not.toMatch(/0\.\d{4}/);
  });

  // ── AC-204: truncation-steering boundary ───────────────────────────────────

  it("AC-204: k=3 over an 8-chunk pool ends with the steering line", async () => {
    const direct = await search.searchCode({
      projectId: P_STEER,
      query: "steering conformance shared beacon token",
      k: 3,
    });
    expect(direct.hits).toHaveLength(3);
    expect(direct.poolSize).toBe(8);
    expect(direct.truncated).toBe(true);

    const { text } = await callTool(steerClient, "code_search", {
      query: "steering conformance shared beacon token",
      k: 3,
    });
    const last = text.trimEnd().split("\n").at(-1) ?? "";
    expect(last).toContain("Showing top 3 of 8 candidates");
    expect(last).toMatch(/optionally/i);
    expect(last).toContain("repo");
    expect(last).toContain("path");
    expect(last).toContain("extension");
    expect(last).toContain("language");
    expect(last).toContain("code_get_file");
  });

  it("AC-204: k ≥ pool size appends no steering line", async () => {
    const direct = await search.searchCode({
      projectId: P_STEER,
      query: "steering conformance shared beacon token",
      k: 20,
    });
    expect(direct.truncated).toBe(false);

    const { text } = await callTool(steerClient, "code_search", {
      query: "steering conformance shared beacon token",
      k: 20,
    });
    expect(text).not.toContain("Showing top");
  });

  // ── AC-206 end-to-end: post-rerank merge through the live pipeline ─────────

  it("AC-206 (e2e): overlapping same-file top-k chunks come back merged and capped", async () => {
    const result = await search.searchCode({
      projectId: P_MERGE,
      query: "gigantic merged module beacon segment",
      k: 5,
    });
    // 3 chunker-overlapped chunks (1-60, 49-108, 97-156) under a 120-line cap
    // always reduce to exactly 2 hits, whichever chunk leads the rerank.
    expect(result.hits).toHaveLength(2);
    const merged = result.hits.filter((h) => (h.mergedCount ?? 1) > 1);
    expect(merged).toHaveLength(1);
    for (const h of result.hits) {
      expect(h.filePath).toBe("src/big/module.ts");
      expect(h.startLine).toBeGreaterThanOrEqual(1);
      expect(h.endLine).toBeLessThanOrEqual(156);
      if ((h.mergedCount ?? 1) > 1) {
        expect(h.endLine - h.startLine).toBeLessThanOrEqual(120);
      }
    }
    // Merging never creates overlap beyond the chunker's pre-existing 12-line
    // overlap (see the fairness note in the pure chain test: the literal
    // "never overlap" plan property is unsatisfiable on cap-split chains).
    const sorted = [...result.hits].sort((a, b) => a.startLine - b.startLine);
    expect(sorted[0].endLine - sorted[1].startLine).toBeLessThanOrEqual(12 - 1);

    // The formatted response renders the merged union range.
    const { text } = await callTool(mergeClient, "code_search", {
      query: "gigantic merged module beacon segment",
      k: 5,
    });
    expect(text).toContain(
      `src/big/module.ts:${merged[0].startLine}-${merged[0].endLine}`,
    );
  });
});
