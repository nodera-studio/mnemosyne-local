// Wave 4 (WS2) — token shaping, codebase side. Covers:
//   AC-201 — path/extension filters are WHERE clauses in both RRF arms
//   AC-202 — zero-pool filtered search retries once unfiltered + notice
//   AC-203 — formatted output is rank-only
//   AC-204 — truncation appends the steering line naming optional filters
//   AC-205 — descriptions carry Use when / Do NOT use clauses

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
  // Probe for the query-embed boundary: counts embedCode invocations (MEDIUM-001 —
  // a retried search must embed exactly once).
  const embedProbe = { calls: 0 };
  return { fakeVec, fakeRerank, embedProbe };
});

vi.mock("../src/voyage.js", () => ({
  embedCode: async (texts: string[]) => {
    H.embedProbe.calls += 1;
    return texts.map((t) => H.fakeVec(`code:${t}`));
  },
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`ctx:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import { TOOL_DESCRIPTIONS, buildServer } from "../src/server.js";
import { formatHits, fuseCodeCandidates, searchCode } from "../src/search.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

const here = dirname(fileURLToPath(import.meta.url));
const PROJ = "ws2-code-shaping";
const EMPTY_PROJ = "ws2-code-shaping-empty";
const MERGE_PROJ = "ws2-code-merge-e2e";
const REPO_A = "ws2-repo-a";
const REPO_B = "ws2-repo-b";
const REPO_M = "ws2-repo-merge";
const cid = (n: number) => `30000000-0000-4000-8000-00000000000${n}`;
// Merge-e2e fixture ids (own prefix — the cid template only fits one digit).
const MID_A = "31000000-0000-4000-8000-000000000001";
const MID_B = "31000000-0000-4000-8000-000000000002";

interface ChunkFixture {
  id: string;
  repo: string;
  path: string;
  language: string;
  startLine: number;
  endLine: number;
  content: string;
}

const CHUNKS: ChunkFixture[] = [
  {
    id: cid(1),
    repo: REPO_A,
    path: "src/graph/expand.ts",
    language: "typescript",
    startLine: 1,
    endLine: 40,
    content:
      "gateway graph expansion walks callers and callees with a depth cap",
  },
  {
    id: cid(2),
    repo: REPO_A,
    path: "src/graph/trace.ts",
    language: "typescript",
    startLine: 41,
    endLine: 80,
    content: "gateway graph trace finds a downstream route to a target symbol",
  },
  {
    id: cid(3),
    repo: REPO_A,
    path: "src/auth/session.ts",
    language: "typescript",
    startLine: 1,
    endLine: 40,
    content: "gateway session auth validates bearer tokens and request origin",
  },
  {
    id: cid(4),
    repo: REPO_A,
    path: "docs/gateway.md",
    language: "markdown",
    startLine: 1,
    endLine: 30,
    content: "gateway documentation explains graph search and file retrieval",
  },
  {
    id: cid(5),
    repo: REPO_A,
    path: "src/weird%name.ts",
    language: "typescript",
    startLine: 1,
    endLine: 20,
    content: "gateway literal percent path should require escaping",
  },
  {
    id: cid(6),
    repo: REPO_A,
    path: "src/weirdXname.ts",
    language: "typescript",
    startLine: 1,
    endLine: 20,
    content: "gateway wildcard control path should not match a percent literal",
  },
  {
    id: cid(7),
    repo: REPO_B,
    path: "scripts/gateway.py",
    language: "python",
    startLine: 1,
    endLine: 40,
    content: "gateway python helper emits a usage report",
  },
  {
    id: cid(8),
    repo: REPO_B,
    path: "README.md",
    language: "markdown",
    startLine: 1,
    endLine: 20,
    content: "gateway readme has setup notes",
  },
];

describe("codebase tool descriptions (AC-205)", () => {
  it("every description has disjoint Use when and Do NOT use clauses", () => {
    expect(Object.keys(TOOL_DESCRIPTIONS).sort()).toEqual(
      [
        "code_get_file",
        "code_graph_expand",
        "code_index_status",
        "code_reindex",
        "code_search",
        "code_symbol_lookup",
        "code_trace_path",
      ].sort(),
    );
    for (const [name, description] of Object.entries(TOOL_DESCRIPTIONS)) {
      expect(description, name).toContain("Use when:");
      expect(description, name).toContain("Do NOT use");
      expect(description, name).toContain("Example:");
    }
    expect(TOOL_DESCRIPTIONS.code_search).toContain("optionally narrow");
  });

  it("LOW-002: repo/language are strict scoping; path/extension are droppable refinements", async () => {
    const server = buildServer("ws2-code-desc");
    const [clientTransport, serverTransport] =
      InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test", version: "1.0.0" });
    await Promise.all([
      client.connect(clientTransport),
      server.connect(serverTransport),
    ]);
    try {
      const { tools } = await client.listTools();
      const cs = tools.find((t) => t.name === "code_search")!;
      const props = (
        cs.inputSchema as {
          properties: Record<string, { description?: string }>;
        }
      ).properties;
      // Scoping params are kept on the zero-result retry, never auto-dropped…
      for (const strict of ["repo", "language"]) {
        expect(props[strict].description, strict).toContain("strict scoping");
        expect(props[strict].description, strict).toContain(
          "never auto-dropped",
        );
      }
      // …while path/extension are the droppable refinements the retry sheds.
      for (const droppable of ["path", "extension"]) {
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

describe.skipIf(skip)("code token shaping (AC-201/202/203/204)", () => {
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
    await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
      PROJ,
    ]);
    const fileIds = new Map<string, string>();
    for (const f of CHUNKS) {
      const { rows } = await pool.query<{ id: string }>(
        `INSERT INTO codebase.files
           (repository_id, project_id, path, language, content_sha256)
         VALUES ($1, $2, $3, $4, $5)
         RETURNING id`,
        [f.repo, PROJ, f.path, f.language, `file-${f.id}`],
      );
      fileIds.set(`${f.repo}:${f.path}`, rows[0].id);
    }
    for (const c of CHUNKS) {
      await pool.query(
        `INSERT INTO codebase.code_chunks
           (id, file_id, repository_id, project_id, file_path, language,
            symbol_name, start_line, end_line, content, content_sha256, embedding)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::halfvec)`,
        [
          c.id,
          fileIds.get(`${c.repo}:${c.path}`),
          c.repo,
          PROJ,
          c.path,
          c.language,
          c.path,
          c.startLine,
          c.endLine,
          c.content,
          `chunk-${c.id}`,
          `[${H.fakeVec(`chunk:${c.id}`).join(",")}]`,
        ],
      );
    }

    // AC-206 e2e fixture: TWO adjacent chunks of ONE file (1-40 touches 41-80) in
    // their own project, so both land in top-k and must come back as one merged hit.
    await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
      MERGE_PROJ,
    ]);
    const { rows: mergeFile } = await pool.query<{ id: string }>(
      `INSERT INTO codebase.files
         (repository_id, project_id, path, language, content_sha256)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING id`,
      [REPO_M, MERGE_PROJ, "src/merged/module.ts", "typescript", "file-merge"],
    );
    const mergeChunks: Array<[string, number, number]> = [
      [MID_A, 1, 40],
      [MID_B, 41, 80],
    ];
    for (const [chunkId, startLine, endLine] of mergeChunks) {
      await pool.query(
        `INSERT INTO codebase.code_chunks
           (id, file_id, repository_id, project_id, file_path, language,
            symbol_name, start_line, end_line, content, content_sha256, embedding)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::halfvec)`,
        [
          chunkId,
          mergeFile[0].id,
          REPO_M,
          MERGE_PROJ,
          "src/merged/module.ts",
          "typescript",
          null,
          startLine,
          endLine,
          `// adjacent beacon segment lines ${startLine}-${endLine}`,
          `chunk-${chunkId}`,
          `[${H.fakeVec(`chunk:${chunkId}`).join(",")}]`,
        ],
      );
    }
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.files WHERE project_id = ANY($1)", [
      [PROJ, MERGE_PROJ],
    ]);
    await pool.end();
  });

  it("AC-201: extension filter excludes non-matching chunks from the pool", async () => {
    const pool_ = await fuseCodeCandidates({
      projectId: PROJ,
      query: "gateway",
      extension: "ts",
    });
    expect(pool_.length).toBeGreaterThan(0);
    for (const c of pool_) expect(c.file_path.endsWith(".ts")).toBe(true);
  });

  it("AC-201: path substring scopes the pool and composes with repo", async () => {
    const pool_ = await fuseCodeCandidates({
      projectId: PROJ,
      query: "gateway",
      repo: REPO_A,
      path: "src/graph",
    });
    expect(new Set(pool_.map((c) => c.file_path))).toEqual(
      new Set(["src/graph/expand.ts", "src/graph/trace.ts"]),
    );
  });

  it("AC-201: LIKE wildcards in path input are escaped", async () => {
    const pool_ = await fuseCodeCandidates({
      projectId: PROJ,
      query: "gateway",
      path: "weird%",
    });
    expect(pool_.map((c) => c.file_path)).toEqual(["src/weird%name.ts"]);
  });

  it("AC-202: impossible path retries unfiltered and renders the notice first", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      path: "no/such/path",
      k: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(formatHits(result.hits, result).split("\n")[0]).toBe(
      "Note: no results matched filters {path=no/such/path}; showing unfiltered results. Filters are optional — retry with different values to narrow.",
    );
  });

  it("AC-202: retry drops path/extension but keeps repo/language scoping", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      repo: REPO_B,
      language: "python",
      extension: "nope",
      k: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBe(1);
    expect(result.hits[0].filePath).toBe("scripts/gateway.py");
  });

  it("AC-202: filtered zero-corpus retry suppresses the unfiltered-results notice", async () => {
    const result = await searchCode({
      projectId: EMPTY_PROJ,
      query: "gateway",
      path: "no/such/path",
      k: 5,
    });
    expect(result.hits).toEqual([]);
    expect(result.retriedWithoutFilters).toBe(true);
    expect(formatHits(result.hits, result)).toBe("No matching code.");
  });

  it("MEDIUM-001: the zero-pool retry reuses the query embedding — exactly ONE embed call", async () => {
    H.embedProbe.calls = 0;
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      path: "no/such/path",
      k: 5,
    });
    expect(result.retriedWithoutFilters).toBe(true);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(H.embedProbe.calls).toBe(1);
  });

  it("AC-206 (e2e): two adjacent same-file top-k chunks merge to one hit with the union range and spans note", async () => {
    const result = await searchCode({
      projectId: MERGE_PROJ,
      query: "adjacent beacon segment",
      k: 5,
    });
    expect(result.hits).toHaveLength(1);
    const [merged] = result.hits;
    expect(merged.filePath).toBe("src/merged/module.ts");
    expect(merged.startLine).toBe(1);
    expect(merged.endLine).toBe(80);
    expect(merged.mergedCount).toBe(2);
    expect([MID_A, MID_B]).toContain(merged.chunkId); // leader = best-ranked constituent
    expect(formatHits(result.hits, result).split("\n")[0]).toBe(
      `1. src/merged/module.ts:1-80 (spans 2 chunks)  [${merged.chunkId}]`,
    );
  });

  it("AC-203: formatted output contains no raw scores", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      k: 5,
    });
    const text = formatHits(result.hits, result);
    expect(text).toMatch(/^1\. /);
    expect(text).toContain(result.hits[0].chunkId);
    expect(text).not.toMatch(/score/i);
    expect(text).not.toMatch(/\b0\.\d{4}\b/);
    for (const h of result.hits) expect("score" in h).toBe(false);
  });

  it("AC-204: truncated result appends the code steering line", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      k: 3,
    });
    expect(result.poolSize).toBe(8);
    expect(result.truncated).toBe(true);
    const lines = formatHits(result.hits, result).split("\n");
    expect(lines[lines.length - 1]).toBe(
      "Showing top 3 of 8 candidates. Prefer several small targeted searches — optionally narrow with repo, path, extension, or language. Fetch full context with code_get_file.",
    );
  });

  it("AC-204: k ≥ pool size → no steering line", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "gateway",
      k: 20,
    });
    expect(result.truncated).toBe(false);
    expect(formatHits(result.hits, result)).not.toContain("Showing top");
  });
});
