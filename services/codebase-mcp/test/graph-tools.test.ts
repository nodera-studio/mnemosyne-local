// Wave 4 — graph MCP tool behavior (AC-009/010/011). Drives the real buildServer() over an
// in-memory MCP transport and asserts: the new graph tools are registered; code_graph_expand
// returns def+callers+callees and labels the ambiguous `format` target; an un-graphed seed
// falls back to a file read (AC-009); code_symbol_lookup returns all matches for a duplicated
// name; code_trace_path returns the route→handler→service→repo chain; and code_search /
// code_get_file / code_index_status are UNCHANGED (AC-011). Requires DATABASE_URL.

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const here = dirname(fileURLToPath(import.meta.url));
const fixturesRoot = join(here, "fixtures"); // <root>/sample-repo/route.ts etc.
const sampleRepoDir = join(fixturesRoot, "sample-repo");
const REPO = "sample-repo"; // repo dir name === REPOS_ROOT-relative dir, so file reads resolve
const PROJ = "tools-proj";

// REPOS_ROOT must be set BEFORE config.js is imported (it reads process.env at module load).
process.env.REPOS_ROOT = fixturesRoot;
process.env.DEFAULT_PROJECT_ID = PROJ;
process.env.VOYAGE_API_KEY = "test-key"; // buildServer never embeds, but config is shared

// Hermetic embedder so indexing the corpus needs no Voyage quota.
vi.mock("../src/voyage.js", () => ({
  embedCode: vi.fn(async (texts: string[]) =>
    texts.map(() => Array.from({ length: 1024 }, () => 0.001)),
  ),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
  rerank: vi.fn(async (_q: string, docs: string[]) =>
    docs.map((_d, i) => ({ index: i, score: 1 - i * 0.01 })),
  ),
}));

const { indexRepo } = await import("../src/indexer.js");
const { buildServer } = await import("../src/server.js");
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } =
  await import("@modelcontextprotocol/sdk/inMemory.js");

type ToolText = { text: string; isError: boolean };

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
): Promise<ToolText> {
  const res = (await client.callTool({ name, arguments: args })) as {
    isError?: boolean;
    content: { type: string; text: string }[];
  };
  return {
    text: res.content.map((c) => c.text).join("\n"),
    isError: res.isError === true,
  };
}

describe.skipIf(skip)("graph MCP tools (AC-009/010/011)", () => {
  let pool: pg.Pool;
  let client: Awaited<ReturnType<typeof connectClient>>;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL });
    const { readFileSync, readdirSync } = await import("node:fs");
    const sqlDir = join(here, "..", "sql");
    await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      // Skip HOLD migrations (e.g. 005_drop_bakeoff_scratch.sql) like the real runner.
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
    await indexRepo(sampleRepoDir, REPO, PROJ);
    client = await connectClient();
  });

  afterAll(async () => {
    await client?.close();
    await pool.query("DELETE FROM codebase.index_runs WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.query("DELETE FROM codebase.files WHERE repository_id=$1", [
      REPO,
    ]);
    await pool.end();
  });

  it("registers exactly the expected tool set (AC-011: originals unchanged + 3 added)", async () => {
    const { tools } = await client.listTools();
    const names = tools.map((t) => t.name).sort();
    expect(names).toEqual(
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
  });

  it("code_get_file input schema is unchanged (AC-011)", async () => {
    const { tools } = await client.listTools();
    const getFile = tools.find((t) => t.name === "code_get_file")!;
    const props = Object.keys(
      (getFile.inputSchema as { properties: Record<string, unknown> })
        .properties,
    ).sort();
    expect(props).toEqual(["endLine", "path", "repo", "startLine"]);
  });

  it("code_search input schema: originals + wave-4 optional refinements (path, extension)", async () => {
    const { tools } = await client.listTools();
    const search = tools.find((t) => t.name === "code_search")!;
    const props = Object.keys(
      (search.inputSchema as { properties: Record<string, unknown> })
        .properties,
    ).sort();
    expect(props).toEqual([
      "extension",
      "k",
      "language",
      "path",
      "query",
      "repo",
    ]);
  });

  it("code_get_file still reads a file range exactly as before (AC-011)", async () => {
    const out = await callTool(client, "code_get_file", {
      repo: REPO,
      path: "route.ts",
      startLine: 5,
      endLine: 5,
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("export function route");
    expect(out.text).toContain("route.ts (lines 5-5 of");
  });

  it("code_graph_expand returns def + callees for a symbol", async () => {
    const out = await callTool(client, "code_graph_expand", {
      repo: REPO,
      symbol: "route",
      direction: "callees",
      depth: 4,
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("route");
    expect(out.text).toContain("handleGetUser");
    expect(out.text).toContain("getUser");
    expect(out.text).toContain("findUser");
    expect(out.text).toContain("[d0 def]");
  });

  it("code_graph_expand labels the ambiguous `format` target (AC-010)", async () => {
    const out = await callTool(client, "code_graph_expand", {
      repo: REPO,
      symbol: "useFormat",
      direction: "callees",
      depth: 2,
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("(name-matched, may be ambiguous)");
    // both format defs surface.
    expect(out.text).toContain("ambiguous.ts");
    expect(out.text).toContain("format-alt.ts");
  });

  it("code_graph_expand falls back to a file read when the graph is empty (AC-009)", async () => {
    // A repo that was never indexed (no symbols), but the file exists under REPOS_ROOT.
    const out = await callTool(client, "code_graph_expand", {
      repo: REPO,
      symbol: "nonexistentSymbol",
      file: "route.ts",
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("graph had no edges");
    expect(out.text).toContain("export function route"); // the file body was read
  });

  it("code_symbol_lookup returns all matches for a duplicated name (AC-010 surface)", async () => {
    const out = await callTool(client, "code_symbol_lookup", {
      repo: REPO,
      name: "format",
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("2 matches");
    expect(out.text).toContain("may be ambiguous");
    expect(out.text).toContain("ambiguous.ts");
    expect(out.text).toContain("format-alt.ts");
  });

  it("code_symbol_lookup returns a single match without an ambiguity note", async () => {
    const out = await callTool(client, "code_symbol_lookup", {
      repo: REPO,
      name: "route",
    });
    expect(out.isError).toBe(false);
    expect(out.text).toContain("route");
    expect(out.text).not.toContain("matches for");
  });

  it("code_trace_path returns the route→handler→service→repo chain", async () => {
    const out = await callTool(client, "code_trace_path", {
      repo: REPO,
      symbol: "route",
      to: "findUser",
      depth: 4,
    });
    expect(out.isError).toBe(false);
    // ordered chain: each name appears, in order, before the next.
    const idxRoute = out.text.indexOf("route");
    const idxHandler = out.text.indexOf("handleGetUser");
    const idxService = out.text.indexOf("getUser");
    const idxRepo = out.text.indexOf("findUser");
    expect(idxRoute).toBeGreaterThanOrEqual(0);
    expect(idxHandler).toBeGreaterThan(idxRoute);
    expect(idxService).toBeGreaterThan(idxHandler);
    expect(idxRepo).toBeGreaterThan(idxService);
  });

  it("code_index_status still reports the indexed repo (AC-011)", async () => {
    const out = await callTool(client, "code_index_status", { repo: REPO });
    expect(out.isError).toBe(false);
    expect(out.text).toContain(REPO);
  });
});
