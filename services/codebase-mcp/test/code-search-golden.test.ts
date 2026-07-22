// Golden-output pin for searchCode (AC-101). Recorded BEFORE the two-phase
// fuseCodeCandidates/rerankCodeHits split so the split can prove "no behavior change":
// the exact ordered chunk-id arrays below MUST stay green, unchanged, after the refactor.
//
// Deterministic throughout: Voyage is module-mocked (query vectors + rerank scores
// derived from seeded PRNGs over the input strings), fixture embeddings are inserted as
// literals with strictly distinct cosine orderings (≤10 rows → seq scan; plan changes
// cannot flip ranks). searchCode has no recency term, so no clock faking is needed.
// Also pins the rerank-empty fallback (order = RRF identity at search.ts) via a
// sentinel query the mock answers with [].
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

// Deterministic helpers, hoisted for the vi.mock factory (hoisted above imports).
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
  /** Sentinel: queries containing this string make the rerank mock return [] so the
   *  RRF-order fallback (search.ts `order = rows` identity) gets pinned too. */
  const RERANK_EMPTY = "RERANK_EMPTY";
  function fakeRerank(
    query: string,
    docs: string[],
    topK: number,
  ): Array<{ index: number; score: number }> {
    if (query.includes(RERANK_EMPTY)) return [];
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
  return { fnv1a, mulberry32, fakeVec, fakeRerank, RERANK_EMPTY };
});

vi.mock("../src/voyage.js", () => ({
  embedCode: async (texts: string[]) =>
    texts.map((t) => H.fakeVec(`code:${t}`)),
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`code:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import { searchCode } from "../src/search.js";

const here = dirname(fileURLToPath(import.meta.url));

const PROJ = "golden-code";
const EMPTY_PROJ = "golden-code-empty";
const REPO_A = "golden-repo-a";
const REPO_B = "golden-repo-b";

const cid = (n: number) => `10000000-0000-4000-8000-00000000000${n}`;

interface ChunkFixture {
  id: string;
  repo: string;
  path: string;
  language: string;
  symbolName: string | null;
  startLine: number;
  endLine: number;
  content: string;
}

// 8 chunks across 3 files (2 repos, 2 languages), distinct content lengths so the
// deterministic rerank mock never ties.
const CHUNKS: ChunkFixture[] = [
  {
    id: cid(1),
    repo: REPO_A,
    path: "src/auth/password.ts",
    language: "typescript",
    symbolName: "hashPassword",
    startLine: 1,
    endLine: 12,
    content:
      "export async function hashPassword(plain: string): Promise<string> {\n  const salt = await bcrypt.genSalt(12);\n  return bcrypt.hash(plain, salt);\n}\n// bcrypt hash for credential storage",
  },
  {
    id: cid(2),
    repo: REPO_A,
    path: "src/auth/password.ts",
    language: "typescript",
    symbolName: "verifyPassword",
    startLine: 14,
    endLine: 24,
    content:
      "export async function verifyPassword(plain: string, hash: string): Promise<boolean> {\n  return bcrypt.compare(plain, hash);\n}",
  },
  {
    id: cid(3),
    repo: REPO_A,
    path: "src/auth/password.ts",
    language: "typescript",
    symbolName: "resetToken",
    startLine: 26,
    endLine: 40,
    content:
      "export function resetToken(): string {\n  // random token for the password reset email flow\n  return crypto.randomBytes(32).toString('hex');\n}",
  },
  {
    id: cid(4),
    repo: REPO_A,
    path: "src/db/pool.ts",
    language: "typescript",
    symbolName: "createPool",
    startLine: 1,
    endLine: 15,
    content:
      "export function createPool(url: string): Pool {\n  // database connection pool shared by every service module\n  return new Pool({ connectionString: url, max: 10 });\n}",
  },
  {
    id: cid(5),
    repo: REPO_A,
    path: "src/db/pool.ts",
    language: "typescript",
    symbolName: "withClient",
    startLine: 17,
    endLine: 30,
    content:
      "export async function withClient<T>(fn: (c: PoolClient) => Promise<T>): Promise<T> {\n  const client = await pool.connect();\n  try {\n    return await fn(client);\n  } finally {\n    client.release();\n  }\n}",
  },
  {
    id: cid(6),
    repo: REPO_A,
    path: "src/db/pool.ts",
    language: "typescript",
    symbolName: null,
    startLine: 32,
    endLine: 38,
    content:
      "// pool error listener: log and continue — a dropped idle connection is not fatal\npool.on('error', (err) => logger.error({ err }, 'pg pool error'));",
  },
  {
    id: cid(7),
    repo: REPO_B,
    path: "scripts/report.py",
    language: "python",
    symbolName: "generate_report",
    startLine: 1,
    endLine: 20,
    content:
      'def generate_report(month: str) -> Path:\n    """Generate the monthly usage report as a CSV."""\n    rows = fetch_usage(month)\n    return write_csv(rows)',
  },
  {
    id: cid(8),
    repo: REPO_B,
    path: "scripts/report.py",
    language: "python",
    symbolName: "fetch_usage",
    startLine: 22,
    endLine: 34,
    content:
      'def fetch_usage(month: str) -> list[dict]:\n    # query the warehouse for usage rows of the month\n    return db.execute(USAGE_SQL, {"month": month}).fetchall()',
  },
];

const byId = new Map(CHUNKS.map((c) => [c.id, c]));

/** The exact snippet shape searchCode produces (first 4 lines, then 240 chars). */
function expectedSnippet(chunkId: string): string {
  return byId
    .get(chunkId)!
    .content.split("\n")
    .slice(0, 4)
    .join("\n")
    .slice(0, 240);
}

describe.skipIf(skip)("searchCode golden pins (AC-101)", () => {
  let pool: pg.Pool;

  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    // Apply the migration chain idempotently (HOLD files skipped) — same pattern as
    // graph-tools.test.ts — so the test is self-sufficient on a fresh disposable DB.
    const sqlDir = join(here, "..", "sql");
    await pool.query("CREATE SCHEMA IF NOT EXISTS codebase;");
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

    // Seed parent files (code_chunks.file_id FK), then the chunks with literal vectors.
    const files = [
      { repo: REPO_A, path: "src/auth/password.ts", language: "typescript" },
      { repo: REPO_A, path: "src/db/pool.ts", language: "typescript" },
      { repo: REPO_B, path: "scripts/report.py", language: "python" },
    ];
    const fileIds = new Map<string, string>();
    for (const f of files) {
      const { rows } = await pool.query<{ id: string }>(
        `INSERT INTO codebase.files (repository_id, project_id, path, language, content_sha256)
         VALUES ($1, $2, $3, $4, 'sha-' || $3) RETURNING id`,
        [f.repo, PROJ, f.path, f.language],
      );
      fileIds.set(f.path, rows[0].id);
    }
    for (const c of CHUNKS) {
      await pool.query(
        `INSERT INTO codebase.code_chunks
           (id, file_id, repository_id, project_id, file_path, language, symbol_name,
            start_line, end_line, content, content_sha256, embedding)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::halfvec)`,
        [
          c.id,
          fileIds.get(c.path),
          c.repo,
          PROJ,
          c.path,
          c.language,
          c.symbolName,
          c.startLine,
          c.endLine,
          c.content,
          `sha-${c.id}`,
          `[${H.fakeVec(`chunk:${c.id}`).join(",")}]`,
        ],
      );
    }
  });

  afterAll(async () => {
    await pool.query("DELETE FROM codebase.files WHERE project_id = $1", [
      PROJ,
    ]);
    await pool.end();
  });

  /** Shared shape assertions for every pinned query. */
  function assertShape(
    hits: Awaited<ReturnType<typeof searchCode>>["hits"],
    k: number,
  ): void {
    expect(hits.length).toBeLessThanOrEqual(k);
    for (const h of hits) {
      const f = byId.get(h.chunkId)!;
      expect(h.filePath).toBe(f.path);
      expect(h.startLine).toBe(f.startLine);
      expect(h.endLine).toBe(f.endLine);
      expect(h.symbolName).toBe(f.symbolName);
      expect(h.language).toBe(f.language);
      expect(h.snippet).toBe(expectedSnippet(h.chunkId));
      expect("score" in h).toBe(false);
    }
  }

  it("pins query 1 (plain): 'hash password bcrypt'", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "hash password bcrypt",
      k: 5,
    });
    const { hits } = result;
    assertShape(hits, 5);
    expect(hits.map((h) => h.chunkId)).toEqual(GOLDEN_C1);
  });

  it("pins query 2 (repo-filtered): 'database pool connection' in golden-repo-a", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "database pool connection",
      repo: REPO_A,
      k: 5,
    });
    const { hits } = result;
    assertShape(hits, 5);
    for (const h of hits) expect(byId.get(h.chunkId)!.repo).toBe(REPO_A);
    expect(hits.map((h) => h.chunkId)).toEqual(GOLDEN_C2);
  });

  it("pins query 3 (language-filtered): 'generate monthly report' in python", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: "generate monthly report",
      language: "python",
      k: 5,
    });
    const { hits } = result;
    assertShape(hits, 5);
    for (const h of hits) expect(h.language).toBe("python");
    expect(hits.map((h) => h.chunkId)).toEqual(GOLDEN_C3);
  });

  it("pins the rerank-empty fallback: RRF order, rank-only output", async () => {
    const result = await searchCode({
      projectId: PROJ,
      query: `${H.RERANK_EMPTY} database pool`,
      k: 5,
    });
    const { hits } = result;
    assertShape(hits, 5);
    expect(hits.map((h) => h.chunkId)).toEqual(GOLDEN_C4);
  });

  it("pins the empty-corpus early return: []", async () => {
    const result = await searchCode({
      projectId: EMPTY_PROJ,
      query: "anything at all",
      k: 5,
    });
    expect(result.hits).toEqual([]);
  });
});

// ── Golden chunk-id orders (recorded 2026-07-03 against the pre-split searchCode) ────
// Do NOT regenerate casually: these pins are the AC-101 no-behavior-change guard for
// the fuseCodeCandidates/rerankCodeHits split. If a deliberate behavior change lands
// later (e.g. wave-4 token shaping / chunk merging), update them in the SAME commit.
const GOLDEN_C1: string[] = [cid(3), cid(8), cid(5), cid(2), cid(6)];
const GOLDEN_C2: string[] = [cid(5), cid(4), cid(3), cid(6), cid(2)];
// Only 2 of the 8 fixtures are language=python, so the filtered pin has 2 hits.
const GOLDEN_C3: string[] = [cid(8), cid(7)];
const GOLDEN_C4: string[] = [cid(8), cid(6), cid(4), cid(5), cid(2)];
