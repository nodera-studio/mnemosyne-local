// ─────────────────────────────────────────────────────────────────────────────
// CONFORMANCE SUITE — Waves 1–3 (codebase-mcp half)
//
// Plan-derived, implementation-blind behavior tests for the retrieval-improvement
// program (plan: .claude/plans/2026-07-03-retrieval-improvement-program/index.md).
// Assertions drafted from the EARS acceptance criteria BEFORE locating exported
// symbols; independent of the Implementer's own tests.
//
// AC map (codebase-mcp scope):
//   AC-101 — composed searchCode output contract (4-line/240-char snippet,
//            rank-only public hits, ≤k, filters, rerank-empty RRF fallback, [], and
//            two-phase composition ≡ one-shot searchCode)
//   AC-102 — fuseCodeCandidates pool: ≤ candidatePool rows, per-arm RRF ranks,
//            live handler consumes exactly this pool (finals ⊆ pool)
//   AC-103 — ndcgAtK / pairedBootstrapCI vs hand-computed values; determinism
//   AC-104 — retrievalConfig() full snapshot; runCodeEval artifact carries the
//            snapshot + per-query scores for BOTH layers; test/runs/ convention
//   AC-105 — gate math semantics (CI regression side, id-drift fails loudly)
//   AC-108 — distill-eval: npm-run entry, side-effect-free import, main-guard
//   AC-109 — code-eval.json v2 (changelog header, id/split/provenance rows);
//            distilled candidates approved:false, gold file untouched
//   AC-401 — expand() routes depth ≥5 to app-side BFS (usesBfs/threshold
//            constants), deep expands terminate on cycles, per-level cap holds
//   AC-402 — import edges excluded from call expansion in BOTH engines;
//            BFS ≡ CTE parity on a fixture that CONTAINS import edges
//   AC-403 — resolveEdges: heavy name-resolution JOINs run OUTSIDE the
//            transaction; the txn window is DELETE + INSERT-from-staged only
//   AC-404 — re-indexing unchanged content makes ZERO embed calls (chunk-level
//            content_sha256 vector reuse); only changed chunks re-embed
//
// Deterministic throughout: Voyage is module-mocked (seeded PRNG vectors/scores,
// call-counting embedCode). DB tests are self-contained under conf-w13-* ids and
// clean up. NO live Voyage/Anthropic quota is ever spent here.
// ─────────────────────────────────────────────────────────────────────────────

import {
  appendFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

// Deterministic helpers + the embed call log, hoisted for the vi.mock factory.
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
  /** Sentinel: queries containing this make the rerank mock return [] so the
   *  RRF-order fallback is observable. */
  const RERANK_EMPTY = "CONF_RERANK_EMPTY";
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
  /** Every text handed to embedCode, in call order (AC-404 counting). */
  const embedLog: string[] = [];
  return { fnv1a, mulberry32, fakeVec, fakeRerank, RERANK_EMPTY, embedLog };
});

// HARD RULE: never call live Voyage — module-mock the boundary.
vi.mock("../src/voyage.js", () => ({
  embedCode: async (texts: string[]) => {
    H.embedLog.push(...texts);
    return texts.map((t) => H.fakeVec(`code:${t}`));
  },
  embedCodeContextual: async (docs: string[][]) =>
    docs.map((d) => d.map((t) => H.fakeVec(`code:${t}`))),
  rerank: async (query: string, docs: string[], topK: number) =>
    H.fakeRerank(query, docs, topK),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  fuseCodeCandidates,
  rerankCodeHits,
  retrievalConfig,
  searchCode,
  RERANK_DOC_TRUNCATION,
  SNIPPET_CHARS,
  SNIPPET_LINES,
} from "../src/search.js";
import {
  loadCodeEval,
  filterRowsBySplit,
  mulberry32 as prng,
  ndcgAtK,
  pairedBootstrapCI,
  perQueryDeltas,
  regressionExcluded,
  runCodeEval,
  writeRunArtifact,
  type CodeEvalFile,
  type RunArtifact,
} from "./code-eval.helper.js";
import {
  expand,
  expandBfs,
  expandCte,
  usesBfs,
  BFS_DEPTH_THRESHOLD,
  type GraphRow,
} from "../src/graph/traverse.js";
import { indexRepo, resolveEdges } from "../src/indexer.js";
import type { ByNameEdge } from "../src/graph/extractor.js";
import { distillEval } from "../src/db/distill-eval.js";

const here = dirname(fileURLToPath(import.meta.url));
const serviceRoot = join(here, "..");

// Unique fixture namespace.
const P_CODE = "conf-w13-code";
const P_POOL = "conf-w13-code-pool";
const P_GRAPH = "conf-w13-graph-proj";
const P_IDX = "conf-w13-idx-proj";
const P_EMPTY = "conf-w13-code-empty";
const R_A = "conf-w13-repo-a";
const R_B = "conf-w13-repo-b";
const R_POOL = "conf-w13-repo-pool";
const R_G = "conf-w13-graph";
const R_RES = "conf-w13-res";
const R_IDX = "conf-w13-idx";
const ALL_REPOS = [R_A, R_B, R_POOL, R_G, R_RES, R_IDX];

const cid = (n: number) =>
  `30000000-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;
const sid = (n: number) =>
  `c0ffee00-0000-4000-8000-${n.toString(16).padStart(12, "0")}`;

// ── Pure: AC-103 — math layer vs hand-computed values ────────────────────────

describe("AC-103: ndcgAtK matches hand-computed fixtures (pure)", () => {
  it("relevant at rank 1 only → 1.0", () => {
    expect(ndcgAtK(["r", "x", "y"], ["r"], 10)).toBeCloseTo(1.0, 10);
  });

  it("single relevant at rank 3, k=10 → 1/log2(4) = 0.5", () => {
    expect(ndcgAtK(["a", "b", "r"], ["r"], 10)).toBeCloseTo(0.5, 10);
  });

  it("two relevant at ranks 2+4, k=10, |rel|=2 → (1/log2(3)+1/log2(5))/(1+1/log2(3))", () => {
    const expected =
      (1 / Math.log2(3) + 1 / Math.log2(5)) / (1 + 1 / Math.log2(3));
    expect(ndcgAtK(["a", "r1", "b", "r2"], ["r1", "r2"], 10)).toBeCloseTo(
      expected,
      10,
    );
  });

  it("relevant absent → 0; k=0 → 0; empty relevant → 0", () => {
    expect(ndcgAtK(["a", "b"], ["r"], 10)).toBe(0);
    expect(ndcgAtK(["r"], ["r"], 0)).toBe(0);
    expect(ndcgAtK(["a"], [], 10)).toBe(0);
  });

  it("IDCG truncates at k < |relevant|: 2 relevants at ranks 1+2, k=2, |rel|=3 → 1.0", () => {
    expect(ndcgAtK(["r1", "r2"], ["r1", "r2", "r3"], 2)).toBeCloseTo(1.0, 10);
  });
});

describe("AC-103: pairedBootstrapCI exact on constants + seed-deterministic (pure)", () => {
  it("constant deltas [0.1 ×20] → mean = ciLow = ciHigh = 0.1", () => {
    const ci = pairedBootstrapCI(Array(20).fill(0.1), { seed: 7 });
    expect(ci.mean).toBeCloseTo(0.1, 10);
    expect(ci.ciLow).toBeCloseTo(0.1, 10);
    expect(ci.ciHigh).toBeCloseTo(0.1, 10);
  });

  it("same seed twice → identical output", () => {
    const deltas = Array.from({ length: 30 }, (_, i) =>
      i % 2 === 0 ? 0.08 : -0.03,
    );
    const a = pairedBootstrapCI(deltas, { seed: 512, iterations: 2000 });
    const b = pairedBootstrapCI(deltas, { seed: 512, iterations: 2000 });
    expect(a).toEqual(b);
  });

  it("empty input → zeros", () => {
    expect(pairedBootstrapCI([], { seed: 1 })).toEqual({
      mean: 0,
      ciLow: 0,
      ciHigh: 0,
    });
  });

  it("mulberry32: same seed → same first 5 values", () => {
    const a = prng(99);
    const b = prng(99);
    expect([a(), a(), a(), a(), a()]).toEqual([b(), b(), b(), b(), b()]);
  });
});

// ── Pure: AC-105 — gate math semantics ───────────────────────────────────────

function syntheticArtifact(ndcgs: number[]): RunArtifact {
  return {
    retrievalConfig: {},
    evalVersion: 2,
    split: "dev",
    rows: ndcgs.length,
    perQuery: ndcgs.map((ndcg, i) => ({
      id: `c-${String(i + 1).padStart(3, "0")}`,
      query: `q${i + 1}`,
      poolRecall: 1,
      ndcg,
      mrr: ndcg,
      rank: 1,
    })),
    aggregates: { recallAt25: 1, ndcgAt10: 0, mrr10: 0 },
  };
}

describe("AC-105: paired-bootstrap gate semantics (pure)", () => {
  const base = Array.from({ length: 20 }, (_, i) => 0.5 + (i % 5) * 0.05);

  it("uniform regression → CI excludes zero on the regression side", () => {
    const deltas = perQueryDeltas(
      syntheticArtifact(base),
      syntheticArtifact(base.map((v) => v - 0.2)),
    );
    expect(regressionExcluded(pairedBootstrapCI(deltas, { seed: 42 }))).toBe(
      true,
    );
  });

  it("identical runs → gate passes", () => {
    const deltas = perQueryDeltas(
      syntheticArtifact(base),
      syntheticArtifact(base),
    );
    expect(regressionExcluded(pairedBootstrapCI(deltas, { seed: 42 }))).toBe(
      false,
    );
  });

  it("row-id drift fails loudly", () => {
    expect(() =>
      perQueryDeltas(syntheticArtifact(base), syntheticArtifact(base.slice(1))),
    ).toThrow(/drift|id/i);
  });
});

// ── Pure: AC-104 — retrievalConfig() snapshot contract ───────────────────────

describe("AC-104: retrievalConfig() full snapshot (pure)", () => {
  it("contains every plan-named key with the values the pipeline actually runs", () => {
    const cfg = retrievalConfig();
    expect(cfg.service).toBe("codebase-mcp");
    expect(typeof cfg.rrfK).toBe("number");
    expect(cfg.candidatePool).toBe(25);
    expect(typeof cfg.recallLimit).toBe("number");
    expect(typeof cfg.codeEmbedModel).toBe("string");
    expect(typeof cfg.codeContextModel).toBe("string");
    expect(typeof cfg.rerankModel).toBe("string");
    expect(cfg.rerankDocTruncation).toBe(RERANK_DOC_TRUNCATION);
    expect(cfg.rerankDocTruncation).toBe(1500);
    expect(cfg.snippetLines).toBe(SNIPPET_LINES);
    expect(cfg.snippetLines).toBe(4);
    expect(cfg.snippetChars).toBe(SNIPPET_CHARS);
    expect(cfg.snippetChars).toBe(240);
  });

  it("is JSON-safe (survives a JSON round-trip verbatim)", () => {
    const cfg = retrievalConfig();
    expect(JSON.parse(JSON.stringify(cfg))).toEqual(cfg);
  });

  it("test/runs/ artifact directory convention exists", () => {
    expect(existsSync(join(here, "runs"))).toBe(true);
  });
});

// ── Pure: AC-108 — quota-spending scripts run only via explicit npm run ──────

describe("AC-108: paid operator scripts (pure)", () => {
  it("package.json exposes the distill-eval entry", () => {
    const pkg = JSON.parse(
      readFileSync(join(serviceRoot, "package.json"), "utf8"),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts["distill-eval"]).toBeTruthy();
  });

  it("distill-eval carries an import.meta.url main-guard", () => {
    const src = readFileSync(
      join(serviceRoot, "src", "db", "distill-eval.ts"),
      "utf8",
    );
    expect(src).toMatch(/import\.meta\.url/);
  });

  it("importing the script module executes nothing (no network, no exit)", async () => {
    const fetchCalls: unknown[][] = [];
    const origFetch = globalThis.fetch;
    globalThis.fetch = (async (...args: unknown[]) => {
      fetchCalls.push(args);
      throw new Error("conformance: no network on import");
    }) as typeof fetch;
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((
      code?: number,
    ) => {
      throw new Error(`conformance: process.exit(${code}) during import`);
    }) as never);
    try {
      await import("../src/db/distill-eval.js");
      expect(fetchCalls).toHaveLength(0);
      expect(exitSpy).not.toHaveBeenCalled();
    } finally {
      globalThis.fetch = origFetch;
      exitSpy.mockRestore();
    }
  });
});

// ── Pure: AC-109 — gold splits are human-gated ───────────────────────────────

describe("AC-109: gold v2 + human approval gate (pure)", () => {
  const goldPath = join(here, "fixtures", "code-eval.json");

  it("code-eval.json is v2: changelog header, id/split/provenance on every row", () => {
    const file = loadCodeEval(goldPath);
    expect(file.version).toBe(2);
    expect(Array.isArray(file.changelog)).toBe(true);
    expect(file.changelog!.length).toBeGreaterThan(0);
    expect(typeof file.k).toBe("number");
    expect(file.rows.length).toBeGreaterThan(0);
    for (const r of file.rows) {
      expect(r.id).toMatch(/^c-\d+/);
      expect(["dev", "test"]).toContain(r.split);
      expect(r.provenance).toBeTruthy();
      expect(Array.isArray(r.relevantPaths)).toBe(true);
      expect(r.relevantPaths.length).toBeGreaterThan(0);
    }
    // Row ids are unique (they are the gate's baseline↔fresh join keys).
    expect(new Set(file.rows.map((r) => r.id)).size).toBe(file.rows.length);
  });

  it("filterRowsBySplit partitions the file completely", () => {
    const file = loadCodeEval(goldPath);
    const dev = filterRowsBySplit(file, "dev");
    const test = filterRowsBySplit(file, "test");
    expect(dev.length + test.length).toBe(file.rows.length);
  });

  it("distillEval emits approved:false candidates and does not touch the gold file", async () => {
    const goldBefore = readFileSync(goldPath, "utf8");
    const tmp = mkdtempSync(join(tmpdir(), "conf-w13-distill-"));
    try {
      const outPath = join(tmp, "candidates.json");
      const sampled = [
        {
          file_path: "src/auth.ts",
          language: "typescript",
          symbol_name: "requireBearer",
          excerpt: "bearer auth middleware",
        },
        {
          file_path: "scripts/report.py",
          language: "python",
          symbol_name: "generate_report",
          excerpt: "monthly report",
        },
      ];
      const modelReply = JSON.stringify([
        {
          query: "where is bearer auth enforced",
          suggestedGold: ["src/auth.ts"],
          archetype: "where-is-X",
        },
        {
          query: "generate_report",
          suggestedGold: ["scripts/report.py"],
          archetype: "symbol-lookup",
        },
      ]);
      const { candidates, path } = await distillEval({
        pool: { query: async () => ({ rows: sampled }) },
        complete: async () => modelReply,
        projectId: "conf-w13-distill",
        outPath,
        log: () => {},
      });
      expect(candidates.length).toBeGreaterThan(0);
      for (const c of candidates) {
        expect(c.approved).toBe(false);
        expect(c.provenance).toBe("distilled");
      }
      const written = JSON.parse(readFileSync(path, "utf8")) as unknown;
      const rows = (
        Array.isArray(written)
          ? written
          : (Object.values(written as object).find(Array.isArray) as unknown[])
      ) as Array<{ approved: boolean }>;
      expect(rows.length).toBeGreaterThan(0);
      for (const r of rows) expect(r.approved).toBe(false);
      expect(readFileSync(goldPath, "utf8")).toBe(goldBefore);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
});

// ── Pure: AC-401 — traversal engine routing constants ────────────────────────

describe("AC-401: engine routing (pure)", () => {
  it("threshold is 5: depth ≤ 4 stays CTE, depth ≥ 5 goes BFS", () => {
    expect(BFS_DEPTH_THRESHOLD).toBe(5);
    expect(usesBfs(1)).toBe(false);
    expect(usesBfs(4)).toBe(false);
    expect(usesBfs(5)).toBe(true);
    expect(usesBfs(10)).toBe(true);
    // Default depth (undefined → 4) stays on the CTE.
    expect(usesBfs(undefined)).toBe(false);
  });
});

// ── DB-backed conformance (skipped cleanly without DATABASE_URL) ─────────────

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

const CHUNKS: ChunkFixture[] = [
  {
    id: cid(1),
    repo: R_A,
    path: "src/auth/password.ts",
    language: "typescript",
    symbolName: "hashPassword",
    startLine: 1,
    endLine: 12,
    content:
      "export async function hashPassword(plain: string): Promise<string> {\n  const salt = await bcrypt.genSalt(12);\n  return bcrypt.hash(plain, salt);\n}\n// bcrypt password hashing for credential storage",
  },
  {
    id: cid(2),
    repo: R_A,
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
    repo: R_A,
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
    repo: R_A,
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
    repo: R_A,
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
    repo: R_A,
    path: "src/db/pool.ts",
    language: "typescript",
    symbolName: null,
    startLine: 32,
    endLine: 38,
    content:
      "// database connection pool error listener: log and continue\npool.on('error', (err) => logger.error({ err }, 'pg pool error'));",
  },
  {
    id: cid(7),
    repo: R_B,
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
    repo: R_B,
    path: "scripts/report.py",
    language: "python",
    symbolName: "fetch_usage",
    startLine: 22,
    endLine: 34,
    content:
      'def fetch_usage(month: str) -> list[dict]:\n    # query the warehouse for monthly usage report rows\n    return db.execute(USAGE_SQL, {"month": month}).fetchall()',
  },
];
const chunkById = new Map(CHUNKS.map((c) => [c.id, c]));

// Graph fixture symbol ids.
const N = (i: number) => sid(i); // chain n1..n9
const IMP_DECOY_1 = sid(20);
const IMP_DECOY_2 = sid(21);
const CYC_1 = sid(30);
const CYC_2 = sid(31);
const AMB_CALLER = sid(40);
const DUP_A = sid(41);
const DUP_B = sid(42);
const STAR_HUB = sid(50);
const STAR_LEAF = (i: number) => sid(50 + i); // 1..7

describe.skipIf(skip)("W1–W3 conformance — DB-backed (codebase-mcp)", () => {
  let db: pg.Pool;

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
      [repo, projectId, path, language, `conf-w13-sha-${repo}-${path}`],
    );
    return rows[0].id;
  }

  async function insertChunk(
    fileId: string,
    projectId: string,
    c: ChunkFixture,
  ): Promise<void> {
    await db.query(
      `INSERT INTO codebase.code_chunks
         (id, file_id, repository_id, project_id, file_path, language,
          symbol_name, start_line, end_line, content, content_sha256, embedding)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::halfvec)`,
      [
        c.id,
        fileId,
        c.repo,
        projectId,
        c.path,
        c.language,
        c.symbolName,
        c.startLine,
        c.endLine,
        c.content,
        `conf-w13-chunk-sha-${c.id}`,
        `[${H.fakeVec(`chunk:${c.id}`).join(",")}]`,
      ],
    );
  }

  async function insertSymbol(
    id: string,
    name: string,
    filePath: string,
  ): Promise<void> {
    await db.query(
      `INSERT INTO codebase.symbols (id, repository_id, project_id, name, kind, file_path, start_line)
       VALUES ($1, $2, $3, $4, 'function', $5, 1)`,
      [id, R_G, P_GRAPH, name, filePath],
    );
  }

  async function insertEdge(
    repo: string,
    from: string,
    to: string,
    kind: "call" | "import",
  ): Promise<void> {
    await db.query(
      `INSERT INTO codebase.symbol_edges (repository_id, from_symbol, to_symbol, kind)
       VALUES ($1, $2, $3, $4)`,
      [repo, from, to, kind],
    );
  }

  async function cleanup(): Promise<void> {
    await db.query(
      `DELETE FROM codebase.index_runs WHERE repository_id = ANY($1)`,
      [ALL_REPOS],
    );
    // files delete cascades chunks + file_id-keyed symbols; the by-hand graph
    // symbols carry no file_id, so sweep them by repository_id (edges cascade).
    await db.query(`DELETE FROM codebase.files WHERE repository_id = ANY($1)`, [
      ALL_REPOS,
    ]);
    await db.query(
      `DELETE FROM codebase.symbols WHERE repository_id = ANY($1)`,
      [ALL_REPOS],
    );
  }

  beforeAll(async () => {
    db = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    const sqlDir = join(serviceRoot, "sql");
    await db.query("CREATE SCHEMA IF NOT EXISTS codebase;");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await db.query(sql);
    }
    await cleanup();

    // Search fixtures.
    const fileIds = new Map<string, string>();
    for (const c of CHUNKS) {
      const key = `${c.repo}|${c.path}`;
      if (!fileIds.has(key)) {
        fileIds.set(key, await insertFile(c.repo, P_CODE, c.path, c.language));
      }
      await insertChunk(fileIds.get(key)!, P_CODE, c);
    }

    // Pool-cap fixtures: 30 chunks in one file, all sharing a query token.
    const poolFileId = await insertFile(
      R_POOL,
      P_POOL,
      "pool/big.ts",
      "typescript",
    );
    for (let i = 1; i <= 30; i++) {
      await insertChunk(poolFileId, P_POOL, {
        id: cid(100 + i),
        repo: R_POOL,
        path: "pool/big.ts",
        language: "typescript",
        symbolName: `poolFn${i}`,
        startLine: i * 10,
        endLine: i * 10 + 8,
        content: `export function poolFn${i}() {\n  // confpool retrieval token variant ${i} ${"x".repeat(i)}\n}`,
      });
    }

    // Graph fixtures: call chain n1→…→n9, import decoys + an import SHORTCUT
    // n1→n5 that would surface n5 at depth 1 if import edges ever leaked into
    // call expansion (AC-402 leak detector), a 2-cycle, an ambiguous pair, and
    // a star hub for the per-level cap.
    for (let i = 1; i <= 9; i++) {
      await insertSymbol(N(i), `chainFn${i}`, `graph/chain${i}.ts`);
    }
    for (let i = 1; i <= 8; i++) {
      await insertEdge(R_G, N(i), N(i + 1), "call");
    }
    await insertSymbol(IMP_DECOY_1, "importedDecoyOne", "graph/impA.ts");
    await insertSymbol(IMP_DECOY_2, "importedDecoyTwo", "graph/impB.ts");
    await insertEdge(R_G, N(1), IMP_DECOY_1, "import");
    await insertEdge(R_G, N(2), IMP_DECOY_2, "import");
    await insertEdge(R_G, N(1), N(5), "import"); // the shortcut

    await insertSymbol(CYC_1, "cycleFnOne", "graph/cyc1.ts");
    await insertSymbol(CYC_2, "cycleFnTwo", "graph/cyc2.ts");
    await insertEdge(R_G, CYC_1, CYC_2, "call");
    await insertEdge(R_G, CYC_2, CYC_1, "call");

    await insertSymbol(AMB_CALLER, "ambCaller", "graph/amb0.ts");
    await insertSymbol(DUP_A, "dupTarget", "graph/dupA.ts");
    await insertSymbol(DUP_B, "dupTarget", "graph/dupB.ts");
    await insertEdge(R_G, AMB_CALLER, DUP_A, "call");
    await insertEdge(R_G, AMB_CALLER, DUP_B, "call");

    await insertSymbol(STAR_HUB, "starHub", "graph/star.ts");
    for (let i = 1; i <= 7; i++) {
      await insertSymbol(STAR_LEAF(i), `starLeaf${i}`, `graph/star${i}.ts`);
      await insertEdge(R_G, STAR_HUB, STAR_LEAF(i), "call");
    }
  });

  afterAll(async () => {
    await cleanup();
    await db.end();
  });

  // ── AC-101: composed searchCode output contract ────────────────────────────

  function assertHitShape(
    hits: Awaited<ReturnType<typeof searchCode>>["hits"],
    k: number,
  ): void {
    expect(hits.length).toBeLessThanOrEqual(k);
    for (const h of hits) {
      const f = chunkById.get(h.chunkId)!;
      expect(f).toBeTruthy();
      // 4-line then 240-char snippet, exactly as the plan pins it.
      expect(h.snippet).toBe(
        f.content.split("\n").slice(0, 4).join("\n").slice(0, 240),
      );
      expect(h.snippet.split("\n").length).toBeLessThanOrEqual(4);
      expect(h.snippet.length).toBeLessThanOrEqual(240);
      expect(h.filePath).toBe(f.path);
      expect(h.startLine).toBe(f.startLine);
      expect(h.endLine).toBe(f.endLine);
      expect(h.symbolName).toBe(f.symbolName);
      expect(h.language).toBe(f.language);
      // AC-203 (wave 4): raw scores never reach the public hit shape.
      expect("score" in h).toBe(false);
    }
  }

  it("AC-101/AC-203: hit shape — 4-line/240-char snippet, rank-only, ≤ k", async () => {
    const result = await searchCode({
      projectId: P_CODE,
      query: "bcrypt password hashing credentials",
      k: 5,
    });
    const { hits } = result;
    expect(hits.length).toBeGreaterThan(0);
    assertHitShape(hits, 5);
  });

  it("AC-101: repo filter narrows to that repository", async () => {
    const result = await searchCode({
      projectId: P_CODE,
      query: "monthly usage report",
      repo: R_B,
      k: 10,
    });
    const { hits } = result;
    expect(hits.length).toBeGreaterThan(0);
    for (const h of hits) {
      expect(chunkById.get(h.chunkId)!.repo).toBe(R_B);
    }
  });

  it("AC-101: language filter narrows to that language", async () => {
    const result = await searchCode({
      projectId: P_CODE,
      query: "database connection pool",
      language: "typescript",
      k: 10,
    });
    const { hits } = result;
    expect(hits.length).toBeGreaterThan(0);
    for (const h of hits) expect(h.language).toBe("typescript");
  });

  it("AC-101: empty corpus → []", async () => {
    const result = await searchCode({
      projectId: P_EMPTY,
      query: "anything at all",
      k: 5,
    });
    expect(result.hits).toEqual([]);
  });

  it("AC-101: identical inputs → identical output (deterministic pipeline)", async () => {
    const input = {
      projectId: P_CODE,
      query: "password reset token email",
      k: 5,
    };
    const a = await searchCode(input);
    const b = await searchCode(input);
    expect(b.hits).toEqual(a.hits);
  });

  it("AC-101: rerank-empty falls back to RRF pool order", async () => {
    const query = `${H.RERANK_EMPTY} database connection pool`;
    const pool = await fuseCodeCandidates({ projectId: P_CODE, query });
    expect(pool.length).toBeGreaterThan(0);
    const result = await searchCode({ projectId: P_CODE, query, k: 5 });
    expect(result.hits.map((h) => h.chunkId)).toEqual(
      pool.slice(0, 5).map((c) => c.id),
    );
  });

  it("AC-101/AC-102: two-phase composition ≡ one-shot searchCode", async () => {
    const input = {
      projectId: P_CODE,
      query: "warehouse usage rows month",
      k: 5,
    };
    const oneShot = await searchCode(input);
    const pool = await fuseCodeCandidates(input);
    expect(pool.length).toBeGreaterThan(0);
    const composed = await rerankCodeHits(input.query, pool, input.k);
    expect(composed.map(({ score: _score, ...hit }) => hit)).toEqual(
      oneShot.hits,
    );
  });

  // ── AC-102: fuseCodeCandidates pool contract ───────────────────────────────

  it("AC-102: pool caps at config.candidatePool rows with per-arm RRF ranks", async () => {
    const pool = await fuseCodeCandidates({
      projectId: P_POOL,
      query: "confpool retrieval token",
    });
    expect(pool.length).toBe(retrievalConfig().candidatePool);
    for (const c of pool) {
      expect(typeof c.id).toBe("string");
      expect("bm25_rank" in c).toBe(true);
      expect("vec_rank" in c).toBe(true);
      expect(c.bm25_rank !== null || c.vec_rank !== null).toBe(true);
      expect(typeof c.rrf).toBe("number");
    }
    for (let i = 1; i < pool.length; i++) {
      expect(pool[i - 1].rrf).toBeGreaterThanOrEqual(pool[i].rrf);
    }
  });

  it("AC-102: the live handler consumes exactly this pool (finals ⊆ pool ids)", async () => {
    const query = "confpool retrieval token";
    const pool = await fuseCodeCandidates({ projectId: P_POOL, query });
    const poolIds = new Set(pool.map((c) => c.id));
    const result = await searchCode({ projectId: P_POOL, query, k: 10 });
    const { hits } = result;
    expect(hits.length).toBe(10);
    for (const h of hits) expect(poolIds.has(h.chunkId)).toBe(true);
  });

  // ── AC-104: two-layer eval artifact ────────────────────────────────────────

  it("AC-104: runCodeEval artifact carries the config snapshot + two-layer per-query scores", async () => {
    const evalFile: CodeEvalFile = {
      version: 2,
      changelog: ["conformance-w13 synthetic gold (never committed)"],
      k: 10,
      rows: [
        {
          id: "c-g1",
          query: "bcrypt password hashing credentials",
          relevantPaths: ["src/auth/password.ts"],
          split: "dev",
          provenance: "seed-v1",
        },
        {
          id: "c-g2",
          query: "monthly usage report csv",
          relevantPaths: ["scripts/report.py"],
          split: "dev",
          provenance: "seed-v1",
        },
      ],
    };
    const artifact = await runCodeEval(evalFile, {
      projectId: P_CODE,
      split: "dev",
    });

    expect(artifact.retrievalConfig).toEqual(retrievalConfig());
    expect(artifact.evalVersion).toBe(2);
    expect(artifact.split).toBe("dev");
    expect(artifact.rows).toBe(2);
    expect(artifact.perQuery.map((q) => q.id)).toEqual(["c-g1", "c-g2"]);
    for (const q of artifact.perQuery) {
      expect(typeof q.poolRecall).toBe("number"); // pool layer (Recall@25)
      expect(typeof q.ndcg).toBe("number"); // final layer (nDCG@10)
      expect(typeof q.mrr).toBe("number");
      // Tiny corpus, pool of 25 → gold must be recalled and ranked somewhere.
      expect(q.poolRecall).toBe(1);
      expect(q.ndcg).toBeGreaterThan(0);
    }
    expect(artifact.aggregates).toEqual({
      recallAt25: expect.any(Number),
      ndcgAt10: expect.any(Number),
      mrr10: expect.any(Number),
    });

    const tmp = mkdtempSync(join(tmpdir(), "conf-w13-runs-"));
    try {
      const path = writeRunArtifact(tmp, "conf-w13", artifact);
      expect(existsSync(path)).toBe(true);
      const parsed = JSON.parse(readFileSync(path, "utf8")) as RunArtifact;
      expect(parsed.retrievalConfig).toEqual(retrievalConfig());
      expect(parsed.perQuery).toHaveLength(2);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });

  // ── AC-401 / AC-402: deep traversal + import exclusion + parity ────────────

  const canon = (rows: GraphRow[]) =>
    rows.map((r) => `${r.id}|${r.depth}|${r.edge_type}|${r.ambiguous}`).sort();

  it("AC-401: depth-8 expand walks the whole chain at min depths (BFS route)", async () => {
    const rows = await expand([N(1)], "callees", 8, 1000);
    const byId2 = new Map(rows.map((r) => [r.id, r]));
    expect(byId2.get(N(1))!.depth).toBe(0);
    expect(byId2.get(N(1))!.edge_type).toBe("def");
    for (let i = 2; i <= 9; i++) {
      expect(byId2.get(N(i)), `chainFn${i} must be reached`).toBeTruthy();
      expect(byId2.get(N(i))!.depth).toBe(i - 1);
    }
    // Each node exactly once.
    expect(rows.length).toBe(new Set(rows.map((r) => r.id)).size);
  });

  it("AC-401: depth-3 expand (CTE route) stops at depth 3", async () => {
    const rows = await expand([N(1)], "callees", 3, 1000);
    const ids = new Set(rows.map((r) => r.id));
    expect(ids).toEqual(new Set([N(1), N(2), N(3), N(4)]));
    expect(Math.max(...rows.map((r) => r.depth))).toBe(3);
  });

  it("AC-401: cycles terminate at depth 8 — each node once, min depth", async () => {
    const rows = await expand([CYC_1], "callees", 8, 1000);
    expect(canon(rows)).toEqual(
      canon([
        {
          id: CYC_1,
          name: "cycleFnOne",
          file: "graph/cyc1.ts",
          line: 1,
          depth: 0,
          edge_type: "def",
          ambiguous: false,
        },
        {
          id: CYC_2,
          name: "cycleFnTwo",
          file: "graph/cyc2.ts",
          line: 1,
          depth: 1,
          edge_type: "call→",
          ambiguous: false,
        },
      ]),
    );
  });

  it("AC-401: per-level cap truncates the frontier without error", async () => {
    const rows = await expandBfs([STAR_HUB], "callees", 2, 1000, 3);
    const depth1 = rows.filter((r) => r.depth === 1);
    expect(depth1.length).toBeGreaterThanOrEqual(1);
    expect(depth1.length).toBeLessThanOrEqual(3);
  });

  it("AC-402: import edges never leak into call expansion — either engine", async () => {
    const deep = await expand([N(1)], "callees", 8, 1000); // BFS
    const shallow = await expand([N(1)], "callees", 4, 1000); // CTE
    for (const rows of [deep, shallow]) {
      const ids = new Set(rows.map((r) => r.id));
      expect(ids.has(IMP_DECOY_1)).toBe(false);
      expect(ids.has(IMP_DECOY_2)).toBe(false);
    }
    // The n1→n5 IMPORT shortcut must not shortcut the call chain: n5 stays at
    // call-depth 4 in both engines.
    expect(deep.find((r) => r.id === N(5))!.depth).toBe(4);
    expect(shallow.find((r) => r.id === N(5))!.depth).toBe(4);
  });

  it("AC-402: BFS ≡ CTE parity at overlapping depths on the import-bearing fixture", async () => {
    const seeds: Array<[string, string[]]> = [
      ["chain head", [N(1)]],
      ["chain tail (callers)", [N(9)]],
      ["ambiguous caller", [AMB_CALLER]],
    ];
    for (const [label, seed] of seeds) {
      for (const depth of [2, 3, 4]) {
        for (const dir of ["callees", "callers", "both"] as const) {
          const cte = await expandCte(seed, dir, depth, 1000);
          const bfs = await expandBfs(seed, dir, depth, 1000);
          expect(
            canon(bfs),
            `parity ${label} dir=${dir} depth=${depth}`,
          ).toEqual(canon(cte));
        }
      }
    }
  });

  it("AC-402: same-name >1-target hops are ambiguous:true in BOTH engines", async () => {
    const cte = await expandCte([AMB_CALLER], "callees", 2, 1000);
    const bfs = await expandBfs([AMB_CALLER], "callees", 2, 1000);
    for (const [engine, rows] of [
      ["cte", cte],
      ["bfs", bfs],
    ] as const) {
      const dupA = rows.find((r) => r.id === DUP_A);
      const dupB = rows.find((r) => r.id === DUP_B);
      expect(dupA, `${engine}: dupA reached`).toBeTruthy();
      expect(dupB, `${engine}: dupB reached`).toBeTruthy();
      expect(dupA!.ambiguous, `${engine}: dupA ambiguous`).toBe(true);
      expect(dupB!.ambiguous, `${engine}: dupB ambiguous`).toBe(true);
    }
  });

  // ── AC-403: resolveEdges lock window ───────────────────────────────────────

  it("AC-403: heavy JOIN resolution runs OUTSIDE the txn; the txn is DELETE + INSERT-from-staged only", async () => {
    // Symbols the by-name edges resolve against.
    await db.query(
      `INSERT INTO codebase.symbols (id, repository_id, project_id, name, kind, file_path, start_line)
       VALUES
         ($1, $4, $5, 'srcFn', 'function', 'res/a.ts', 1),
         ($2, $4, $5, 'dstFn', 'function', 'res/b.ts', 1),
         ($3, $4, $5, 'impFn', 'function', 'res/d.ts', 1)`,
      [sid(60), sid(61), sid(62), R_RES, P_GRAPH],
    );
    const edges: ByNameEdge[] = [
      {
        fromName: "srcFn",
        fromFile: "res/a.ts",
        toName: "dstFn",
        kind: "call",
        siteLine: 3,
      },
      {
        fromName: null,
        fromFile: "res/a.ts",
        toName: "impFn",
        kind: "import",
        module: "./d",
        siteLine: 1,
      },
    ];

    const trace: string[] = [];
    const real = await db.connect();
    const instrumented = new Proxy(real, {
      get(target, prop) {
        if (prop === "query") {
          return (...args: unknown[]) => {
            const q = args[0];
            trace.push(
              typeof q === "string"
                ? q
                : String((q as { text?: string })?.text ?? q),
            );
            return (target.query as unknown as (...a: unknown[]) => unknown)(
              ...args,
            );
          };
        }
        const v = Reflect.get(target, prop, target);
        return typeof v === "function"
          ? (v as (...a: unknown[]) => unknown).bind(target)
          : v;
      },
    }) as pg.PoolClient;

    try {
      const inserted = await resolveEdges(R_RES, edges, instrumented);
      expect(inserted).toBe(2);
      // Trace of run 1 only — the second (replace-semantics) run below appends
      // its own statements to `trace`, which must not pollute the window scan.
      const run1 = [...trace];

      // Behavior: exactly the two resolved edges landed.
      const { rows } = await db.query<{
        from_name: string;
        to_name: string;
        kind: string;
      }>(
        `SELECT sf.name AS from_name, st.name AS to_name, e.kind
         FROM codebase.symbol_edges e
         JOIN codebase.symbols sf ON sf.id = e.from_symbol
         JOIN codebase.symbols st ON st.id = e.to_symbol
         WHERE e.repository_id = $1
         ORDER BY e.kind, sf.name`,
        [R_RES],
      );
      expect(rows).toEqual([
        { from_name: "srcFn", to_name: "dstFn", kind: "call" },
        { from_name: "srcFn", to_name: "impFn", kind: "import" },
      ]);

      // Replace semantics: a second run does not duplicate.
      await resolveEdges(R_RES, edges, instrumented);
      const { rows: again } = await db.query<{ n: string }>(
        `SELECT count(*) AS n FROM codebase.symbol_edges WHERE repository_id = $1`,
        [R_RES],
      );
      expect(Number(again[0].n)).toBe(2);

      // Lock-window trace (first run): exactly one BEGIN…COMMIT window.
      const beginIdx = run1.findIndex((q) => /^\s*BEGIN\b/i.test(q));
      expect(beginIdx).toBeGreaterThan(-1);
      const commitIdx = run1.findIndex(
        (q, i) => i > beginIdx && /^\s*COMMIT\b/i.test(q),
      );
      expect(commitIdx).toBeGreaterThan(beginIdx);

      const inTxn = run1.slice(beginIdx + 1, commitIdx);
      expect(inTxn.length).toBeGreaterThan(0);
      for (const q of inTxn) {
        // Only the swap statements live inside the lock window…
        expect(
          /^\s*(DELETE\s+FROM\s+codebase\.symbol_edges|INSERT\s+INTO\s+codebase\.symbol_edges)/i.test(
            q,
          ),
          `unexpected statement inside the lock window: ${q.slice(0, 120)}`,
        ).toBe(true);
        // …and none of them re-runs name resolution.
        expect(q).not.toMatch(/JOIN\s+codebase\.symbols/i);
      }

      // The heavy name-resolution JOINs all executed BEFORE the txn opened.
      const joinIdxs = run1
        .map((q, i) => [q, i] as const)
        .filter(([q]) => /JOIN\s+codebase\.symbols/i.test(q))
        .map(([, i]) => i);
      expect(joinIdxs.length).toBeGreaterThan(0);
      for (const i of joinIdxs) expect(i).toBeLessThan(beginIdx);
    } finally {
      real.release();
    }
  });

  // ── AC-404: chunk-level content-hash embedding reuse ───────────────────────

  it("AC-404: re-indexing unchanged content makes ZERO embed calls; only changed chunks re-embed", async () => {
    const repoDir = mkdtempSync(join(tmpdir(), "conf-w13-repo-"));
    try {
      writeFileSync(
        join(repoDir, "alpha.ts"),
        `export function alphaOne(): string {\n  return "conf w13 alpha one body marker";\n}\n\nexport function alphaTwo(): string {\n  return alphaOne() + " twice";\n}\n`,
      );
      writeFileSync(
        join(repoDir, "beta.ts"),
        `export function betaOne(): string {\n  return "conf w13 beta one body marker";\n}\n`,
      );

      // Run 1 (force): everything embeds.
      H.embedLog.length = 0;
      await indexRepo(repoDir, R_IDX, P_IDX, true);
      const firstRun = [...H.embedLog];
      expect(firstRun.length).toBeGreaterThan(0);

      // Run 2 (force, unchanged): the chunk-level content_sha256 cache reuses
      // every stored vector — zero Voyage calls.
      H.embedLog.length = 0;
      await indexRepo(repoDir, R_IDX, P_IDX, true);
      expect(H.embedLog).toEqual([]);

      // Run 3 (force, one file changed): only the changed content embeds.
      appendFileSync(
        join(repoDir, "beta.ts"),
        `\nexport function betaTwo(): string {\n  return betaOne() + " CONF_W13_CHANGED";\n}\n`,
      );
      H.embedLog.length = 0;
      await indexRepo(repoDir, R_IDX, P_IDX, true);
      expect(H.embedLog.length).toBeGreaterThan(0);
      expect(H.embedLog.length).toBeLessThan(firstRun.length + 1);
      expect(H.embedLog.some((t) => t.includes("CONF_W13_CHANGED"))).toBe(true);
      for (const t of H.embedLog) {
        expect(
          t.includes("alpha one body marker"),
          "unchanged alpha content must be served from the vector cache",
        ).toBe(false);
      }
    } finally {
      rmSync(repoDir, { recursive: true, force: true });
    }
  });
});
