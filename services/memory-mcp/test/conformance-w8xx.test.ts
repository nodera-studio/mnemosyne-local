/**
 * Conformance suite for the retrieval/token-efficiency plan (AC-801…AC-816).
 *
 * AUTHOR-BLIND GATE: every assertion below is derived from the approved plan
 * (originally .claude/plans/2026-07-04-mnemosyne-retrieval-token-efficiency/ in the
 * consuming project's repo) — NOT from the implementation or the implementer's own
 * test files. It exercises
 * only PUBLIC surfaces: exported functions of src/memory.ts / src/summarize.ts,
 * the MCP tool layer via buildServer, and the operator scripts' exported entry
 * functions with injected deps.
 *
 * Approved amendment honored here: snippet highlight detection uses private-use
 * sentinel delimiters internally, rendered as `**` in output. AC-801's `**` contract
 * is about the RENDERED snippet; a literal `**` already present in stored content
 * must NOT be treated as a match marker (AC-802 prefix fallback applies).
 *
 * Run:
 *   npx vitest run conformance-w8xx                          # no-DB mode
 *   DATABASE_URL=postgres://postgres:postgres@localhost:5544/mnemosyne \
 *     npx vitest run conformance-w8xx                        # DB mode (:5544)
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  afterEach,
  vi,
} from "vitest";
import { readFileSync, readdirSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { Pool, PoolClient } from "pg";

// Deterministic gate state for the WHOLE file: the write path must stay LLM-free
// (AC-808 default-off double gate). Set BEFORE any src module is imported — all src
// imports below are dynamic, so config.ts reads these values.
process.env.ANTHROPIC_API_KEY = "";
process.env.SUMMARIZE_ON_STORE = "";

const HERE = dirname(fileURLToPath(import.meta.url));
const SERVICE_ROOT = join(HERE, "..");
const SRC_DIR = join(SERVICE_ROOT, "src");
const SQL_DIR = join(SERVICE_ROOT, "sql");
const DATABASE_URL = process.env.DATABASE_URL ?? "";

const execFileP = promisify(execFile);

// ── Voyage module mock (PAID-API guard; deterministic vectors) ────────────────────────
const voyage = vi.hoisted(() => {
  const DIM = 1024;
  const basis = (i: number): number[] => {
    const v = new Array<number>(DIM).fill(0);
    v[i] = 1;
    return v;
  };
  // Substring → basis rules. Query-side and content-side needles are DISJOINT words
  // mapping to the SAME basis, so a "vector-only" hit shares no lexeme with its query.
  const RULES: Array<[string, number]> = [
    ["Renderer notes", 1], // F2 content (vector-only fixture)
    ["quarkline", 1], // Q2 query → same vector as F2, zero lexical overlap
    ["migration ledger", 2], // F1 content (lexical fixture)
    ["the of and", 2], // stop-words-only query → ranks F1 by vector
  ];
  const hash = (s: string): number => {
    let h = 7;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 1000;
    return 20 + h;
  };
  const vecFor = (text: string): number[] => {
    for (const [needle, idx] of RULES)
      if (text.includes(needle)) return basis(idx);
    return basis(hash(text));
  };
  const embed = vi.fn(async (texts: string[], _t: string) => texts.map(vecFor));
  const embedContextualSingle = vi.fn(async (texts: string[], _t: string) =>
    texts.map(vecFor),
  );
  const embedContextual = vi.fn(async (docs: string[][], _t: string) =>
    docs.map((d) => d.map(vecFor)),
  );
  const rerank = vi.fn(async (_q: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_d, index) => ({ index, score: 1 / (index + 2) })),
  );
  const toVectorLiteral = (v: number[]): string => `[${v.join(",")}]`;
  return {
    DIM,
    basis,
    embed,
    embedContextual,
    embedContextualSingle,
    rerank,
    toVectorLiteral,
  };
});

vi.mock("../src/voyage.js", () => ({
  embed: voyage.embed,
  embedContextual: voyage.embedContextual,
  embedContextualSingle: voyage.embedContextualSingle,
  rerank: voyage.rerank,
  toVectorLiteral: voyage.toVectorLiteral,
}));

const collapse = (s: string): string => s.replace(/\s+/g, " ");
const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// ═══════════════════════════════════════════════════════════════════════════════════
// PURE — runs with and without DATABASE_URL
// ═══════════════════════════════════════════════════════════════════════════════════
describe("conformance-w8xx — pure contracts", () => {
  let mem: typeof import("../src/memory.js");

  beforeAll(async () => {
    mem = await import("../src/memory.js");
  });

  describe("AC-805/806 — budgetMemoryBody (pure half)", () => {
    it("default budget constant is 6000 chars", () => {
      expect(mem.MEMORY_GET_DEFAULT_MAX_CHARS).toBe(6000);
    });

    it("content exactly at budget passes through with no truncation fields", () => {
      const row = { id: "x", content: "a".repeat(6000), metadata: { k: 1 } };
      const out = mem.budgetMemoryBody(row, 6000);
      expect(out).toEqual(row);
      expect("truncated" in out).toBe(false);
      expect("totalChars" in out).toBe(false);
      expect("note" in out).toBe(false);
    });

    it("oversized content is cut to maxChars with truncated/totalChars and a re-fetch note", () => {
      const content = "b".repeat(6001);
      const row = { id: "x", content, metadata: { keep: "me" } };
      const out = mem.budgetMemoryBody(row, 6000);
      expect((out.content as string).length).toBe(6000);
      expect(out.content).toBe(content.slice(0, 6000));
      expect(out.truncated).toBe(true);
      expect(out.totalChars).toBe(6001);
      expect(String(out.note)).toMatch(/full=true/);
      expect(String(out.note)).toMatch(/maxChars=0/);
      // metadata must stay intact — only the content FIELD VALUE is budgeted
      expect(out.metadata).toEqual({ keep: "me" });
    });

    it("maxChars=0 means unlimited (escape hatch)", () => {
      const row = { id: "x", content: "c".repeat(50_000) };
      expect(mem.budgetMemoryBody(row, 0)).toEqual(row);
    });

    it("non-string content passes through unchanged", () => {
      const row = { id: "x", content: 42 as unknown };
      expect(mem.budgetMemoryBody(row as Record<string, unknown>, 10)).toEqual(
        row,
      );
    });
  });

  describe("AC-810 — buildRerankDoc composition", () => {
    const title = "Doc title";
    const longContent = "z".repeat(3000);

    it("truncation budget is 1200 chars", () => {
      expect(mem.RERANK_DOC_TRUNCATION).toBe(1200);
    });

    it("NULL summary → byte-identical to title\\ncontent prefix (today's doc)", () => {
      const expected = `${title}\n${longContent}`.slice(0, 1200);
      expect(
        mem.buildRerankDoc({ title, content: longContent, summary: null }),
      ).toBe(expected);
      // omitted summary behaves the same as null
      expect(mem.buildRerankDoc({ title, content: longContent })).toBe(
        expected,
      );
    });

    it("summary present → title\\nsummary\\ncontent, WHOLE-truncated to 1200 (prefix shape, never summary-only)", () => {
      const summary = "A dense two-sentence summary of the doc.";
      const doc = mem.buildRerankDoc({ title, content: longContent, summary });
      expect(doc).toBe(`${title}\n${summary}\n${longContent}`.slice(0, 1200));
      expect(doc.length).toBeLessThanOrEqual(1200);
      expect(doc.startsWith(`${title}\n${summary}\n`)).toBe(true);
      // the summary DISPLACES content tail within the same budget
      const nullDoc = mem.buildRerankDoc({
        title,
        content: longContent,
        summary: null,
      });
      expect(doc.length).toBe(nullDoc.length);
      expect(doc).not.toBe(nullDoc);
    });
  });

  describe("AC-808 — summarizeMemory gate matrix (zero network, injected judge)", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
      vi.unstubAllGlobals();
      vi.resetModules();
    });

    const freshSummarize = async (key: string, flag: string) => {
      vi.stubEnv("ANTHROPIC_API_KEY", key);
      vi.stubEnv("SUMMARIZE_ON_STORE", flag);
      vi.resetModules();
      return await import("../src/summarize.js");
    };

    it("key unset (flag set) → null, judge never called, zero fetch", async () => {
      const fetchSpy = vi.fn(async () => {
        throw new Error("network disabled by conformance");
      });
      vi.stubGlobal("fetch", fetchSpy);
      const sum = await freshSummarize("", "1");
      const judge = vi.fn(async () => "never");
      await expect(
        sum.summarizeMemory("t", "c", { judge }),
      ).resolves.toBeNull();
      expect(judge).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("flag unset (key set) → null, judge never called, zero fetch", async () => {
      const fetchSpy = vi.fn(async () => {
        throw new Error("network disabled by conformance");
      });
      vi.stubGlobal("fetch", fetchSpy);
      const sum = await freshSummarize("sk-ant-conformance", "");
      const judge = vi.fn(async () => "never");
      await expect(
        sum.summarizeMemory("t", "c", { judge }),
      ).resolves.toBeNull();
      expect(judge).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("both set + summarizer failure → null (never throws)", async () => {
      const sum = await freshSummarize("sk-ant-conformance", "1");
      const judge = vi.fn(async () => {
        throw new Error("judge exploded");
      });
      await expect(
        sum.summarizeMemory("t", "c", { judge }),
      ).resolves.toBeNull();
      expect(judge).toHaveBeenCalledTimes(1);
    });

    it("both set + judge hangs past timeoutMs → null within the bound", async () => {
      const sum = await freshSummarize("sk-ant-conformance", "1");
      const judge = () => new Promise<string>(() => {}); // never settles
      const t0 = Date.now();
      await expect(
        sum.summarizeMemory("t", "c", { judge, timeoutMs: 50 }),
      ).resolves.toBeNull();
      expect(Date.now() - t0).toBeLessThan(3000);
    });

    it("happy path trims; empty/whitespace answer → null", async () => {
      const sum = await freshSummarize("sk-ant-conformance", "1");
      await expect(
        sum.summarizeMemory("t", "c", { judge: async () => "  A summary.  " }),
      ).resolves.toBe("A summary.");
      await expect(
        sum.summarizeMemory("t", "c", { judge: async () => "   " }),
      ).resolves.toBeNull();
    });
  });

  describe("AC-809 — llm.js stays OUT of the server's static import graph", () => {
    const walk = (dir: string): string[] =>
      readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
        e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)],
      );

    it("no static import of llm.js outside src/db/*", () => {
      const files = walk(SRC_DIR).filter(
        (f) =>
          f.endsWith(".ts") &&
          !f.includes(`${join(SRC_DIR, "db")}/`) &&
          basename(f) !== "llm.ts",
      );
      expect(files.length).toBeGreaterThan(0);
      for (const f of files) {
        const text = readFileSync(f, "utf8");
        expect(text, `${f} must not statically import llm.js`).not.toMatch(
          /^\s*(import|export)\s[^;]*from\s*["'][^"']*llm\.js["']/m,
        );
      }
    });

    it("summarize.ts reaches llm.js only via a lazy dynamic import", () => {
      const text = readFileSync(join(SRC_DIR, "summarize.ts"), "utf8");
      expect(text).toContain('await import("./llm.js")');
    });
  });

  describe("AC-811/AC-813 — paid-script consent gates (pure)", () => {
    it("backfill:summaries refuses without --yes and passes with it", async () => {
      const bf = await import("../src/db/backfill-summaries.js");
      const refusal = bf.guardPaidRun([]);
      expect(refusal).toBeTruthy();
      expect(String(refusal)).toMatch(/--yes/);
      expect(bf.guardPaidRun(["--yes"])).toBeNull();
    });

    it("consolidate refuses without --yes; dry-run is the documented default", async () => {
      const cons = await import("../src/db/consolidate.js");
      const refusal = cons.guardConsolidateRun([]);
      expect(refusal).toBeTruthy();
      expect(String(refusal)).toMatch(/--yes/);
      expect(String(refusal)).toMatch(/dry-run/i);
      expect(String(refusal)).toMatch(/--apply/);
      expect(cons.guardConsolidateRun(["--yes"])).toBeNull();
    });
  });

  describe("AC-815/AC-810 — retrievalConfig serialized knobs", () => {
    it("ships rrfK=60 as the default and records the summary-bearing rerank doc", () => {
      const cfg = mem.retrievalConfig() as Record<string, unknown>;
      expect(cfg.rrfK).toBe(60);
      expect(cfg.rerankDocIncludesSummary).toBe(true);
    });
  });

  // ── AC-814 (cheap static half — the live run is operator-verified on the box) ──────
  // The digest cron script lives in the CONSUMING project's own .claude/scripts/cron/,
  // not in this repo — point MNEMO_DIGEST_SCRIPT at it locally to exercise this block;
  // it self-skips (describe.skipIf) when unset or the path doesn't exist.
  const digestPath = [process.env.MNEMO_DIGEST_SCRIPT ?? ""]
    .filter(Boolean)
    .find((p) => existsSync(p));

  describe.skipIf(!digestPath)("AC-814 — digest cron script (static)", () => {
    it("is syntax-clean, mutation-free SQL, prints cosine bands, prompts the gated consolidate", async () => {
      await execFileP("bash", ["-n", digestPath!]);
      const text = readFileSync(digestPath!, "utf8");
      // read-only: no mutating SQL statements anywhere in the script
      expect(text).not.toMatch(
        /\b(insert\s+into|update\s+[a-z_."]+\s+set|delete\s+from|drop\s+table|truncate\s+table|alter\s+table)\b/i,
      );
      // band boundaries from the plan: 0.90–0.925 / 0.925–0.95 / >=0.95
      expect(text).toMatch(/0\.925/);
      expect(text).toMatch(/0\.95/);
      // prompts the operator command; mutation stays behind the gated consolidate
      expect(text).toMatch(/consolidate/);
      expect(text).toMatch(/--yes/);
    });
  });

  // ── AC-816 (cheap text half — plan amendment lives at /opt) ───────────────────────
  const wave7Path =
    "/opt/mnemosyne/.claude/plans/2026-07-03-retrieval-improvement-program/wave-7-blend-decay-bakeoff.md";

  describe.skipIf(!existsSync(wave7Path))(
    "AC-816 — wave-7 amendment text",
    () => {
      it("contains an RRF_K sweep axis with an explicit re-pool-per-k requirement", () => {
        const text = readFileSync(wave7Path, "utf8");
        expect(text).toMatch(/rrf_?k/i);
        expect(text).toMatch(/re-?pool|re-?run.*per k|per k value/i);
        // the TODO closure stays wave-7 business (this plan does NOT close it)
        expect(text).toMatch(/TODO\(operator\/retune\)/);
      });
    },
  );
});

// ═══════════════════════════════════════════════════════════════════════════════════
// DB — requires the :5544 disposable Postgres (progress.md "Test environment facts")
// ═══════════════════════════════════════════════════════════════════════════════════
describe.skipIf(!DATABASE_URL)("conformance-w8xx — DB behavior (:5544)", () => {
  const P1 = "w8xx-conf-snippets";
  const P2 = "w8xx-conf-gate";
  const PBF = "w8xx-conf-backfill";
  const PC1 = "w8xx-conf-consolidate-dry";
  const PC2 = "w8xx-conf-consolidate-apply";
  const PC3 = "w8xx-conf-consolidate-fail";

  const F1_CONTENT =
    "The w8xx migration ledger records every applied checksum so an operator can audit drift across environments. Retention windows for the migration ledger follow quarterly compliance review cadence and archived checksums stay immutable for audit purposes.";
  const F2_CONTENT =
    "Renderer notes: **bold** emphasis tokens appear verbatim in stored markdown payloads. The twin fixture proves literal asterisk pairs survive as plain text rather than being read as highlight markers by the snippet decorator in any conforming build of this retrieval service.";
  const F3_CONTENT = `F3start ${"y".repeat(6992)}`; // exactly 7000 chars

  const Q1 = "ledger checksum audit"; // lexically matches F1 content
  const Q2 = "quarkline flux capacitor tuning"; // vector-twin of F2, zero lexical overlap
  const Q3 = "the of and"; // stop-words only

  let mem: typeof import("../src/memory.js");
  let pool: Pool;
  let mcp: {
    callTool: (a: {
      name: string;
      arguments: Record<string, unknown>;
    }) => Promise<unknown>;
  };
  let f1 = "";
  let f2 = "";
  let f3 = "";
  let q1Ids: string[] = [];
  const contentById: Record<string, string> = {};

  type Captured = { text: string; params: unknown[] | undefined };
  const record = () => {
    const orig = pool.query.bind(pool);
    const calls: Captured[] = [];
    (pool as unknown as { query: unknown }).query = (
      text: unknown,
      params?: unknown,
    ) => {
      calls.push({
        text: String(text),
        params: params as unknown[] | undefined,
      });
      return (orig as (t: unknown, p?: unknown) => Promise<unknown>)(
        text,
        params,
      );
    };
    return {
      calls,
      restore: () => {
        (pool as unknown as { query: unknown }).query = orig;
      },
    };
  };

  beforeAll(async () => {
    mem = await import("../src/memory.js");
    ({ pool } = await import("../src/db/pool.js"));
    const { buildServer } = await import("../src/server.js");

    // Apply the migration chain idempotently (documented contract: sorted sql/*.sql,
    // one query per file, `-- HOLD` first line opts out).
    for (const f of readdirSync(SQL_DIR)
      .filter((n) => n.endsWith(".sql"))
      .sort()) {
      const text = readFileSync(join(SQL_DIR, f), "utf8");
      if (text.split("\n", 1)[0]?.startsWith("-- HOLD")) continue;
      await pool.query(text);
    }
    await pool.query(
      "DELETE FROM memory.memories WHERE project_id LIKE 'w8xx-%'",
    );

    f1 = (
      await mem.storeMemory({
        projectId: P1,
        type: "semantic",
        title: "w8xx-fixture-lexical",
        content: F1_CONTENT,
      })
    ).id;
    f2 = (
      await mem.storeMemory({
        projectId: P1,
        type: "semantic",
        title: "w8xx-fixture-vector-twin",
        content: F2_CONTENT,
      })
    ).id;
    f3 = (
      await mem.storeMemory({
        projectId: P1,
        type: "semantic",
        title: "w8xx-fixture-big",
        content: F3_CONTENT,
      })
    ).id;
    contentById[f1] = F1_CONTENT;
    contentById[f2] = F2_CONTENT;
    contentById[f3] = F3_CONTENT;

    const { Client } =
      await import("@modelcontextprotocol/sdk/client/index.js");
    const { InMemoryTransport } =
      await import("@modelcontextprotocol/sdk/inMemory.js");
    const server = buildServer(P1);
    const [clientTransport, serverTransport] =
      InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "conformance-w8xx", version: "1.0.0" });
    await Promise.all([
      client.connect(clientTransport),
      server.connect(serverTransport),
    ]);
    mcp = client as unknown as typeof mcp;
  }, 90_000);

  afterAll(async () => {
    try {
      await pool.query(
        "DELETE FROM memory.memories WHERE project_id LIKE 'w8xx-%'",
      );
    } catch {
      /* best-effort cleanup */
    }
    await new Promise((r) => setTimeout(r, 150)); // let fire-and-forget logSearch settle
    await pool.end();
  }, 30_000);

  // ── AC-807 ──────────────────────────────────────────────────────────────────────
  it("AC-807 — migration 007 is idempotent; summary is nullable text", async () => {
    const sql007 = readFileSync(join(SQL_DIR, "007_summary.sql"), "utf8");
    await pool.query(sql007);
    await pool.query(sql007); // second apply must be a no-op, not an error
    const { rows } = await pool.query(
      `SELECT data_type, is_nullable FROM information_schema.columns
        WHERE table_schema = 'memory' AND table_name = 'memories' AND column_name = 'summary'`,
    );
    expect(rows).toHaveLength(1);
    expect(rows[0].data_type).toBe("text");
    expect(rows[0].is_nullable).toBe("YES");
  });

  // ── AC-801 ──────────────────────────────────────────────────────────────────────
  it("AC-801 — lexical match renders **-marked headline via one SQL over ONLY the final hit ids", async () => {
    const rec = record();
    let res: Awaited<ReturnType<typeof mem.searchMemory>>;
    try {
      res = await mem.searchMemory({ projectId: P1, query: Q1, limit: 5 });
    } finally {
      rec.restore();
    }
    q1Ids = res.hits.map((h) => h.id).sort();

    const hit = res.hits.find((h) => h.id === f1);
    expect(
      hit,
      "lexical fixture must be a hit for its own terms",
    ).toBeDefined();
    // rendered contract: ** markers around a query lexeme, ≤2 fragments, collapsed
    expect(hit!.snippet).toMatch(/\*\*(ledger|checksum|audit)/i);
    expect(hit!.snippet.split(" … ").length).toBeLessThanOrEqual(2);
    expect(hit!.snippet).not.toContain("\n");
    expect(hit!.snippet).not.toMatch(/ {2}/);
    // amendment: internal private-use sentinels must NEVER leak into rendered output
    expect(hit!.snippet).not.toContain("\uE000");
    expect(hit!.snippet).not.toContain("\uE001");
    // wave-2 carry: pre-backfill fixtures expose summary === null on every hit
    for (const h of res.hits) expect(h.summary).toBeNull();

    // the headline is a SECOND query scoped to exactly the final ≤limit hit ids
    const headlineCalls = rec.calls.filter((c) => /ts_headline/i.test(c.text));
    expect(headlineCalls).toHaveLength(1);
    const idParam = (headlineCalls[0].params ?? []).find(
      (p): p is string[] =>
        Array.isArray(p) &&
        p.length > 0 &&
        p.every((x) => typeof x === "string" && UUID_RE.test(x)),
    );
    expect(
      idParam,
      "headline query must carry the final hit-id array",
    ).toBeDefined();
    expect([...idParam!].sort()).toEqual(q1Ids);
  });

  // ── AC-802 ──────────────────────────────────────────────────────────────────────
  it("AC-802 — vector-only hit falls back to the 180-char prefix; literal ** in content is NOT a marker", async () => {
    const res = await mem.searchMemory({ projectId: P1, query: Q2, limit: 5 });
    const hit = res.hits.find((h) => h.id === f2);
    expect(hit, "vector twin must rank in on its embedding").toBeDefined();
    const prefix = collapse(F2_CONTENT).slice(0, 180);
    expect(hit!.snippet).toBe(prefix);
    // the stored literal ** survives as plain content inside the prefix
    expect(hit!.snippet).toContain("**bold**");
    for (const h of res.hits) {
      expect(h.snippet).not.toContain("\uE000");
      expect(h.snippet).not.toContain("\uE001");
    }
  });

  it("AC-802 — stop-words-only query → every snippet is the deterministic prefix", async () => {
    const res = await mem.searchMemory({ projectId: P1, query: Q3, limit: 5 });
    expect(res.hits.length).toBeGreaterThan(0);
    for (const h of res.hits) {
      expect(h.snippet).toBe(collapse(contentById[h.id] ?? "").slice(0, 180));
    }
  });

  it("AC-802 — headline SQL failure → search still succeeds, prefixes, pool unchanged", async () => {
    const orig = pool.query.bind(pool);
    (pool as unknown as { query: unknown }).query = (
      text: unknown,
      params?: unknown,
    ) => {
      if (/ts_headline/i.test(String(text))) {
        return Promise.reject(new Error("conformance: headline disabled"));
      }
      return (orig as (t: unknown, p?: unknown) => Promise<unknown>)(
        text,
        params,
      );
    };
    let res: Awaited<ReturnType<typeof mem.searchMemory>>;
    try {
      res = await mem.searchMemory({ projectId: P1, query: Q1, limit: 5 });
    } finally {
      (pool as unknown as { query: unknown }).query = orig;
    }
    // snippet decoration is display-only: hit composition identical to the happy run
    expect(res.hits.map((h) => h.id).sort()).toEqual(q1Ids);
    const hit = res.hits.find((h) => h.id === f1);
    expect(hit!.snippet).toBe(collapse(F1_CONTENT).slice(0, 180));
  });

  // ── AC-803 ──────────────────────────────────────────────────────────────────────
  it("AC-803 — getRecent issues NO headline SQL and keeps prefix snippets", async () => {
    const rec = record();
    let hits: Awaited<ReturnType<typeof mem.getRecent>>;
    try {
      hits = await mem.getRecent({ projectId: P1, limit: 10 });
    } finally {
      rec.restore();
    }
    expect(hits.length).toBeGreaterThan(0);
    expect(rec.calls.some((c) => /ts_headline/i.test(c.text))).toBe(false);
    for (const h of hits) {
      expect(h.snippet).toBe(collapse(contentById[h.id] ?? "").slice(0, 180));
      expect(h.summary).toBeNull();
    }
  });

  // ── AC-805 / AC-806 (MCP tool layer) ───────────────────────────────────────────
  const callGet = async (args: Record<string, unknown>) => {
    const r = (await mcp.callTool({ name: "memory_get", arguments: args })) as {
      isError?: boolean;
      content: Array<{ type: string; text: string }>;
    };
    expect(r.isError ?? false).toBe(false);
    const text = r.content[0].text;
    return JSON.parse(text.slice(text.indexOf("{"))) as Record<string, unknown>;
  };

  it("AC-805 — memory_get defaults to a 6000-char budget with explicit truncation markers", async () => {
    const parsed = await callGet({ id: f3 });
    expect((parsed.content as string).length).toBe(6000);
    expect(parsed.content).toBe(F3_CONTENT.slice(0, 6000));
    expect(parsed.truncated).toBe(true);
    expect(parsed.totalChars).toBe(7000);
    expect(String(parsed.note)).toMatch(/full=true/);
    expect(String(parsed.note)).toMatch(/maxChars=0/);
    // wave-2 carry: the stored summary column rides along (NULL pre-backfill)
    expect("summary" in parsed).toBe(true);
    expect(parsed.summary).toBeNull();
  });

  it("AC-806 — full=true returns the complete content with no truncation fields", async () => {
    const parsed = await callGet({ id: f3, full: true });
    expect(parsed.content).toBe(F3_CONTENT);
    expect("truncated" in parsed).toBe(false);
    expect("totalChars" in parsed).toBe(false);
    expect("note" in parsed).toBe(false);
  });

  it("AC-806 — maxChars=0 is an equivalent escape hatch", async () => {
    const parsed = await callGet({ id: f3, maxChars: 0 });
    expect(parsed.content).toBe(F3_CONTENT);
    expect("truncated" in parsed).toBe(false);
  });

  // ── AC-808 (DB half — the row is ALWAYS written) ─────────────────────────────────
  describe("AC-808 — storeMemory always writes; summary NULL on every failure path", () => {
    afterEach(() => {
      vi.unstubAllEnvs();
      vi.unstubAllGlobals();
      vi.resetModules();
    });

    it("key+flag unset → row written, summary NULL, zero LLM network", async () => {
      vi.stubEnv("ANTHROPIC_API_KEY", "");
      vi.stubEnv("SUMMARIZE_ON_STORE", "");
      const fetchSpy = vi.fn(async () => {
        throw new Error("network disabled by conformance");
      });
      vi.stubGlobal("fetch", fetchSpy);
      vi.resetModules();
      const memF = await import("../src/memory.js");
      const { pool: poolF } = await import("../src/db/pool.js");
      try {
        const { id } = await memF.storeMemory({
          projectId: P2,
          type: "semantic",
          title: "w8xx-gate-off",
          content:
            "Gate disabled path unique payload one for the conformance write test.",
        });
        const { rows } = await pool.query(
          "SELECT summary FROM memory.memories WHERE id = $1",
          [id],
        );
        expect(rows).toHaveLength(1);
        expect(rows[0].summary).toBeNull();
        expect(fetchSpy).not.toHaveBeenCalled();
      } finally {
        await poolF.end();
      }
    }, 30_000);

    it("both set + summarizer network failure → write still succeeds with summary NULL", async () => {
      vi.stubEnv("ANTHROPIC_API_KEY", "sk-ant-conformance-test");
      vi.stubEnv("SUMMARIZE_ON_STORE", "1");
      const fetchSpy = vi.fn(async () => {
        throw new Error("conformance: anthropic unreachable");
      });
      vi.stubGlobal("fetch", fetchSpy);
      vi.resetModules();
      const memF = await import("../src/memory.js");
      const { pool: poolF } = await import("../src/db/pool.js");
      try {
        const { id } = await memF.storeMemory({
          projectId: P2,
          type: "semantic",
          title: "w8xx-gate-fail",
          content:
            "Gate enabled but failing summarizer unique payload two for conformance.",
        });
        const { rows } = await pool.query(
          "SELECT summary FROM memory.memories WHERE id = $1",
          [id],
        );
        expect(rows).toHaveLength(1);
        expect(rows[0].summary).toBeNull();
      } finally {
        await poolF.end();
      }
    }, 30_000);
  });

  // ── AC-811 ──────────────────────────────────────────────────────────────────────
  it("AC-811 — backfill pages by keyset incl. edge ids, skips per-row failures, resumes", async () => {
    const bf = await import("../src/db/backfill-summaries.js");
    const client: PoolClient = await pool.connect();
    try {
      await client.query("BEGIN");
      const txPool = {
        query: (t: string, p?: unknown[]) => client.query(t, p),
      } as unknown as Pool;

      const c0 = await bf.countPendingSummaries(txPool);

      const ZERO = "00000000-0000-0000-0000-000000000000";
      const MAX = "ffffffff-ffff-ffff-ffff-ffffffffffff";
      const ins = async (
        id: string | null,
        title: string,
        status = "active",
        archived = false,
      ): Promise<string> => {
        const r = await client.query(
          `INSERT INTO memory.memories (id, project_id, type, title, content, status, archived_at)
           VALUES (COALESCE($1::uuid, gen_random_uuid()), $2, 'semantic', $3, $4, $5,
                   CASE WHEN $6 THEN now() ELSE NULL END)
           RETURNING id`,
          [
            id,
            PBF,
            title,
            `importer-style raw content for ${title}`,
            status,
            archived,
          ],
        );
        return r.rows[0].id as string;
      };
      // importer-style raw INSERTs (bypass storeMemory), incl. keyset EDGE ids
      const idZero = await ins(ZERO, "w8xx-bf-zero");
      const idA = await ins(null, "w8xx-bf-a");
      const idFail = await ins(null, "w8xx-bf-fail");
      const idB = await ins(null, "w8xx-bf-b");
      const idMax = await ins(MAX, "w8xx-bf-max");
      const idArchived = await ins(null, "w8xx-bf-archived", "active", true);
      const idSuperseded = await ins(
        null,
        "w8xx-bf-superseded",
        "superseded",
        false,
      );

      const c1 = await bf.countPendingSummaries(txPool);
      expect(c1 - c0).toBe(5); // archived + non-active rows are NOT pending

      const calls1: string[] = [];
      const run1 = await bf.backfillSummaries({
        pool: txPool,
        summarize: async (title: string) => {
          calls1.push(title);
          if (!title.startsWith("w8xx-bf-")) return null; // foreign rows: skip, never write
          if (title === "w8xx-bf-fail") return null; // per-row failure
          return `sum:${title}`;
        },
        batchSize: 2,
        log: () => {},
      });
      expect(run1.processed).toBe(4);
      expect(run1.skipped).toBeGreaterThanOrEqual(1);

      const summaryOf = async (id: string) =>
        (
          await client.query(
            "SELECT summary FROM memory.memories WHERE id = $1",
            [id],
          )
        ).rows[0].summary as string | null;

      // edge ids are COVERED by the keyset paging
      expect(await summaryOf(idZero)).toBe("sum:w8xx-bf-zero");
      expect(await summaryOf(idMax)).toBe("sum:w8xx-bf-max");
      expect(await summaryOf(idA)).toBe("sum:w8xx-bf-a");
      expect(await summaryOf(idB)).toBe("sum:w8xx-bf-b");
      // per-row failure skipped, run not aborted
      expect(await summaryOf(idFail)).toBeNull();
      // archived / non-active rows never visited, never written
      expect(await summaryOf(idArchived)).toBeNull();
      expect(await summaryOf(idSuperseded)).toBeNull();
      expect(calls1).not.toContain("w8xx-bf-archived");
      expect(calls1).not.toContain("w8xx-bf-superseded");

      const c2 = await bf.countPendingSummaries(txPool);
      expect(c1 - c2).toBe(4);

      // resumability: a second run re-visits ONLY the still-NULL row
      const calls2: string[] = [];
      const run2 = await bf.backfillSummaries({
        pool: txPool,
        summarize: async (title: string) => {
          calls2.push(title);
          return title.startsWith("w8xx-bf-") ? `recovered:${title}` : null;
        },
        batchSize: 2,
        log: () => {},
      });
      expect(run2.processed).toBe(1);
      expect(calls2.filter((t) => t.startsWith("w8xx-bf-"))).toEqual([
        "w8xx-bf-fail",
      ]);
      expect(await summaryOf(idFail)).toBe("recovered:w8xx-bf-fail");
      // already-summarized rows were NOT re-summarized
      expect(await summaryOf(idZero)).toBe("sum:w8xx-bf-zero");

      const c3 = await bf.countPendingSummaries(txPool);
      expect(c2 - c3).toBe(1);
    } finally {
      await client.query("ROLLBACK");
      client.release();
    }
  }, 60_000);

  it("AC-811 — CLI refuses without --yes (cost note, non-zero exit, zero writes)", async () => {
    const before = (
      await pool.query(
        "SELECT count(*)::int AS n FROM memory.memories WHERE summary IS NOT NULL",
      )
    ).rows[0].n as number;
    const tsxBin = join(SERVICE_ROOT, "node_modules", ".bin", "tsx");
    const res = await execFileP(
      tsxBin,
      [join("src", "db", "backfill-summaries.ts")],
      {
        cwd: SERVICE_ROOT,
        env: {
          ...process.env,
          DATABASE_URL,
          ANTHROPIC_API_KEY: "",
          SUMMARIZE_ON_STORE: "",
        },
        timeout: 80_000,
      },
    ).then(
      (r) => ({ code: 0, out: `${r.stdout}${r.stderr}` }),
      (e: { code?: number; stdout?: string; stderr?: string }) => ({
        code: e.code ?? 1,
        out: `${e.stdout ?? ""}${e.stderr ?? ""}`,
      }),
    );
    expect(res.code).not.toBe(0);
    expect(res.out).toMatch(/--yes/);
    const after = (
      await pool.query(
        "SELECT count(*)::int AS n FROM memory.memories WHERE summary IS NOT NULL",
      )
    ).rows[0].n as number;
    expect(after).toBe(before);
  }, 90_000);

  // ── AC-813 spot checks (dry-run, decision/pinned exclusions, fail-open) ──────────
  describe("AC-813 — consolidation spot checks (injected fixtures, tx-rolled-back)", () => {
    const tmp = mkdtempSync(join(tmpdir(), "w8xx-consolidate-"));
    const vecLit = (i: number) => `[${voyage.basis(i).join(",")}]`;

    const insRow = async (
      client: PoolClient,
      opts: {
        project: string;
        title: string;
        createdAt: string;
        basisIdx: number;
        pinned?: boolean;
        sourceKind?: string | null;
      },
    ): Promise<string> => {
      const r = await client.query(
        `INSERT INTO memory.memories
           (project_id, type, title, content, created_at, pinned, source_kind, status, embedding_v2)
         VALUES ($1, 'semantic', $2, $3, $4, $5, $6, 'active', $7::halfvec)
         RETURNING id`,
        [
          opts.project,
          opts.title,
          `consolidation fixture content for ${opts.title}`,
          opts.createdAt,
          opts.pinned ?? false,
          opts.sourceKind ?? null,
          vecLit(opts.basisIdx),
        ],
      );
      return r.rows[0].id as string;
    };

    const snapshot = async (client: PoolClient, project: string) =>
      (
        await client.query(
          "SELECT to_jsonb(m) AS row FROM memory.memories m WHERE project_id = $1 ORDER BY id",
          [project],
        )
      ).rows.map((r) => r.row as Record<string, unknown>);

    const makeJudge = (
      equivalent: Array<{ ids: [string, string]; winner: string }>,
    ) =>
      vi.fn(async (_system: string, user: string): Promise<string> => {
        const out: Record<string, unknown> = {};
        const re =
          /Pair (\d+):\s*\n\s*a \[([0-9a-f-]{36})\][\s\S]*?\n\s*b \[([0-9a-f-]{36})\]/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(user))) {
          const [, num, a, b] = m;
          const match = equivalent.find(
            (p) => p.ids.includes(a) && p.ids.includes(b),
          );
          out[num] = match
            ? {
                verdict: 1,
                keep: match.winner === a ? "a" : "b",
                reason: "conformance dup",
              }
            : { verdict: 0 };
        }
        return JSON.stringify(out);
      });

    it("dry-run (default) makes ZERO row changes and writes the report artifact", async () => {
      const cons = await import("../src/db/consolidate.js");
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const txPool = {
          query: (t: string, p?: unknown[]) => client.query(t, p),
        };
        const aOld = await insRow(client, {
          project: PC1,
          title: "w8xx-dry-old",
          createdAt: "2026-01-01T00:00:00Z",
          basisIdx: 30,
        });
        const bNew = await insRow(client, {
          project: PC1,
          title: "w8xx-dry-new",
          createdAt: "2026-02-01T00:00:00Z",
          basisIdx: 30,
        });
        const before = await snapshot(client, PC1);
        const judge = makeJudge([{ ids: [aOld, bNew], winner: bNew }]);
        const reportPath = join(tmp, "dry.json");
        const { reportPath: written } = await cons.consolidate({
          pool: txPool,
          judge,
          projectId: PC1,
          reportPath,
          log: () => {},
        });
        expect(existsSync(written)).toBe(true);
        expect(() => JSON.parse(readFileSync(written, "utf8"))).not.toThrow();
        const after = await snapshot(client, PC1);
        expect(after).toEqual(before); // dry-run: zero writes
      } finally {
        await client.query("ROLLBACK");
        client.release();
      }
    }, 60_000);

    it("apply: plain loser superseded non-destructively; decision + pinned rows NEVER lose", async () => {
      const cons = await import("../src/db/consolidate.js");
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const txPool = {
          query: (t: string, p?: unknown[]) => client.query(t, p),
        };
        // pair 1: plain old/new duplicates — old must lose to new
        const aOld = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-old",
          createdAt: "2026-01-01T00:00:00Z",
          basisIdx: 31,
        });
        const bNew = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-new",
          createdAt: "2026-02-01T00:00:00Z",
          basisIdx: 31,
        });
        // pair 2: pinned vs normal — the judge TRIES to make the pinned row lose
        const pPin = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-pinned",
          createdAt: "2026-01-05T00:00:00Z",
          basisIdx: 32,
          pinned: true,
        });
        const qNorm = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-normal",
          createdAt: "2026-02-05T00:00:00Z",
          basisIdx: 32,
        });
        // pair 3 (must NEVER form): decision row + its near-identical neighbor
        const dDec = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-decision",
          createdAt: "2026-01-10T00:00:00Z",
          basisIdx: 33,
          sourceKind: "decision",
        });
        const n4 = await insRow(client, {
          project: PC2,
          title: "w8xx-apply-partner",
          createdAt: "2026-02-10T00:00:00Z",
          basisIdx: 33,
        });

        const judge = makeJudge([
          { ids: [aOld, bNew], winner: bNew },
          { ids: [pPin, qNorm], winner: qNorm }, // pinned-loses attempt — must be refused
        ]);
        await cons.consolidate({
          pool: txPool,
          judge,
          projectId: PC2,
          apply: true,
          reportPath: join(tmp, "apply.json"),
          log: () => {},
        });

        const rowOf = async (id: string) =>
          (
            await client.query(
              "SELECT status, superseded_by, archived_at, content FROM memory.memories WHERE id = $1",
              [id],
            )
          ).rows[0] as {
            status: string;
            superseded_by: string | null;
            archived_at: Date | null;
            content: string;
          };

        // plain pair: supersession applied, chain non-destructive
        const loser = await rowOf(aOld);
        expect(loser.status).toBe("superseded");
        expect(loser.superseded_by).toBe(bNew);
        expect(loser.archived_at).toBeNull();
        expect(loser.content).toContain("w8xx-apply-old"); // content untouched
        expect((await rowOf(bNew)).status).toBe("active");

        // pinned row can never be a loser — even when the judge says so
        const pinned = await rowOf(pPin);
        expect(pinned.status).toBe("active");
        expect(pinned.superseded_by).toBeNull();

        // decision rows are excluded from candidate pairs entirely
        const decision = await rowOf(dDec);
        expect(decision.status).toBe("active");
        expect(decision.superseded_by).toBeNull();
        expect((await rowOf(n4)).status).toBe("active");
        const judgeSawDecision = judge.mock.calls.some((c) =>
          c[1].includes(dDec),
        );
        expect(judgeSawDecision).toBe(false);
      } finally {
        await client.query("ROLLBACK");
        client.release();
      }
    }, 60_000);

    it("judge transport failure → fail-open UNJUDGED, zero row changes even with --apply", async () => {
      const cons = await import("../src/db/consolidate.js");
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const txPool = {
          query: (t: string, p?: unknown[]) => client.query(t, p),
        };
        await insRow(client, {
          project: PC3,
          title: "w8xx-fail-old",
          createdAt: "2026-01-01T00:00:00Z",
          basisIdx: 34,
        });
        await insRow(client, {
          project: PC3,
          title: "w8xx-fail-new",
          createdAt: "2026-02-01T00:00:00Z",
          basisIdx: 34,
        });
        const before = await snapshot(client, PC3);
        const judge = vi.fn(async (): Promise<string> => {
          throw new Error("conformance: judge transport down");
        });
        await expect(
          cons.consolidate({
            pool: txPool,
            judge,
            projectId: PC3,
            apply: true,
            reportPath: join(tmp, "fail.json"),
            log: () => {},
          }),
        ).resolves.toBeDefined();
        const after = await snapshot(client, PC3);
        expect(after).toEqual(before);
      } finally {
        await client.query("ROLLBACK");
        client.release();
      }
    }, 60_000);
  });

  // ── AC-815 ──────────────────────────────────────────────────────────────────────
  describe("AC-815 — injectable rrfK", () => {
    const capture = async (rrfK?: number) => {
      const rec = record();
      try {
        const rows = await mem.fuseCandidates({
          projectId: P1,
          query: "identity probe payload",
          limit: 5,
          ...(rrfK === undefined ? {} : { rrfK }),
        });
        return {
          sql: rec.calls.map((c) => c.text).join("\n---\n"),
          params: rec.calls[0]?.params ?? [],
          ids: rows.map((r) => r.id),
        };
      } finally {
        rec.restore();
      }
    };

    it("omitted rrfK is IDENTICAL to rrfK=60 (SQL, params, pool); k reaches BOTH RRF terms as a bound param", async () => {
      // Wave-7 merge: k is a bound parameter ($5::float8), not text interpolation —
      // the SQL text is k-invariant; only the bound value changes with k.
      const def = await capture();
      const k60 = await capture(60);
      const k61 = await capture(61);
      expect(k60.sql).toBe(def.sql);
      expect(k60.params).toEqual(def.params);
      expect([...k60.ids].sort()).toEqual([...def.ids].sort());
      expect(def.params[4]).toBe(60);
      // exactly the two RRF-term slots read the k parameter
      expect((def.sql.match(/\$5::float8/g) ?? []).length).toBe(2);
      // a non-default k changes ONLY the bound value, never the SQL text
      expect(k61.sql).toBe(def.sql);
      expect(k61.params[4]).toBe(61);
      expect(k61.params.filter((_, i) => i !== 4)).toEqual(
        def.params.filter((_, i) => i !== 4),
      );
    });

    it("non-finite or non-positive rrfK throws BEFORE the embed and BEFORE any SQL", async () => {
      for (const bad of [0, -5, Number.NaN, Number.POSITIVE_INFINITY]) {
        voyage.embed.mockClear();
        voyage.embedContextual.mockClear();
        voyage.embedContextualSingle.mockClear();
        const rec = record();
        try {
          await expect(
            mem.fuseCandidates({ projectId: P1, query: "boom", rrfK: bad }),
          ).rejects.toThrow();
        } finally {
          rec.restore();
        }
        expect(voyage.embed).not.toHaveBeenCalled();
        expect(voyage.embedContextual).not.toHaveBeenCalled();
        expect(voyage.embedContextualSingle).not.toHaveBeenCalled();
        expect(rec.calls).toHaveLength(0);
      }
    });
  });
});
