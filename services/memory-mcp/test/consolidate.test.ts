// Wave 5 (WS3), Step 4 — the consolidation guarantees as tests (AC-303/305/306):
//
//   AC-303 — dry-run DEFAULT modifies zero rows and writes the report artifact;
//            --apply only marks losers status='superseded' + superseded_by=<winner>
//            (content never edited/deleted; pinned rows never lose).
//   AC-305 — fail-open judge: transport failure → whole batch UNJUDGED; per-pair
//            schema violations → that pair UNJUDGED while siblings stand; UNJUDGED
//            pairs modify nothing, get NO judged-marker, and re-enter the next run.
//   AC-306 — the batch self-join sweep is the completeness source of truth: rows
//            born via direct SQL (no content_sha256, no metadata.dupCandidates)
//            are still discovered and CAN lose.
//
// DB-backed (disposable :5544 Postgres); the judge is a MOCK (zero network, call
// counters); Voyage is module-mocked for the searchMemory assertion. Skipped
// gracefully without DATABASE_URL.

import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const H = vi.hoisted(() => {
  /** 1024-dim vector with the given (index, value) components; zeros elsewhere. */
  function sparse(components: Array<[number, number]>): number[] {
    const v = new Array<number>(1024).fill(0);
    for (const [i, x] of components) v[i] = x;
    return v;
  }
  /** Unit basis vector for cluster k (index 2k). */
  const base = (k: number) => sparse([[2 * k, 1]]);
  /** Vector at cosine ≈ sim to base(k), orthogonal to every other cluster. */
  const near = (k: number, sim: number) =>
    sparse([
      [2 * k, sim],
      [2 * k + 1, Math.sqrt(1 - sim * sim)],
    ]);
  return { sparse, base, near };
});

// consolidate.ts itself never calls Voyage, but src/memory.js (imported for the
// threshold constant + the searchMemory assertion) does — mock the module so no
// network path exists anywhere in this suite.
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map(() => H.base(400)),
  embedContextualSingle: async (texts: string[]) =>
    texts.map((t) => (t.includes("CHAINTOPIC") ? H.base(0) : H.base(401))),
  rerank: async (_q: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_, i) => ({ index: i, score: 1 - i * 0.01 })),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  CONSOLIDATE_COST_NOTE,
  JUDGE_SYSTEM,
  JudgeTransportError,
  buildJudgeUser,
  consolidate,
  guardConsolidateRun,
  parseVerdicts,
  type CandidatePair,
  type ConsolidateReport,
  type PairRowMeta,
} from "../src/db/consolidate.js";
import { searchMemory } from "../src/memory.js";
import { resolveGoldIds } from "./recall.helper.js";

const here = dirname(fileURLToPath(import.meta.url));

// ── Mock judge: decides per (aId, bId); records every user prompt ────────────────────

type JudgeEntry = { verdict: number; keep?: string; reason?: string };

function makeJudge(
  decide: (a: string, b: string, call: number) => JudgeEntry | undefined,
  opts: { rawOverride?: (call: number) => string | null } = {},
) {
  const calls: string[] = [];
  const judge = async (_system: string, user: string): Promise<string> => {
    calls.push(user);
    const override = opts.rawOverride?.(calls.length);
    if (override !== null && override !== undefined) return override;
    const out: Record<string, unknown> = {};
    const re =
      /Pair (\d+):\n\s*a \[([0-9a-f-]{36})\][\s\S]*?\n\s*b \[([0-9a-f-]{36})\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(user))) {
      const d = decide(m[2], m[3], calls.length);
      if (d !== undefined) out[m[1]] = d;
    }
    return JSON.stringify(out);
  };
  return { judge, calls };
}

// ── Shared DB scaffolding ────────────────────────────────────────────────────────────

let pool: pg.Pool;
let outDir: string;
const RUN = Date.now();
const projects: string[] = [];
const proj = (name: string) => {
  const p = `consolidate-${name}-${RUN}`;
  if (!projects.includes(p)) projects.push(p);
  return p;
};
let reportSeq = 0;
const reportPath = () => join(outDir, `consolidate-${reportSeq++}.json`);

interface SeedOpts {
  title: string;
  content: string;
  summary?: string;
  vec?: number[];
  createdAt?: string;
  pinned?: boolean;
  sourceKind?: string;
  status?: string;
  supersededBy?: string;
  metadata?: Record<string, unknown>;
  sha?: string;
}

async function seed(projectId: string, o: SeedOpts): Promise<string> {
  const { rows } = await pool.query<{ id: string }>(
    `INSERT INTO memory.memories
       (project_id, type, title, content, summary, importance, metadata, pinned,
        source_kind, status, superseded_by, content_sha256, created_at, embedding_v2)
     VALUES ($1, 'semantic', $2, $3, $4, 0.5, $5, $6, $7, $8, $9, $10,
             COALESCE($11::timestamptz, now()), $12::halfvec)
     RETURNING id`,
    [
      projectId,
      o.title,
      o.content,
      o.summary ?? null,
      JSON.stringify(o.metadata ?? {}),
      o.pinned ?? false,
      o.sourceKind ?? null,
      o.status ?? "active",
      o.supersededBy ?? null,
      o.sha ?? null,
      o.createdAt ?? null,
      o.vec ? `[${o.vec.join(",")}]` : null,
    ],
  );
  return rows[0].id;
}

interface RowState {
  status: string;
  superseded_by: string | null;
  title: string;
  content: string;
  metadata: Record<string, unknown>;
}

async function rowState(id: string): Promise<RowState> {
  const { rows } = await pool.query<RowState>(
    `SELECT COALESCE(status,'active') AS status, superseded_by, title, content, metadata
     FROM memory.memories WHERE id = $1`,
    [id],
  );
  return rows[0];
}

const judgedMarker = (s: RowState): unknown =>
  (s.metadata as { consolidation?: { judged?: unknown } }).consolidation
    ?.judged;

const readReport = (path: string): ConsolidateReport =>
  JSON.parse(readFileSync(path, "utf8")) as ConsolidateReport;

describe.skipIf(skip)("consolidate (AC-303/305/306, mocked judge)", () => {
  beforeAll(async () => {
    pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
    outDir = mkdtempSync(join(tmpdir(), "consolidate-test-"));
    const sqlDir = join(here, "..", "sql");
    for (const f of readdirSync(sqlDir)
      .filter((x) => x.endsWith(".sql"))
      .sort()) {
      const sql = readFileSync(join(sqlDir, f), "utf8");
      if (/^\s*--\s*HOLD\b/im.test(sql.split("\n")[0] ?? "")) continue;
      await pool.query(sql);
    }
  });

  afterAll(async () => {
    // superseded_by FK: clear pointers before deleting the fixture projects.
    await pool.query(
      `UPDATE memory.memories SET superseded_by = NULL WHERE project_id = ANY($1)`,
      [projects],
    );
    await pool.query(`DELETE FROM memory.memories WHERE project_id = ANY($1)`, [
      projects,
    ]);
    await pool.end();
    rmSync(outDir, { recursive: true, force: true });
  });

  // ── Block 1: dry-run default, apply, resumability ──────────────────────────────────
  describe("dry-run default + apply + judged-marker resumability", () => {
    const P = proj("basic");
    let A: string, B: string, C: string, X1: string, X2: string;

    beforeAll(async () => {
      A = await seed(P, {
        title: "pool sizing",
        content: "keep pg pool max at 10",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
      });
      B = await seed(P, {
        title: "pool sizing (restated)",
        content: "pg pool max should be 10",
        vec: H.near(0, 0.98),
        createdAt: "2026-06-10T00:00:00Z",
        // Write-path flag pointing at A: the pair must merge to source "both".
        metadata: { dupCandidates: [{ id: "", sim: 0.98 }] },
      });
      // fix the flag id now that A exists
      await pool.query(
        `UPDATE memory.memories
         SET metadata = jsonb_set(metadata, '{dupCandidates}', $2::jsonb)
         WHERE id = $1`,
        [B, JSON.stringify([{ id: A, sim: 0.98 }])],
      );
      C = await seed(P, {
        title: "mid topic",
        content: "loosely related",
        vec: H.near(0, 0.7),
        createdAt: "2026-06-05T00:00:00Z",
      });
      X1 = await seed(P, {
        title: "distinct claim one",
        content: "API tokens rotate every 90 days",
        vec: H.base(5),
        createdAt: "2026-06-02T00:00:00Z",
      });
      X2 = await seed(P, {
        title: "distinct claim two",
        content: "API tokens rotate every 90 days in prod only",
        vec: H.near(5, 0.95),
        createdAt: "2026-06-12T00:00:00Z",
      });
    });

    const decide = (a: string, b: string): JudgeEntry | undefined => {
      if (a === A && b === B) {
        return { verdict: 1, keep: "b", reason: "same claim" };
      }
      if (a === X1 && b === X2) return { verdict: 0 };
      return undefined;
    };

    it("dry-run (DEFAULT): report written with pairs + verdicts, ZERO rows changed", async () => {
      const { judge } = makeJudge(decide);
      const path = reportPath();
      const { report, reportPath: written } = await consolidate({
        pool,
        judge,
        projectId: P,
        reportPath: path,
        scanBatch: 2, // exercise sweep paging + cursor
        log: () => {},
      });
      expect(written).toBe(path);
      const onDisk = readReport(path);
      expect(onDisk.pairs).toEqual(report.pairs);

      expect(report.config.complete).toBe(true);
      expect(report.config.apply).toBe(false);
      expect(report.pairs).toHaveLength(2);
      const ab = report.pairs.find((p) => p.a === A && p.b === B)!;
      expect(ab.verdict).toBe(1);
      expect(ab.keep).toBe("b");
      expect(ab.sim).toBeGreaterThan(0.9);
      expect(ab.source).toBe("both"); // dupCandidates flag ∪ self-join sweep
      const x = report.pairs.find((p) => p.a === X1 && p.b === X2)!;
      expect(x.verdict).toBe(0);
      expect(x.source).toBe("selfJoin");
      // The mid-similarity row never pairs (cosine ≤ 0.9).
      expect(report.pairs.some((p) => p.a === C || p.b === C)).toBe(false);
      expect(report.wouldSupersede).toBe(1);
      expect(report.applied).toBe(0);

      // AC-303 first half: zero rows modified — statuses, pointers, markers.
      for (const id of [A, B, C, X1, X2]) {
        const s = await rowState(id);
        expect(s.status).toBe("active");
        expect(s.superseded_by).toBeNull();
        expect(judgedMarker(s)).toBeUndefined();
      }
    });

    it("--apply: loser flips to superseded + superseded_by=winner; content byte-identical; winner untouched", async () => {
      const beforeA = await rowState(A);
      const beforeB = await rowState(B);
      const { judge } = makeJudge(decide);
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true,
        reportPath: reportPath(),
        log: () => {},
      });
      expect(report.applied).toBe(1);

      const afterA = await rowState(A);
      expect(afterA.status).toBe("superseded");
      expect(afterA.superseded_by).toBe(B);
      expect(afterA.title).toBe(beforeA.title);
      expect(afterA.content).toBe(beforeA.content); // content NEVER edited
      const afterB = await rowState(B);
      expect(afterB.status).toBe("active");
      expect(afterB.superseded_by).toBeNull();
      expect(afterB.content).toBe(beforeB.content);

      // DISTINCT pair: nothing flips, but the judged-marker lands on the OLDER row.
      const afterX1 = await rowState(X1);
      expect(afterX1.status).toBe("active");
      expect(judgedMarker(afterX1)).toEqual([X2]);
    });

    it("resumability: a judged pair is never re-judged — second run makes ZERO judge calls", async () => {
      const { judge, calls } = makeJudge(decide);
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        reportPath: reportPath(),
        log: () => {},
      });
      // A is superseded (excluded by status); (X1,X2) is marker-skipped.
      expect(report.pairs).toHaveLength(0);
      expect(calls).toHaveLength(0);
    });

    it("searchMemory excludes the superseded loser (the user-visible effect)", async () => {
      const result = await searchMemory({
        projectId: P,
        query: "pool sizing",
        limit: 10,
      });
      const ids = result.hits.map((h) => h.id);
      expect(ids).not.toContain(A);
      expect(ids).toContain(B);
    });
  });

  // ── Block 2: pinned + decision-kind + paraphrase guard ─────────────────────────────
  describe("pinned rows, decision rows, paraphrase guard", () => {
    const P = proj("guard");
    let PN1: string, PN2: string, D1: string, D2: string;
    let R1: string, R2: string;

    beforeAll(async () => {
      PN1 = await seed(P, {
        title: "pinned anchor",
        content: "the pinned canonical claim",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
        pinned: true,
      });
      PN2 = await seed(P, {
        title: "pinned near-dup",
        content: "the pinned canonical claim, roughly",
        vec: H.near(0, 0.97),
        createdAt: "2026-06-10T00:00:00Z",
      });
      D1 = await seed(P, {
        title: "decision A",
        content: "we choose drizzle",
        vec: H.base(2),
        createdAt: "2026-06-01T00:00:00Z",
        sourceKind: "decision",
      });
      D2 = await seed(P, {
        title: "decision A restated",
        content: "we choose drizzle orm",
        vec: H.near(2, 0.97),
        createdAt: "2026-06-10T00:00:00Z",
        sourceKind: "decision",
      });
      R1 = await seed(P, {
        title: "RLS for tenants",
        content: "Use Postgres RLS for tenant isolation",
        vec: H.base(4),
        createdAt: "2026-06-01T00:00:00Z",
      });
      R2 = await seed(P, {
        title: "RLS for tenants + org var",
        content:
          "Use Postgres RLS for tenant isolation and set app.current_org_id before every query",
        vec: H.near(4, 0.96),
        createdAt: "2026-06-10T00:00:00Z",
      });
    });

    it("apply run: pinned row never loses (keep→pinned-loser is UNJUDGED); decision rows never enter; restatement stays DISTINCT", async () => {
      const { judge } = makeJudge((a, b) => {
        // Judge (mis)declares the pinned pair equivalent, keeping the NON-pinned
        // side — the harness must refuse (pinned may never lose).
        if (a === PN1 && b === PN2)
          return { verdict: 1, keep: "b", reason: "x" };
        // Paraphrase guard: added operational detail → DISTINCT (mocked as the
        // prompt instructs a real judge).
        if (a === R1 && b === R2) return { verdict: 0 };
        return undefined;
      });
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true,
        reportPath: reportPath(),
        log: () => {},
      });

      // Decision-kind rows are entirely absent from the candidate set (AC-303).
      expect(
        report.pairs.some(
          (p) => [D1, D2].includes(p.a) || [D1, D2].includes(p.b),
        ),
      ).toBe(false);

      // Pinned pair: UNJUDGED (-1), zero changes, NO marker → re-enters next run.
      const pn = report.pairs.find((p) => p.a === PN1 && p.b === PN2)!;
      expect(pn.verdict).toBe(-1);
      const sPN1 = await rowState(PN1);
      const sPN2 = await rowState(PN2);
      expect(sPN1.status).toBe("active");
      expect(sPN2.status).toBe("active");
      expect(judgedMarker(sPN1)).toBeUndefined();

      // Restatement pair: DISTINCT → both stand, marker written (judged).
      const r = report.pairs.find((p) => p.a === R1 && p.b === R2)!;
      expect(r.verdict).toBe(0);
      expect((await rowState(R1)).status).toBe("active");
      expect((await rowState(R2)).status).toBe("active");
      expect(report.applied).toBe(0);
    });

    it("the UNJUDGED pinned pair re-enters the next run (no marker was written)", async () => {
      const { judge, calls } = makeJudge((a, b) =>
        a === PN1 && b === PN2 ? { verdict: 0 } : undefined,
      );
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        reportPath: reportPath(),
        log: () => {},
      });
      expect(calls.length).toBeGreaterThan(0);
      expect(report.pairs).toHaveLength(1); // ONLY the re-entering pinned pair
      expect(report.pairs[0].verdict).toBe(0);
    });
  });

  // ── Block 3: fail-open transport behavior ──────────────────────────────────────────
  describe("fail-open judge (AC-305 transport level)", () => {
    const P = proj("failopen");
    let E1: string, E2: string, F1: string, F2: string;

    beforeAll(async () => {
      E1 = await seed(P, {
        title: "claim E",
        content: "claim E body",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
      });
      E2 = await seed(P, {
        title: "claim E again",
        content: "claim E body again",
        vec: H.near(0, 0.98),
        createdAt: "2026-06-10T00:00:00Z",
      });
      F1 = await seed(P, {
        title: "claim F",
        content: "claim F body",
        vec: H.base(2),
        createdAt: "2026-06-01T00:00:00Z",
      });
      F2 = await seed(P, {
        title: "claim F again",
        content: "claim F body again",
        vec: H.near(2, 0.96),
        createdAt: "2026-06-10T00:00:00Z",
      });
    });

    it("judge throws → the WHOLE batch is UNJUDGED, zero row changes, and the run continues", async () => {
      const judge = vi.fn(async () => {
        throw new Error("ECONNRESET");
      });
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true, // even in apply mode: UNJUDGED must modify nothing
        reportPath: reportPath(),
        log: () => {},
      });
      expect(report.pairs).toHaveLength(2);
      expect(report.pairs.every((p) => p.verdict === -1)).toBe(true);
      expect(report.unjudged).toBe(2);
      expect(report.applied).toBe(0);
      for (const id of [E1, E2, F1, F2]) {
        const s = await rowState(id);
        expect(s.status).toBe("active");
        expect(judgedMarker(s)).toBeUndefined();
      }
    });

    it("a second run RE-adjudicates the failed pairs (no judged-marker was written)", async () => {
      const seen: Array<[string, string]> = [];
      const { judge } = makeJudge((a, b) => {
        seen.push([a, b]);
        return { verdict: 0 };
      });
      await consolidate({
        pool,
        judge,
        projectId: P,
        reportPath: reportPath(),
        log: () => {},
      });
      expect(seen).toContainEqual([E1, E2]);
      expect(seen).toContainEqual([F1, F2]);
    });

    it("wholly unparseable judge JSON → that batch UNJUDGED, the run CONTINUES to the next batch", async () => {
      const Q = proj("failopen2");
      const G1 = await seed(Q, {
        title: "claim G",
        content: "claim G body",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
      });
      const G2 = await seed(Q, {
        title: "claim G again",
        content: "claim G body again",
        vec: H.near(0, 0.98),
        createdAt: "2026-06-10T00:00:00Z",
      });
      const K1 = await seed(Q, {
        title: "claim K",
        content: "claim K body",
        vec: H.base(2),
        createdAt: "2026-06-01T00:00:00Z",
      });
      const K2 = await seed(Q, {
        title: "claim K again",
        content: "claim K body again",
        vec: H.near(2, 0.95),
        createdAt: "2026-06-10T00:00:00Z",
      });
      // judgeBatch=1 → two calls; the FIRST returns garbage, the second is valid.
      const { judge, calls } = makeJudge(() => ({ verdict: 0 }), {
        rawOverride: (call) => (call === 1 ? "not json at all" : null),
      });
      const { report } = await consolidate({
        pool,
        judge,
        projectId: Q,
        judgeBatch: 1,
        reportPath: reportPath(),
        log: () => {},
      });
      expect(calls).toHaveLength(2);
      // Higher-sim pair (G, 0.98) is batched first → UNJUDGED; K pair judged.
      const g = report.pairs.find((p) => p.a === G1 && p.b === G2)!;
      const k = report.pairs.find((p) => p.a === K1 && p.b === K2)!;
      expect(g.verdict).toBe(-1);
      expect(k.verdict).toBe(0);
      expect(report.judged).toBe(1);
      expect(report.unjudged).toBe(1);
    });
  });

  // ── Block 4: per-pair schema violations, siblings stand ────────────────────────────
  describe("per-pair schema violations (AC-305 pair level)", () => {
    const P = proj("schema");
    let U1: string, U2: string, V1: string, V2: string, W1: string, W2: string;

    beforeAll(async () => {
      U1 = await seed(P, {
        title: "claim U",
        content: "claim U body",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
      });
      U2 = await seed(P, {
        title: "claim U again",
        content: "claim U body again",
        vec: H.near(0, 0.98),
        createdAt: "2026-06-10T00:00:00Z",
      });
      V1 = await seed(P, {
        title: "claim V",
        content: "claim V body",
        vec: H.base(2),
        createdAt: "2026-06-01T00:00:00Z",
      });
      V2 = await seed(P, {
        title: "claim V again",
        content: "claim V body again",
        vec: H.near(2, 0.96),
        createdAt: "2026-06-10T00:00:00Z",
      });
      W1 = await seed(P, {
        title: "claim W",
        content: "claim W body",
        vec: H.base(4),
        createdAt: "2026-06-01T00:00:00Z",
      });
      W2 = await seed(P, {
        title: "claim W again",
        content: "claim W body again",
        vec: H.near(4, 0.94),
        createdAt: "2026-06-10T00:00:00Z",
      });
    });

    it("unknown integer / missing keep → ONLY that pair UNJUDGED; the valid sibling still applies", async () => {
      const { judge } = makeJudge((a) => {
        if (a === U1) return { verdict: 2 }; // unknown integer
        if (a === V1) return { verdict: 1 }; // EQUIVALENT without keep
        if (a === W1) return { verdict: 1, keep: "a", reason: "dup" }; // valid
        return undefined;
      });
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true,
        judgeBatch: 3, // ONE batch containing all three pairs
        reportPath: reportPath(),
        log: () => {},
      });

      const byA = new Map(report.pairs.map((p) => [p.a, p]));
      expect(byA.get(U1)!.verdict).toBe(-1);
      expect(byA.get(V1)!.verdict).toBe(-1);
      expect(byA.get(W1)!.verdict).toBe(1);
      expect(byA.get(W1)!.keep).toBe("a");

      // Violating pairs: zero changes, no markers.
      for (const id of [U1, U2, V1, V2]) {
        const s = await rowState(id);
        expect(s.status).toBe("active");
        expect(judgedMarker(s)).toBeUndefined();
      }
      // Sibling applies: keep "a" → loser is W2 (the newer slot-b row).
      const w2 = await rowState(W2);
      expect(w2.status).toBe("superseded");
      expect(w2.superseded_by).toBe(W1);
      expect(report.applied).toBe(1);

      // The full verdict vocabulary round-trips into the report (1/0/-1 → here 1/-1).
      expect(new Set(report.pairs.map((p) => p.verdict))).toEqual(
        new Set([1, -1]),
      );
    });
  });

  // ── Block 5: importer-born rows (AC-306) ───────────────────────────────────────────
  describe("self-join sweep completeness (AC-306)", () => {
    const P = proj("importer");

    it("a direct-SQL row with NO content_sha256 and NO dupCandidates is discovered and CAN lose", async () => {
      // Importer-style: bare INSERT, no write-path invariants (importer.mjs:207-212).
      const I1 = await seed(P, {
        title: "imported claim",
        content: "imported claim body",
        vec: H.base(0),
        createdAt: "2026-06-01T00:00:00Z",
        sourceKind: "claude-session",
      });
      const I2 = await seed(P, {
        title: "imported claim again",
        content: "imported claim body again",
        vec: H.near(0, 0.97),
        createdAt: "2026-06-10T00:00:00Z",
        sourceKind: "codex-session",
      });
      // Premise guard: neither row carries the write-path invariants.
      const pre = await pool.query<{
        content_sha256: string | null;
        has_flags: boolean;
      }>(
        `SELECT content_sha256, (metadata ? 'dupCandidates') AS has_flags
         FROM memory.memories WHERE id = ANY($1::uuid[])`,
        [[I1, I2]],
      );
      for (const r of pre.rows) {
        expect(r.content_sha256).toBeNull();
        expect(r.has_flags).toBe(false);
      }

      const { judge } = makeJudge((a, b) =>
        a === I1 && b === I2
          ? { verdict: 1, keep: "b", reason: "same claim" }
          : undefined,
      );
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true,
        reportPath: reportPath(),
        log: () => {},
      });
      const pair = report.pairs.find((p) => p.a === I1 && p.b === I2)!;
      expect(pair.source).toBe("selfJoin"); // found WITHOUT any write-path flag
      const s = await rowState(I1);
      expect(s.status).toBe("superseded");
      expect(s.superseded_by).toBe(I2);
    });
  });

  // ── Block 5b: summary-aware judge input (wave-3 delta) ─────────────────────────────
  describe("summary-aware judge input (wave-3 delta)", () => {
    const P = proj("summary");

    it("both collection passes hydrate summary: the judge prompt carries summary ?? content", async () => {
      // S1/S2: near-dup pair WITH stored summaries; S2 also carries a write-path
      // dupCandidates flag pointing at S1 so the pair flows through pass A AND the
      // self-join sweep (source: "both") — proving BOTH SQL paths select summary.
      const S1 = await seed(P, {
        title: "summarized claim",
        content: "LONGRAW-ONE body that the judge must not see",
        summary: "DENSE-ONE the stored wave-2 summary",
        vec: H.base(7),
        createdAt: "2026-06-01T00:00:00Z",
      });
      const S2 = await seed(P, {
        title: "summarized claim again",
        content: "LONGRAW-TWO body that the judge must not see",
        summary: "DENSE-TWO the stored wave-2 summary",
        vec: H.near(7, 0.97),
        createdAt: "2026-06-10T00:00:00Z",
      });
      await pool.query(
        `UPDATE memory.memories
         SET metadata = jsonb_set(metadata, '{dupCandidates}', $2::jsonb)
         WHERE id = $1`,
        [S2, JSON.stringify([{ id: S1, sim: 0.97 }])],
      );
      // N1/N2: near-dup pair WITHOUT summaries (self-join only) — content fallback.
      const N1 = await seed(P, {
        title: "plain claim",
        content: "PLAIN-ONE fallback body",
        vec: H.base(8),
        createdAt: "2026-06-01T00:00:00Z",
      });
      await seed(P, {
        title: "plain claim again",
        content: "PLAIN-TWO fallback body",
        vec: H.near(8, 0.97),
        createdAt: "2026-06-10T00:00:00Z",
      });

      const { judge, calls } = makeJudge(() => ({ verdict: 0 }));
      const { report } = await consolidate({
        pool,
        judge,
        projectId: P,
        reportPath: reportPath(),
        log: () => {},
      });

      const sPair = report.pairs.find((p) => p.a === S1 && p.b === S2)!;
      expect(sPair.source).toBe("both"); // flag path AND sweep both contributed
      const prompts = calls.join("\n");
      expect(prompts).toContain("DENSE-ONE the stored wave-2 summary");
      expect(prompts).toContain("DENSE-TWO the stored wave-2 summary");
      expect(prompts).not.toContain("LONGRAW-ONE");
      expect(prompts).not.toContain("LONGRAW-TWO");
      expect(prompts).toContain("PLAIN-ONE fallback body");
      expect(prompts).toContain("PLAIN-TWO fallback body");
      expect(report.pairs.some((p) => p.a === N1)).toBe(true);
    });
  });

  // ── Block 6: supersession chains + gold resolution (AC-106 cross-test) ─────────────
  describe("chains stay resolvable by the eval harness", () => {
    const P = proj("chain");

    it("A→B (pre-existing) then B loses to C → resolveGoldIds(A) = {A,B,C}", async () => {
      const B2 = await seed(P, {
        title: "CHAINTOPIC current",
        content: "CHAINTOPIC decision v2",
        vec: H.base(0),
        createdAt: "2026-05-10T00:00:00Z",
      });
      const A2 = await seed(P, {
        title: "CHAINTOPIC old",
        content: "CHAINTOPIC decision v1",
        createdAt: "2026-05-01T00:00:00Z",
        status: "superseded",
        supersededBy: B2,
      });
      const C2 = await seed(P, {
        title: "CHAINTOPIC newest",
        content: "CHAINTOPIC decision v3",
        vec: H.near(0, 0.97),
        createdAt: "2026-05-20T00:00:00Z",
      });

      const { judge } = makeJudge((a, b) =>
        a === B2 && b === C2
          ? { verdict: 1, keep: "b", reason: "same claim" }
          : undefined,
      );
      await consolidate({
        pool,
        judge,
        projectId: P,
        apply: true,
        reportPath: reportPath(),
        log: () => {},
      });

      const sB = await rowState(B2);
      expect(sB.status).toBe("superseded");
      expect(sB.superseded_by).toBe(C2);

      // Wave-2 gold resolution walks the whole forward chain (AC-106).
      const chains = await resolveGoldIds(pool, [A2]);
      expect(chains.get(A2)).toEqual(new Set([A2, B2, C2]));

      // And the winner is what search returns for the topic.
      const result = await searchMemory({
        projectId: P,
        query: "CHAINTOPIC decision",
        limit: 5,
      });
      const ids = result.hits.map((h) => h.id);
      expect(ids).toContain(C2);
      expect(ids).not.toContain(B2);
      expect(ids).not.toContain(A2);
    });
  });
});

// ── Pure: schema validation, prompt contract, paid gate (no DB) ──────────────────────

describe("parseVerdicts (pure, AC-305 schema)", () => {
  const rowsById = new Map<string, PairRowMeta>([
    [
      "a1",
      {
        id: "a1",
        title: "A",
        content: "a",
        summary: null,
        createdAt: "1",
        pinned: false,
      },
    ],
    [
      "b1",
      {
        id: "b1",
        title: "B",
        content: "b",
        summary: null,
        createdAt: "2",
        pinned: false,
      },
    ],
    [
      "p1",
      {
        id: "p1",
        title: "P",
        content: "p",
        summary: null,
        createdAt: "1",
        pinned: true,
      },
    ],
  ]);
  const pair = (a: string, b: string): CandidatePair => ({
    a,
    b,
    sim: 0.95,
    source: "selfJoin",
  });

  it("valid mixed entries: 1-with-keep, 0, and a missing entry (-1)", () => {
    const out = parseVerdicts(
      JSON.stringify({
        "1": { verdict: 1, keep: "a", reason: "same" },
        "2": { verdict: 0 },
      }),
      [pair("a1", "b1"), pair("a1", "b1"), pair("a1", "b1")],
      rowsById,
    );
    expect(out.map((p) => p.verdict)).toEqual([1, 0, -1]);
    expect(out[0].keep).toBe("a");
    expect(out[0].reason).toBe("same");
  });

  it("unknown integer, non-integer, string verdict, bad keep → that pair UNJUDGED", () => {
    const out = parseVerdicts(
      JSON.stringify({
        "1": { verdict: 2 },
        "2": { verdict: 0.5 },
        "3": { verdict: "1", keep: "a" },
        "4": { verdict: 1, keep: "c" },
        "5": { verdict: 1 },
      }),
      Array.from({ length: 5 }, () => pair("a1", "b1")),
      rowsById,
    );
    expect(out.map((p) => p.verdict)).toEqual([-1, -1, -1, -1, -1]);
  });

  it("a keep that would make a PINNED row lose → UNJUDGED", () => {
    const out = parseVerdicts(
      JSON.stringify({ "1": { verdict: 1, keep: "b", reason: "x" } }),
      [pair("p1", "b1")], // keep b → loser is pinned p1
      rowsById,
    );
    expect(out[0].verdict).toBe(-1);
    // …while keeping the pinned side is fine.
    const ok = parseVerdicts(
      JSON.stringify({ "1": { verdict: 1, keep: "a", reason: "x" } }),
      [pair("p1", "b1")],
      rowsById,
    );
    expect(ok[0].verdict).toBe(1);
  });

  it("wholly unparseable output → JudgeTransportError (whole batch fails open)", () => {
    for (const raw of ["no braces", "{ not json", "[1, 2]"]) {
      expect(() => parseVerdicts(raw, [pair("a1", "b1")], rowsById)).toThrow(
        JudgeTransportError,
      );
    }
  });

  it("tolerates prose/fences around the JSON object", () => {
    const out = parseVerdicts(
      'Sure! ```json\n{"1": {"verdict": 0}}\n```',
      [pair("a1", "b1")],
      rowsById,
    );
    expect(out[0].verdict).toBe(0);
  });
});

describe("judge prompt + paid gate (pure)", () => {
  it("JUDGE_SYSTEM pins the AC-305 contract: integer verdicts, paraphrase guard + worked example, doubt→DISTINCT, pinned rule", () => {
    expect(JUDGE_SYSTEM).toContain("SAME CLAIM");
    expect(JUDGE_SYSTEM).toContain("INTEGER 1 (EQUIVALENT) or 0 (DISTINCT)");
    expect(JUDGE_SYSTEM).toContain(
      "Restatements or added-detail memories are NOT equivalent",
    );
    expect(JUDGE_SYSTEM).toContain("app.current_org_id"); // the worked example
    expect(JUDGE_SYSTEM).toContain("When in doubt, answer 0");
    expect(JUDGE_SYSTEM).toContain("pinned");
  });

  it("buildJudgeUser numbers pairs and includes ids, created dates, and the pinned mark", () => {
    const rows = new Map<string, PairRowMeta>([
      [
        "id-a",
        {
          id: "id-a",
          title: "T A",
          content: "body a",
          summary: null,
          createdAt: "2026-06-01T00:00:00.000Z",
          pinned: true,
        },
      ],
      [
        "id-b",
        {
          id: "id-b",
          title: "T B",
          content: "body b",
          summary: null,
          createdAt: "2026-06-02T00:00:00.000Z",
          pinned: false,
        },
      ],
    ]);
    const user = buildJudgeUser(
      [{ a: "id-a", b: "id-b", sim: 0.95, source: "selfJoin" }],
      rows,
    );
    expect(user).toContain("Pair 1:");
    expect(user).toContain(
      "a [id-a] (created 2026-06-01T00:00:00.000Z, pinned): T A",
    );
    expect(user).toContain("b [id-b] (created 2026-06-02T00:00:00.000Z): T B");
  });

  it("buildJudgeUser prefers the dense summary over content; content is the fallback for NULL and empty-string summaries (wave-3 delta)", () => {
    const rows = new Map<string, PairRowMeta>([
      [
        "id-s",
        {
          id: "id-s",
          title: "T S",
          content: "RAW CONTENT that must not appear when a summary exists",
          summary: "DENSE SUMMARY the judge should read",
          createdAt: "2026-06-01T00:00:00.000Z",
          pinned: false,
        },
      ],
      [
        "id-n",
        {
          id: "id-n",
          title: "T N",
          content: "fallback content for the summaryless row",
          summary: null,
          createdAt: "2026-06-02T00:00:00.000Z",
          pinned: false,
        },
      ],
      [
        "id-e",
        {
          id: "id-e",
          title: "T E",
          content: "EMPTY-SUMMARY fallback content",
          // Out-of-band empty string (the schema writes NULL, but the judge body
          // must match buildRerankDoc's truthiness — never an empty body).
          summary: "",
          createdAt: "2026-06-03T00:00:00.000Z",
          pinned: false,
        },
      ],
    ]);
    const user = buildJudgeUser(
      [
        { a: "id-s", b: "id-n", sim: 0.95, source: "selfJoin" },
        { a: "id-e", b: "id-n", sim: 0.93, source: "selfJoin" },
      ],
      rows,
    );
    expect(user).toContain("DENSE SUMMARY the judge should read");
    expect(user).not.toContain("RAW CONTENT");
    expect(user).toContain("fallback content for the summaryless row");
    expect(user).toContain("EMPTY-SUMMARY fallback content");
  });

  it("guardConsolidateRun refuses without --yes (AC-108) and passes with it", () => {
    expect(guardConsolidateRun([])).toBe(CONSOLIDATE_COST_NOTE);
    expect(guardConsolidateRun(["--apply"])).toBe(CONSOLIDATE_COST_NOTE);
    expect(guardConsolidateRun(["--yes"])).toBeNull();
    expect(guardConsolidateRun(["--yes", "--apply"])).toBeNull();
  });

  it("consolidate is npm-run gated with an import.meta.url main-guard (AC-108)", () => {
    const pkg = JSON.parse(
      readFileSync(join(here, "..", "package.json"), "utf8"),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts["consolidate"]).toBeTruthy();
    const src = readFileSync(
      join(here, "..", "src", "db", "consolidate.ts"),
      "utf8",
    );
    expect(src).toMatch(/import\.meta\.url/);
  });
});
