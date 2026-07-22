// Wave 5 fix pass — the resume/checkpoint + CLI-safety guarantees as tests:
//
//   - The report file is a REAL resumability checkpoint: (re)written complete:false
//     after every judge batch (and every sweep page via onPage), so a crash at any
//     point never re-buys already-judged verdicts.
//   - Resume semantics: prior judged pairs are carried (never re-judged), prior
//     UNJUDGED pairs re-enter, the sweep resumes after the cursor, and re-entered
//     pairs are hydrated (real titles in the judge prompt, never "(missing)").
//   - The CLI decision table (resolveCliAction): --apply never judge-and-applies
//     fresh; an apply record is never overwritten by a dry-run.
//   - Apply path: the fully-judged report lands BEFORE the first row flips (undo-log
//     guarantee); applyFromReport re-checks the DB guards (a row pinned AFTER the
//     dry-run review survives) and re-applying is idempotent.
//   - Candidate collection survives malformed caller-supplied metadata (scalar
//     dupCandidates, non-uuid ids, non-numeric sims, scalar consolidation/judged)
//     and the marker write repairs a non-object consolidation instead of no-opping.
//
// DB-backed (disposable :5544 Postgres); the judge is a MOCK (zero network).

import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import pg from "pg";

const DATABASE_URL = process.env.DATABASE_URL ?? "";
const skip = !DATABASE_URL;

const H = vi.hoisted(() => {
  function sparse(components: Array<[number, number]>): number[] {
    const v = new Array<number>(1024).fill(0);
    for (const [i, x] of components) v[i] = x;
    return v;
  }
  const base = (k: number) => sparse([[2 * k, 1]]);
  const near = (k: number, sim: number) =>
    sparse([
      [2 * k, sim],
      [2 * k + 1, Math.sqrt(1 - sim * sim)],
    ]);
  return { sparse, base, near };
});

// consolidate.ts never calls Voyage, but src/memory.js (imported transitively for the
// threshold constant) does — mock the module so no network path exists in this suite.
vi.mock("../src/voyage.js", () => ({
  embed: async (texts: string[]) => texts.map(() => H.base(400)),
  embedContextualSingle: async (texts: string[]) =>
    texts.map(() => H.base(401)),
  rerank: async (_q: string, docs: string[], topK: number) =>
    docs.slice(0, topK).map((_, i) => ({ index: i, score: 1 - i * 0.01 })),
  toVectorLiteral: (v: number[]) => `[${v.join(",")}]`,
}));

process.env.VOYAGE_API_KEY ??= "test-key";

import {
  applyFromReport,
  collectPairs,
  consolidate,
  resolveCliAction,
  type ConsolidateReport,
} from "../src/db/consolidate.js";

// ── Mock judge (same shape as consolidate.test.ts) ────────────────────────────────────

type JudgeEntry = { verdict: number; keep?: string; reason?: string };

function makeJudge(
  decide: (a: string, b: string, call: number) => JudgeEntry | undefined,
  opts: { onCall?: (call: number) => void } = {},
) {
  const calls: string[] = [];
  const judge = async (_system: string, user: string): Promise<string> => {
    calls.push(user);
    opts.onCall?.(calls.length);
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
  const p = `consolidate-resume-${name}-${RUN}`;
  if (!projects.includes(p)) projects.push(p);
  return p;
};
let reportSeq = 0;
const nextReportPath = () => join(outDir, `consolidate-${reportSeq++}.json`);

const here = dirname(fileURLToPath(import.meta.url));

// Strictly-increasing created_at per seeded row: two fast successive INSERTs can
// land in the same now() microsecond, and slot normalization (older-first) then
// tie-breaks by RANDOM uuid — flipping which row carries the marker/loses ~50%
// of the time (the applyFromReport flake the wave-7 gate reproduced).
let seedClock = Date.parse("2026-01-01T00:00:00Z");

async function seed(
  projectId: string,
  o: {
    title: string;
    content: string;
    vec?: number[];
    pinned?: boolean;
    metadata?: Record<string, unknown> | string;
  },
): Promise<string> {
  seedClock += 60_000;
  const { rows } = await pool.query<{ id: string }>(
    `INSERT INTO memory.memories
       (project_id, type, title, content, importance, metadata, pinned, embedding_v2, created_at)
     VALUES ($1, 'semantic', $2, $3, 0.5, $4::jsonb, $5, $6::halfvec, $7::timestamptz)
     RETURNING id`,
    [
      projectId,
      o.title,
      o.content,
      typeof o.metadata === "string"
        ? o.metadata
        : JSON.stringify(o.metadata ?? {}),
      o.pinned ?? false,
      o.vec ? `[${o.vec.join(",")}]` : null,
      new Date(seedClock).toISOString(),
    ],
  );
  return rows[0].id;
}

async function status(
  id: string,
): Promise<{ status: string; superseded_by: string | null }> {
  const { rows } = await pool.query<{
    status: string;
    superseded_by: string | null;
  }>(
    `SELECT COALESCE(status,'active') AS status, superseded_by
     FROM memory.memories WHERE id = $1`,
    [id],
  );
  return rows[0];
}

const readReport = (path: string): ConsolidateReport =>
  JSON.parse(readFileSync(path, "utf8")) as ConsolidateReport;

/** Minimal fabricated report for the pure CLI-table tests. */
function fakeReport(
  over: Partial<ConsolidateReport["config"]>,
): ConsolidateReport {
  return {
    config: {
      threshold: 0.9,
      model: "test",
      batch: 500,
      judgeBatch: 8,
      topS: 3,
      cursor: null,
      complete: true,
      apply: false,
      projectId: "p",
      generatedAt: "2026-07-04T00:00:00.000Z",
      ...over,
    },
    pairs: [],
    judged: 0,
    unjudged: 0,
    wouldSupersede: 0,
    applied: 0,
  };
}

// ── Pure: the CLI decision table ─────────────────────────────────────────────────────

describe("resolveCliAction (CLI decision table — --apply never judges fresh)", () => {
  it("--report without --apply refuses", () => {
    const a = resolveCliAction({
      apply: false,
      existing: null,
      explicitReport: fakeReport({}),
    });
    expect(a.kind).toBe("refuse");
  });

  it("--report with --apply applies the explicit report", () => {
    const rep = fakeReport({});
    const a = resolveCliAction({
      apply: true,
      existing: null,
      explicitReport: rep,
    });
    expect(a).toMatchObject({ kind: "applyReport", report: rep });
  });

  it("--apply with no same-day report refuses (dry-run first)", () => {
    const a = resolveCliAction({ apply: true, existing: null });
    expect(a.kind).toBe("refuse");
    expect((a as { message: string }).message).toMatch(/dry-run first/);
  });

  it("--apply on an incomplete dry-run checkpoint refuses (finish + review first)", () => {
    const a = resolveCliAction({
      apply: true,
      existing: fakeReport({ complete: false }),
    });
    expect(a.kind).toBe("refuse");
    expect((a as { message: string }).message).toMatch(/INCOMPLETE/);
  });

  it("--apply on a reviewed complete dry-run report applies it", () => {
    const rep = fakeReport({ complete: true, apply: false });
    const a = resolveCliAction({ apply: true, existing: rep });
    expect(a).toMatchObject({ kind: "applyReport", report: rep });
  });

  it("--apply on a same-day APPLY record re-applies idempotently (crash recovery)", () => {
    const rep = fakeReport({ apply: true });
    const a = resolveCliAction({ apply: true, existing: rep });
    expect(a).toMatchObject({ kind: "applyReport", report: rep });
    expect((a as { note: string }).note).toMatch(/idempotent/);
  });

  it("dry-run refuses to overwrite a same-day APPLY record (the undo log)", () => {
    const a = resolveCliAction({
      apply: false,
      existing: fakeReport({ apply: true }),
    });
    expect(a.kind).toBe("refuse");
    expect((a as { message: string }).message).toMatch(/undo log/);
  });

  it("dry-run feeds a same-day non-apply report as priorReport (resume OR verdict reuse)", () => {
    const complete = fakeReport({ complete: true });
    const partial = fakeReport({ complete: false });
    expect(resolveCliAction({ apply: false, existing: complete })).toEqual({
      kind: "run",
      apply: false,
      priorReport: complete,
    });
    expect(resolveCliAction({ apply: false, existing: partial })).toEqual({
      kind: "run",
      apply: false,
      priorReport: partial,
    });
    expect(resolveCliAction({ apply: false, existing: null })).toEqual({
      kind: "run",
      apply: false,
      priorReport: undefined,
    });
  });
});

// ── DB-backed: checkpoints, resume, apply path, malformed metadata ──────────────────

describe.skipIf(skip)(
  "consolidate resume/checkpoint + apply safety (DB)",
  () => {
    beforeAll(async () => {
      pool = new pg.Pool({ connectionString: DATABASE_URL, max: 4 });
      outDir = mkdtempSync(join(tmpdir(), "consolidate-resume-"));
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
      if (projects.length > 0) {
        await pool.query(
          `DELETE FROM memory.memories WHERE project_id = ANY($1)`,
          [projects],
        );
      }
      await pool.end();
      rmSync(outDir, { recursive: true, force: true });
    });

    /** Seed one >threshold pair in cluster k; returns [olderId, newerId] by creation. */
    async function seedPair(
      projectId: string,
      k: number,
    ): Promise<[string, string]> {
      const a = await seed(projectId, {
        title: `cluster${k} older`,
        content: `cluster ${k} assertion, original`,
        vec: H.base(k),
      });
      const b = await seed(projectId, {
        title: `cluster${k} newer`,
        content: `cluster ${k} assertion, restated`,
        vec: H.near(k, 0.95),
      });
      return [a, b];
    }

    it("writes a complete:false checkpoint after every judge batch — a crash mid-judging keeps bought verdicts", async () => {
      const PROJ = proj("ckpt");
      await seedPair(PROJ, 0);
      await seedPair(PROJ, 1);
      const reportPath = nextReportPath();

      let midRun: ConsolidateReport | null = null;
      const { judge, calls } = makeJudge(() => ({ verdict: 0 }), {
        onCall: (call) => {
          // Batch 2's judge call happens AFTER batch 1's checkpoint write: the state
          // on disk at this moment is exactly what a crash here would leave behind.
          if (call === 2 && existsSync(reportPath)) {
            midRun = readReport(reportPath);
          }
        },
      });

      await consolidate({
        pool,
        judge,
        projectId: PROJ,
        judgeBatch: 1,
        reportPath,
        log: () => {},
      });

      expect(calls.length).toBe(2);
      expect(midRun).not.toBeNull();
      const ck = midRun as unknown as ConsolidateReport;
      expect(ck.config.complete).toBe(false);
      expect(ck.pairs).toHaveLength(2);
      expect(ck.pairs.filter((p) => p.verdict !== -1)).toHaveLength(1); // bought
      expect(ck.pairs.filter((p) => p.verdict === -1)).toHaveLength(1); // still queued

      // The final write flips complete:true with everything judged.
      const final = readReport(reportPath);
      expect(final.config.complete).toBe(true);
      expect(final.judged).toBe(2);
    });

    it("collectPairs onPage checkpoints every non-final sweep page with an advancing cursor", async () => {
      const PROJ = proj("onpage");
      await seedPair(PROJ, 0);
      await seedPair(PROJ, 1);

      const snapshots: Array<{ cursor: string | null; n: number }> = [];
      const collected = await collectPairs({
        pool,
        projectId: PROJ,
        scanBatch: 1, // 4 a-rows → pages of 1 → onPage after each full page
        onPage: (cursor, pairsSoFar) =>
          snapshots.push({ cursor, n: pairsSoFar.length }),
      });

      expect(collected.complete).toBe(true);
      expect(collected.pairs).toHaveLength(2);
      expect(snapshots.length).toBeGreaterThanOrEqual(3);
      for (const s of snapshots) expect(s.cursor).toBeTruthy();
      // Cursors advance monotonically; collected pairs never shrink.
      for (let i = 1; i < snapshots.length; i++) {
        expect(snapshots[i].cursor! >= snapshots[i - 1].cursor!).toBe(true);
        expect(snapshots[i].n).toBeGreaterThanOrEqual(snapshots[i - 1].n);
      }
    });

    it("resumes from a complete:false prior report: carried never re-judged, UNJUDGED re-enter hydrated, sweep skips ≤ cursor", async () => {
      const PROJ = proj("resume");
      const [a0, b0] = await seedPair(PROJ, 0); // carried (already judged, verdict 1)
      const [a1, b1] = await seedPair(PROJ, 1); // UNJUDGED — must re-enter
      const maxId = [a0, b0, a1, b1].sort().at(-1)!;

      const prior: ConsolidateReport = {
        ...fakeReport({
          complete: false,
          cursor: maxId, // sweep already covered everything before the "crash"
          projectId: PROJ,
        }),
        pairs: [
          {
            a: a0 < b0 ? a0 : b0,
            b: a0 < b0 ? b0 : a0,
            sim: 0.95,
            source: "selfJoin",
            verdict: 1,
            keep: "a",
          },
          {
            a: a1 < b1 ? a1 : b1,
            b: a1 < b1 ? b1 : a1,
            sim: 0.95,
            source: "selfJoin",
            verdict: -1,
          },
        ],
        judged: 1,
        unjudged: 1,
        wouldSupersede: 1,
        applied: 0,
      };

      const { judge, calls } = makeJudge(() => ({ verdict: 0 }));
      const reportPath = nextReportPath();
      const { report } = await consolidate({
        pool,
        judge,
        projectId: PROJ,
        priorReport: prior,
        reportPath,
        log: () => {},
      });

      // Exactly ONE judge call, containing ONLY the re-entered cluster-1 pair —
      // the carried pair was not re-bought, the sweep found nothing past the cursor.
      expect(calls).toHaveLength(1);
      expect(calls[0]).toContain(a1);
      expect(calls[0]).toContain(b1);
      expect(calls[0]).not.toContain(a0);
      // hydrateRowMeta proof: real titles in the prompt, not the "(missing)" fallback.
      expect(calls[0]).toContain("cluster1");
      expect(calls[0]).not.toContain("(missing)");

      expect(report.pairs).toHaveLength(2);
      expect(report.judged).toBe(2); // carried verdict + fresh verdict
      expect(report.config.complete).toBe(true);
    });

    it("a COMPLETE prior dry-run's verdicts are reused — re-running judges nothing new", async () => {
      const PROJ = proj("reuse");
      await seedPair(PROJ, 0);

      const first = makeJudge(() => ({ verdict: 0 }));
      const reportPath = nextReportPath();
      const run1 = await consolidate({
        pool,
        judge: first.judge,
        projectId: PROJ,
        reportPath,
        log: () => {},
      });
      expect(first.calls).toHaveLength(1);
      expect(run1.report.config.complete).toBe(true);

      // Second same-day dry-run (the CLI feeds the completed report as priorReport):
      // zero judge calls — the bought verdict is carried, the re-found pair deduped.
      const second = makeJudge(() => ({ verdict: 0 }));
      const run2 = await consolidate({
        pool,
        judge: second.judge,
        projectId: PROJ,
        priorReport: run1.report,
        reportPath: nextReportPath(),
        log: () => {},
      });
      expect(second.calls).toHaveLength(0);
      expect(run2.report.judged).toBe(1);
    });

    it("apply mode writes the fully-judged report BEFORE the first row flips (undo-log guarantee)", async () => {
      const PROJ = proj("undolog");
      const [a, b] = await seedPair(PROJ, 0);
      const reportPath = nextReportPath();

      let reportAtFirstFlip: ConsolidateReport | null = null;
      const wrappedPool: typeof pool = Object.create(pool);
      wrappedPool.query = ((text: string, params?: unknown[]) => {
        if (
          typeof text === "string" &&
          text.includes("SET status = 'superseded'") &&
          reportAtFirstFlip === null &&
          existsSync(reportPath)
        ) {
          reportAtFirstFlip = readReport(reportPath);
        }
        return pool.query(text as never, params as never);
      }) as typeof pool.query;

      const { judge } = makeJudge(() => ({
        verdict: 1,
        keep: "a",
        reason: "dup",
      }));
      await consolidate({
        pool: wrappedPool,
        judge,
        projectId: PROJ,
        apply: true,
        reportPath,
        log: () => {},
      });

      expect(reportAtFirstFlip).not.toBeNull();
      const pre = reportAtFirstFlip as unknown as ConsolidateReport;
      expect(pre.config.complete).toBe(true);
      expect(pre.config.apply).toBe(true);
      expect(pre.applied).toBe(0); // written pre-apply
      expect(pre.pairs.some((p) => p.verdict === 1)).toBe(true);

      expect((await status(b)).status).toBe("superseded");
      expect((await status(b)).superseded_by).toBe(a);
    });

    it("applyFromReport: a loser pinned AFTER the dry-run review survives (DB guard), siblings apply, re-apply is a no-op", async () => {
      const PROJ = proj("applyrep");
      const [, b0] = await seedPair(PROJ, 0);
      const [a1, b1] = await seedPair(PROJ, 1);

      const { judge } = makeJudge(() => ({
        verdict: 1,
        keep: "a",
        reason: "dup",
      }));
      const reportPath = nextReportPath();
      const { report } = await consolidate({
        pool,
        judge,
        projectId: PROJ,
        reportPath,
        log: () => {},
      });
      expect(report.wouldSupersede).toBe(2);
      expect((await status(b0)).status).toBe("active"); // dry-run: zero flips

      // Operator pins one loser AFTER reviewing the report — the report is now stale
      // for that pair; the DB-level pinned guard must be the last line of defense.
      await pool.query(
        `UPDATE memory.memories SET pinned = true WHERE id = $1`,
        [b0],
      );

      const res1 = await applyFromReport(
        pool,
        readReport(reportPath),
        () => {},
      );
      expect(res1.superseded).toBe(1); // only the unpinned sibling
      expect((await status(b0)).status).toBe("active"); // pinned survived
      expect((await status(b1)).status).toBe("superseded");
      expect((await status(b1)).superseded_by).toBe(a1);

      // Idempotent re-apply: no further flips, no duplicate judged-marker appends.
      const res2 = await applyFromReport(
        pool,
        readReport(reportPath),
        () => {},
      );
      expect(res2.superseded).toBe(0);
      const { rows } = await pool.query<{ judged: string[] }>(
        `SELECT ARRAY(SELECT jsonb_array_elements_text(metadata #> '{consolidation,judged}')) AS judged
       FROM memory.memories WHERE id = $1`,
        [a1],
      );
      expect(rows[0].judged.filter((x) => x === b1)).toHaveLength(1);
    });

    it("applyFromReport warns on a projectId mismatch (env mix-up flag)", async () => {
      const logs: string[] = [];
      await applyFromReport(
        pool,
        { ...fakeReport({ projectId: "project-A" }), pairs: [] },
        (m) => logs.push(m),
        "project-B",
      );
      expect(logs.join("\n")).toMatch(/WARNING.*project-A.*project-B/s);
    });

    it("malformed caller metadata never aborts the run, and the marker write repairs a scalar consolidation", async () => {
      const PROJ = proj("malformed");
      // A real >threshold pair whose rows carry every malformed shape at once:
      // scalar consolidation (marker target), scalar dupCandidates, and a dupCandidates
      // array with a non-uuid id + non-numeric sim.
      const a = await seed(PROJ, {
        title: "malformed older",
        content: "assertion original",
        vec: H.base(7),
        metadata: JSON.stringify({
          consolidation: "scalar-garbage",
          dupCandidates: "not-an-array",
        }),
      });
      const b = await seed(PROJ, {
        title: "malformed newer",
        content: "assertion restated",
        vec: H.near(7, 0.95),
        metadata: JSON.stringify({
          dupCandidates: [
            { id: "not-a-uuid", sim: "NaN-ish" },
            "a bare string",
          ],
          consolidation: { judged: "scalar-not-array" },
        }),
      });

      const { judge } = makeJudge(() => ({ verdict: 0 }));
      const reportPath = nextReportPath();
      // Apply mode so the verdict-0 marker lands — proving MARKER_SQL repaired the
      // scalar `consolidation` instead of silently no-opping (perpetual re-judge).
      const { report } = await consolidate({
        pool,
        judge,
        projectId: PROJ,
        apply: true,
        reportPath,
        log: () => {},
      });

      expect(report.pairs).toHaveLength(1); // found via the sweep despite the garbage
      expect(report.judged).toBe(1);

      // The marker lives on the older-by-created_at row: `a` was seeded first.
      const older = a;
      const partner = b;
      const { rows } = await pool.query<{ judged: string[] | null }>(
        `SELECT CASE WHEN jsonb_typeof(metadata #> '{consolidation,judged}') = 'array'
                   THEN ARRAY(SELECT jsonb_array_elements_text(metadata #> '{consolidation,judged}'))
                   ELSE NULL END AS judged
       FROM memory.memories WHERE id = $1`,
        [older],
      );
      expect(rows[0].judged).toContain(partner);

      // Re-run: the marker is now readable → the pair is skipped, zero judge calls.
      const rerun = makeJudge(() => ({ verdict: 0 }));
      const { report: r2 } = await consolidate({
        pool,
        judge: rerun.judge,
        projectId: PROJ,
        reportPath: nextReportPath(),
        log: () => {},
      });
      expect(rerun.calls).toHaveLength(0);
      expect(r2.pairs).toHaveLength(0);
    });
  },
);
