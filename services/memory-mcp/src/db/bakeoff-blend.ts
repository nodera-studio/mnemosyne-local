// Wave 7 bake-off: blend/decay arms (WS6) + the RRF_K sweep axis — capture once, score
// arms OFFLINE.
//
// The measurement isolates the BLEND MATH: fuse + rerank run exactly ONCE per query per
// k value (the only Voyage spend), persisting the candidate pool + rerank relevances;
// every blend arm is then scored offline from that capture via the pure `blendScores`
// (AC-703). Calling `searchMemory()` per arm instead would rerank per arm — measuring
// model noise and spending |arms|× quota (the Step-2 gotcha).
//
// Two PREDECLARED axes (p-hacking guard — no post-hoc arm invention):
//  - BLEND arms A0..A4 (A0 = the live default, asserted against config.blendConfig at
//    startup) — all scored from the ONE k=60 capture.
//  - RRF_K arms K20/K120 (k=60 IS the control, i.e. A0's capture). HARD RULE: changing
//    k changes POOL COMPOSITION, not just ordering, so the capture phase re-runs per k
//    — k arms canNOT be scored offline from the k=60 capture. AC-703's capture-once
//    invariant applies WITHIN one k. The query embedding is computed once per query and
//    reused across k (qvec), so the k axis adds rerank calls only.
//
// All cross-query aggregation is RANK-based (per-query nDCG deltas, seed-42 paired
// bootstrap, sign test) — Voyage rerank relevances are uncalibrated across queries and
// are never averaged or compared cross-query.
//
// PAID + operator-gated (AC-108/AC-704): `npm run bakeoff:blend -- --yes`, main-guarded,
// refuses without --yes BEFORE any dependency is touched, never imported by server.ts,
// never CI. Defaults to the DEV split — the frozen TEST split is spent ONCE, on the
// single predeclared two-arm confirmation of the one winner (`--split test --arms
// A0,<winner>`), recorded as a flip gate. The winner ships as compose env pins ONLY
// (code defaults remain A0) — see test/retune.md.

import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pool as defaultPool } from "./pool.js";
import { config } from "../config.js";
import {
  blendScores,
  fuseCandidates,
  retrievalConfig,
  RERANK_DOC_TRUNCATION,
  type BlendConfig,
  type FusedCandidate,
} from "../memory.js";
import {
  canonicalizeRanked,
  filterRowsBySplit,
  mrr,
  ndcgAtK,
  pairedBootstrapCI,
  recallAtK,
  resolveGoldIds,
  signTest,
  tryLoadRecallEvalV2,
  type RecallEvalFileV2,
  type RecallEvalRowV2,
} from "../eval-core.js";
import { embedContextualSingle, rerank } from "../voyage.js";

export const SEED = 42;
export const MIN_SPLIT_ROWS = 15;
export const NDCG_K = 10;
export const POOL_RECALL_K = 25;
/** Per-query deltas below this are floating-point noise, not an affected query. */
export const AFFECTED_EPS = 1e-9;

// ── Predeclared arms (FROZEN — the p-hacking guard) ─────────────────────────────────

const A0_BLEND: BlendConfig = {
  form: "additive",
  weights: { relevance: 0.7, recency: 0.2, importance: 0.1 },
  decay: {
    shape: "exp",
    tauDays: 30,
    tauDaysByType: {},
    powerExponent: 0.5,
    exempt: { types: [], sourceKinds: [] },
  },
};

/** A1/A2 shared decay routing: episodic keeps τ30 (the base), semantic + procedural
 *  slow to τ90, entity rows and decision records are exempt (reference material). */
const PER_TYPE_TAUS = { semantic: 90, procedural: 90 } as const;
const PER_TYPE_EXEMPT = {
  types: ["entity" as const],
  sourceKinds: ["decision"],
};

export interface ArmSpec {
  name: string;
  /** The RRF k this arm's CAPTURE runs at (pool composition is k-dependent). */
  rrfK: number;
  blend: BlendConfig;
  note: string;
}

/** The 5 predeclared blend arms + the 2 non-control k arms. A0 doubles as the k=60
 *  control for the RRF_K axis; K-arms score the A0 blend over their own capture so the
 *  k change is isolated. FROZEN in code — never invent arms against results. */
export function predeclaredArms(): ArmSpec[] {
  return [
    {
      name: "A0",
      rrfK: 60,
      blend: A0_BLEND,
      note: "control (= live defaults)",
    },
    {
      name: "A1",
      rrfK: 60,
      blend: {
        ...A0_BLEND,
        decay: {
          ...A0_BLEND.decay,
          tauDaysByType: { ...PER_TYPE_TAUS },
          exempt: {
            types: [...PER_TYPE_EXEMPT.types],
            sourceKinds: [...PER_TYPE_EXEMPT.sourceKinds],
          },
        },
      },
      note: "per-type exp: semantic/procedural τ90; entity + decision exempt",
    },
    {
      name: "A2",
      rrfK: 60,
      blend: {
        ...A0_BLEND,
        decay: {
          ...A0_BLEND.decay,
          shape: "power",
          powerExponent: 0.5,
          tauDaysByType: { ...PER_TYPE_TAUS },
          exempt: {
            types: [...PER_TYPE_EXEMPT.types],
            sourceKinds: [...PER_TYPE_EXEMPT.sourceKinds],
          },
        },
      },
      note: "per-type power 1/(1+age/τ)^0.5; same τ/exemptions as A1",
    },
    {
      name: "A3",
      rrfK: 60,
      blend: {
        form: "multiplicative",
        // weights.relevance is unused by the multiplicative form (rel is the factor);
        // recorded as 1 so the artifact never suggests a 0.7 scaling that never runs.
        weights: { relevance: 1, recency: 0.2, importance: 0.05 },
        decay: { ...A0_BLEND.decay },
      },
      note: "multiplicative rel × (1 + 0.2·recency + 0.05·importance); decay as A0 (isolates the form change)",
    },
    {
      name: "A4",
      rrfK: 60,
      blend: {
        ...A0_BLEND,
        weights: { relevance: 1, recency: 0, importance: 0 },
      },
      note: "relevance-only (decay inert at weight 0) — pure rerank ordering",
    },
    {
      name: "K20",
      rrfK: 20,
      blend: A0_BLEND,
      note: "RRF_K axis: k=20 capture, A0 blend (k=60 control is A0)",
    },
    {
      name: "K120",
      rrfK: 120,
      blend: A0_BLEND,
      note: "RRF_K axis: k=120 capture, A0 blend (k=60 control is A0)",
    },
  ];
}

// ── Injectable dependencies (mock-tested with zero network, AC-703) ─────────────────

/** Minimal query surface — resolveGoldIds + the one batched type-slice SELECT. Row
 *  shapes differ per statement, so rows are deliberately untyped here (the live dep is
 *  the pg Pool; tests inject a plain object). */
export interface Queryable {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  query: (text: string, params: unknown[]) => Promise<{ rows: any[] }>;
}

export interface BakeoffBlendDeps {
  pool: Queryable;
  fuse: typeof fuseCandidates;
  rerankFn: (
    query: string,
    docs: string[],
    topK: number,
  ) => Promise<Array<{ index: number; score: number }>>;
  /** One paid embed per query, cached and REUSED across k values (qvec). */
  embedQuery: (query: string) => Promise<number[]>;
  log?: (msg: string) => void;
  /** Frozen scoring instant for blendScores age math (defaults to run start). */
  now?: number;
}

// ── Capture + arm scoring ────────────────────────────────────────────────────────────

interface CapturedQuery {
  row: RecallEvalRowV2;
  chains: Map<string, Set<string>>;
  candidates: FusedCandidate[];
  relevanceById: Map<string, number>;
  poolRecall: number;
}

export interface ArmPerQuery {
  id: string;
  query: string;
  ndcg: number;
  mrr: number;
  rank: number | null;
  poolRecall: number;
}

export interface ArmResult {
  armConfig: { rrfK: number; blend: BlendConfig; note: string };
  perQuery: ArmPerQuery[];
  ndcgMean: number;
  poolRecall: number;
}

export interface SliceStat {
  meanDelta: number;
  n: number;
}

export interface ArmComparison {
  meanDelta: number;
  ci: { mean: number; ciLow: number; ciHigh: number };
  signTest: { wins: number; losses: number; ties: number; p: number };
  affected: number;
  sliceByType: Record<string, SliceStat>;
  sliceTemporal: SliceStat | null;
  label: "WIN" | "REGRESSION" | "INCONCLUSIVE";
}

export interface BakeoffBlendArtifact {
  config: ReturnType<typeof retrievalConfig>;
  seed: number;
  split: "dev" | "test";
  rows: number;
  arms: Record<string, ArmResult>;
  comparisons: Record<string, ArmComparison>;
  verdict: { winner: string; reason: string };
  rejectedForNow: string[];
}

function deepEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || !a || !b) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a as object).sort();
  const kb = Object.keys(b as object).sort();
  if (ka.length !== kb.length || ka.some((k, i) => k !== kb[i])) return false;
  return ka.every((k) =>
    deepEqual(
      (a as Record<string, unknown>)[k],
      (b as Record<string, unknown>)[k],
    ),
  );
}

/** Capture ONE query at ONE k: fuse → truncate docs exactly as live → rerank. The only
 *  Voyage spend of the bakeoff; every blend arm reads this capture for free. */
async function captureQuery(
  deps: BakeoffBlendDeps,
  projectId: string,
  row: RecallEvalRowV2,
  qvec: number[],
  rrfK: number,
): Promise<Pick<CapturedQuery, "candidates" | "relevanceById">> {
  const candidates = await deps.fuse({
    projectId,
    query: row.query,
    qvec,
    rrfK,
  });
  const relevanceById = new Map<string, number>();
  if (candidates.length > 0) {
    const docs = candidates.map((r) =>
      `${r.title}\n${r.content}`.slice(0, RERANK_DOC_TRUNCATION),
    );
    const ranked = await deps.rerankFn(row.query, docs, candidates.length);
    for (const r of ranked) relevanceById.set(candidates[r.index].id, r.score);
  }
  return { candidates, relevanceById };
}

/** Score one arm OFFLINE from its k's capture: blendScores → top-10 → canonicalize →
 *  nDCG@10 (+ MRR / first-relevant-rank diagnostics). Zero network. */
function scoreArm(
  arm: ArmSpec,
  captures: CapturedQuery[],
  now: number,
): ArmResult {
  const perQuery: ArmPerQuery[] = captures.map((cap) => {
    const scored = blendScores(
      cap.candidates.map((c) => ({
        ...c,
        relevance: cap.relevanceById.get(c.id) ?? 0,
      })),
      arm.blend,
      now,
    );
    const topIds = scored.slice(0, NDCG_K).map((c) => c.id);
    const canonical = canonicalizeRanked(topIds, cap.chains);
    const rel = new Set(cap.row.relevantIds);
    let rank: number | null = null;
    for (let i = 0; i < canonical.length; i++) {
      if (rel.has(canonical[i])) {
        rank = i + 1;
        break;
      }
    }
    return {
      id: cap.row.id,
      query: cap.row.query,
      ndcg: ndcgAtK(canonical, cap.row.relevantIds, NDCG_K),
      mrr: mrr(canonical, cap.row.relevantIds),
      rank,
      poolRecall: cap.poolRecall,
    };
  });
  const mean = (f: (q: ArmPerQuery) => number) =>
    perQuery.length === 0
      ? 0
      : perQuery.reduce((a, q) => a + f(q), 0) / perQuery.length;
  return {
    armConfig: { rrfK: arm.rrfK, blend: arm.blend, note: arm.note },
    perQuery,
    ndcgMean: mean((q) => q.ndcg),
    poolRecall: mean((q) => q.poolRecall),
  };
}

function sliceMean(deltas: number[]): SliceStat {
  return {
    meanDelta:
      deltas.length === 0
        ? 0
        : deltas.reduce((a, b) => a + b, 0) / deltas.length,
    n: deltas.length,
  };
}

/** Compare one arm against A0: per-query deltas joined by STABLE row id (never array
 *  position), seed-42 bootstrap CI, sign test, affected count, per-type + temporal
 *  slices. */
function compareArm(
  arm: ArmResult,
  control: ArmResult,
  rowsById: Map<string, RecallEvalRowV2>,
  typeByRowId: Map<string, string>,
): ArmComparison {
  const controlById = new Map(control.perQuery.map((q) => [q.id, q]));
  const joined = arm.perQuery.map((q) => {
    const base = controlById.get(q.id);
    if (!base) {
      throw new Error(`bakeoff:blend id drift: control is missing row ${q.id}`);
    }
    return { id: q.id, delta: q.ndcg - base.ndcg };
  });
  const deltas = joined.map((j) => j.delta);
  const ci = pairedBootstrapCI(deltas, { seed: SEED });
  const st = signTest(deltas);
  const byType = new Map<string, number[]>();
  const temporal: number[] = [];
  for (const j of joined) {
    const t = typeByRowId.get(j.id) ?? "unknown";
    (byType.get(t) ?? byType.set(t, []).get(t)!).push(j.delta);
    if (rowsById.get(j.id)?.facet === "temporal") temporal.push(j.delta);
  }
  const label: ArmComparison["label"] =
    ci.ciHigh < 0 ? "REGRESSION" : ci.ciLow > 0 ? "WIN" : "INCONCLUSIVE";
  return {
    meanDelta: ci.mean,
    ci,
    signTest: st,
    affected: deltas.filter((d) => Math.abs(d) > AFFECTED_EPS).length,
    sliceByType: Object.fromEntries(
      [...byType.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([t, ds]) => [t, sliceMean(ds)]),
    ),
    sliceTemporal: temporal.length > 0 ? sliceMean(temporal) : null,
    label,
  };
}

export interface RunBakeoffBlendOpts {
  evalFile: RecallEvalFileV2;
  projectId: string;
  split: "dev" | "test";
  /** Arm subset (Step-4 confirmation runs use `--arms A0,<winner>`). A0 (the control)
   *  is always included. Default: all predeclared arms. */
  arms?: string[];
}

export async function runBakeoffBlend(
  deps: BakeoffBlendDeps,
  opts: RunBakeoffBlendOpts,
): Promise<BakeoffBlendArtifact> {
  const log = deps.log ?? ((m: string) => console.error(m));
  const now = deps.now ?? Date.now();

  // Startup drift assertion: the control MUST equal the live serialized config, or
  // every comparison is against the wrong baseline. (Deliberately checked against
  // retrievalConfig() — the same snapshot the artifact records.)
  const live = retrievalConfig();
  if (!deepEqual(A0_BLEND, live.blend) || live.rrfK !== 60) {
    throw new Error(
      "bakeoff:blend drift: arm A0 no longer equals the live blend config " +
        "(check BLEND_*/DECAY_*/RECENCY_TAU_DAYS* env pins — the control must be " +
        "the live baseline). Live: " +
        JSON.stringify({ rrfK: live.rrfK, blend: live.blend }),
    );
  }

  const all = predeclaredArms();
  const names = opts.arms ?? all.map((a) => a.name);
  const unknown = names.filter((n) => !all.some((a) => a.name === n));
  if (unknown.length > 0) {
    throw new Error(
      `bakeoff:blend: unknown arm(s) ${unknown.join(", ")} — predeclared arms: ${all
        .map((a) => a.name)
        .join(", ")}`,
    );
  }
  const selected = all.filter((a) => a.name === "A0" || names.includes(a.name));

  const rows = filterRowsBySplit(opts.evalFile, opts.split);
  if (rows.length < MIN_SPLIT_ROWS) {
    throw new Error(
      `bakeoff:blend needs at least ${MIN_SPLIT_ROWS} ${opts.split}-split rows; found ${rows.length}. ` +
        "Check the eval file before spending quota.",
    );
  }
  // Empty-gold rows score 0 on every arm, silently diluting all deltas — refuse
  // BEFORE any spend, like the row-count guard above.
  const goldless = rows.filter((r) => r.relevantIds.length === 0);
  if (goldless.length > 0) {
    throw new Error(
      `bakeoff:blend: ${goldless.length} ${opts.split}-split row(s) have empty relevantIds ` +
        `(first: ${goldless[0].id}) — fix the eval file before spending quota.`,
    );
  }

  // One batched SELECT: memory type of each row's FIRST gold id (the per-type slice).
  const firstGoldIds = [...new Set(rows.map((r) => r.relevantIds[0]))];
  const { rows: typeRows } = await deps.pool.query(
    `SELECT id, type FROM memory.memories WHERE id = ANY($1::uuid[])`,
    [firstGoldIds],
  );
  const typeByGold = new Map(typeRows.map((r) => [r.id, r.type]));
  const typeByRowId = new Map(
    rows.map((r) => [r.id, typeByGold.get(r.relevantIds[0]) ?? "unknown"]),
  );

  // Gold chains are k-independent — resolve once per row, reuse across every capture.
  const chainsByRow = new Map<string, Map<string, Set<string>>>();
  for (const row of rows) {
    chainsByRow.set(row.id, await resolveGoldIds(deps.pool, row.relevantIds));
  }

  // ── Capture phase: fuse + rerank exactly ONCE per query per k (AC-703) ────────────
  const kValues = [...new Set(selected.map((a) => a.rrfK))].sort(
    (a, b) => a - b,
  );
  const qvecByRow = new Map<string, number[]>();
  const capturesByK = new Map<number, CapturedQuery[]>();
  for (const k of kValues) {
    const captures: CapturedQuery[] = [];
    for (const row of rows) {
      let qvec = qvecByRow.get(row.id);
      if (!qvec) {
        qvec = await deps.embedQuery(row.query);
        qvecByRow.set(row.id, qvec);
      }
      const chains = chainsByRow.get(row.id)!;
      const { candidates, relevanceById } = await captureQuery(
        deps,
        opts.projectId,
        row,
        qvec,
        k,
      );
      const poolIds = canonicalizeRanked(
        candidates.map((c) => c.id),
        chains,
      );
      captures.push({
        row,
        chains,
        candidates,
        relevanceById,
        poolRecall: recallAtK(poolIds, row.relevantIds, POOL_RECALL_K),
      });
    }
    capturesByK.set(k, captures);
    log(
      `bakeoff:blend capture k=${k}: ${captures.length} queries (fuse+rerank once each)`,
    );
  }

  // ── Arm phase: offline, free ──────────────────────────────────────────────────────
  const arms: Record<string, ArmResult> = {};
  for (const arm of selected) {
    arms[arm.name] = scoreArm(arm, capturesByK.get(arm.rrfK)!, now);
    log(
      `bakeoff:blend arm ${arm.name}: nDCG@10=${arms[arm.name].ndcgMean.toFixed(4)} poolRecall@25=${arms[arm.name].poolRecall.toFixed(4)} (${arm.note})`,
    );
  }

  const rowsById = new Map(rows.map((r) => [r.id, r]));
  const comparisons: Record<string, ArmComparison> = {};
  for (const arm of selected) {
    if (arm.name === "A0") continue;
    comparisons[arm.name] = compareArm(
      arms[arm.name],
      arms.A0,
      rowsById,
      typeByRowId,
    );
  }

  // ── Predeclared winner rule ───────────────────────────────────────────────────────
  // Qualify: CI excludes zero on the WIN side AND wins > losses AND affected > 0.
  // Among qualifiers pick the highest mean delta; none ⇒ KEEP A0. Shared across both
  // axes — ONE overall winner, ONE frozen-TEST confirmation (never one per axis).
  const qualifiers = Object.entries(comparisons).filter(
    ([, c]) =>
      c.ci.ciLow > 0 && c.signTest.wins > c.signTest.losses && c.affected > 0,
  );
  qualifiers.sort(([, a], [, b]) => b.meanDelta - a.meanDelta);
  const verdict =
    qualifiers.length > 0
      ? {
          winner: qualifiers[0][0],
          reason: `CI excludes zero on the win side (ciLow ${qualifiers[0][1].ci.ciLow.toFixed(4)} > 0), wins ${qualifiers[0][1].signTest.wins} > losses ${qualifiers[0][1].signTest.losses}, affected ${qualifiers[0][1].affected} — highest mean delta ${qualifiers[0][1].meanDelta.toFixed(4)}. Confirm ONCE on the frozen test split before pinning.`,
        }
      : {
          winner: "A0",
          reason:
            "no arm qualified (win-side CI + wins > losses + affected > 0) — KEEP A0 (code defaults; compose untouched).",
        };

  return {
    config: live,
    seed: SEED,
    split: opts.split,
    rows: rows.length,
    arms,
    comparisons,
    verdict,
    // Design lock: getMemory MUTATES access_count/last_accessed_at (state-dependent
    // evals + self-reinforcing boosts) — revisit only with a shadow-logging design.
    rejectedForNow: ["access-based decay"],
  };
}

// ── Artifact + verdict printing ──────────────────────────────────────────────────────

export function writeBakeoffBlendArtifact(
  artifact: BakeoffBlendArtifact,
  dir: string,
): string {
  mkdirSync(dir, { recursive: true });
  const date = new Date().toISOString().slice(0, 10);
  // The test-split confirmation gets its own name so it never clobbers the same-day
  // dev-split selection artifact.
  const suffix = artifact.split === "test" ? "-test" : "";
  const path = join(dir, `${date}-bakeoff-blend${suffix}.json`);
  writeFileSync(path, JSON.stringify(artifact, null, 2) + "\n");
  return path;
}

export function formatVerdict(artifact: BakeoffBlendArtifact): string {
  const f4 = (x: number) => x.toFixed(4);
  const lines: string[] = [
    `=== BLEND/DECAY BAKEOFF (${artifact.split} split, ${artifact.rows} rows, seed ${artifact.seed}) ===`,
    `A0 control: nDCG@10=${f4(artifact.arms.A0.ndcgMean)} poolRecall@25=${f4(artifact.arms.A0.poolRecall)}`,
  ];
  for (const [name, c] of Object.entries(artifact.comparisons)) {
    const st = c.signTest;
    const slices = Object.entries(c.sliceByType)
      .map(
        ([t, s]) =>
          `${t} ${s.meanDelta >= 0 ? "+" : ""}${f4(s.meanDelta)} (n=${s.n})`,
      )
      .join(", ");
    lines.push(
      `${name} vs A0: Δ=${f4(c.meanDelta)} CI[${f4(c.ci.ciLow)}, ${f4(c.ci.ciHigh)}] ` +
        `sign ${st.wins}W/${st.losses}L/${st.ties}T (p=${st.p.toFixed(4)}) affected=${c.affected} — ${c.label}`,
    );
    lines.push(`   by-type: ${slices}`);
    lines.push(
      `   temporal: ${c.sliceTemporal ? `${f4(c.sliceTemporal.meanDelta)} (n=${c.sliceTemporal.n})` : "no temporal-facet rows in this split"}`,
    );
  }
  lines.push(
    `verdict: ${artifact.verdict.winner} — ${artifact.verdict.reason}`,
  );
  lines.push(`rejected for now: ${artifact.rejectedForNow.join(", ")}`);
  return lines.join("\n");
}

// ── CLI plumbing (pure — unit-tested without invoking any dependency) ───────────────

export const BAKEOFF_BLEND_COST_NOTE =
  "bakeoff:blend is a PAID operator script: it embeds every split query once and " +
  "reranks once per query per RRF-k value (blend arms are scored offline for free). " +
  "Nothing was run. Re-run with an explicit consent flag:\n" +
  "  npm run bakeoff:blend -- --yes [--split dev|test] [--arms A0,A3,...] " +
  "[--eval <path>] [--project-id <id>] [--out-dir <dir>]";

export interface BakeoffBlendArgs {
  yes: boolean;
  split: "dev" | "test";
  evalPath?: string;
  arms?: string[];
  projectId: string;
  outDir?: string;
}

/** Parse CLI args; returns the refusal/cost note (a string) when --yes is absent —
 *  BEFORE any pool/Voyage/file dependency is touched (AC-704). */
export function parseBakeoffBlendArgs(
  argv: string[],
): BakeoffBlendArgs | string {
  if (!argv.includes("--yes")) return BAKEOFF_BLEND_COST_NOTE;
  const opt = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`);
    const v = i !== -1 ? argv[i + 1] : undefined;
    return v !== undefined && v.startsWith("--") ? undefined : v;
  };
  const split = opt("split") ?? "dev";
  if (split !== "dev" && split !== "test") {
    return `bakeoff:blend: invalid --split "${split}" — use "dev" or "test" (dev is the selection surface; test is the ONE confirmation run)`;
  }
  const armsCsv = opt("arms");
  return {
    yes: true,
    split,
    evalPath: opt("eval"),
    arms: armsCsv
      ? armsCsv
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
      : undefined,
    projectId: opt("project-id") ?? config.defaultProjectId,
    outDir: opt("out-dir"),
  };
}

// ── CLI entrypoint (npm run bakeoff:blend -- --yes) ──────────────────────────────────
// Skipped when imported (tests import the exported functions directly).
const isMain = process.argv[1]
  ? import.meta.url === new URL(`file://${process.argv[1]}`).href
  : false;

if (isMain) {
  const args = parseBakeoffBlendArgs(process.argv.slice(2));
  if (typeof args === "string") {
    console.error(args);
    process.exit(1);
  }
  (async () => {
    const evalFile = tryLoadRecallEvalV2(args.evalPath);
    if (!evalFile) {
      throw new Error(
        "bakeoff:blend: the eval file is not v2 — migrate/approve gold first (AC-109)",
      );
    }
    const rows = filterRowsBySplit(evalFile, args.split).length;
    const kGrid = [
      ...new Set(
        predeclaredArms()
          .filter(
            (a) => !args.arms || a.name === "A0" || args.arms.includes(a.name),
          )
          .map((a) => a.rrfK),
      ),
    ].sort((a, b) => a - b);
    // PAID gate: the cost line prints BEFORE any live Voyage call.
    console.error(
      `bakeoff:blend cost: ${rows} query embeds (reused across k) + ${rows}×${kGrid.length} rerank calls ` +
        `(${args.split} split, k values: ${kGrid.join(", ")}). Blend arms score offline for free.`,
    );
    const artifact = await runBakeoffBlend(
      {
        pool: defaultPool,
        fuse: fuseCandidates,
        rerankFn: rerank,
        embedQuery: async (q) => (await embedContextualSingle([q], "query"))[0],
        log: (m) => console.error(m),
      },
      {
        evalFile,
        projectId: args.projectId,
        split: args.split,
        arms: args.arms,
      },
    );
    const here = dirname(fileURLToPath(import.meta.url));
    const outDir = args.outDir ?? join(here, "..", "..", "test", "runs");
    const path = writeBakeoffBlendArtifact(artifact, outDir);
    console.error("");
    console.error(formatVerdict(artifact));
    console.error(`artifact: ${path}`);
    await defaultPool.end();
  })().catch(async (e) => {
    console.error(e);
    await defaultPool.end().catch(() => {});
    process.exit(1);
  });
}
