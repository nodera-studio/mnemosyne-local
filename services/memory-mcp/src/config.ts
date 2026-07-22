import type {
  BlendConfig,
  BlendForm,
  DecayShape,
  MemoryType,
} from "./memory.js";

// ── Blend/decay env parsing (wave-7, AC-702) ─────────────────────────────────────────
// Every knob is ALWAYS materialized to its default so the serialized retrievalConfig()
// key set is stable whether or not a pin is set. Invalid values THROW loudly at load:
// these knobs reorder search results, and a typo'd compose pin silently falling back to
// the default would pass every post-swap gate while shipping the wrong config.

function parseEnum<T extends string>(
  name: string,
  raw: string | undefined,
  allowed: readonly T[],
  fallback: T,
): T {
  if (raw === undefined || raw === "") return fallback;
  if ((allowed as readonly string[]).includes(raw)) return raw as T;
  throw new Error(
    `${name}="${raw}" is invalid — use one of: ${allowed.join(", ")}`,
  );
}

function parseNum(
  name: string,
  raw: string | undefined,
  fallback: number,
  opts: { positive?: boolean } = {},
): number {
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error(`${name}="${raw}" is not a number`);
  if (opts.positive && n <= 0) {
    // A zero/negative τ inverts or zeroes the decay curve — the exact silent
    // misconfiguration the fail-loud contract above exists to prevent.
    throw new Error(`${name}="${raw}" must be a finite number > 0`);
  }
  return n;
}

const MEMORY_TYPES = ["episodic", "semantic", "procedural", "entity"] as const;

/** `RECENCY_TAU_DAYS_BY_TYPE` — JSON object map of memory type → τ days, e.g.
 *  `{"semantic":90,"procedural":90}`. Unknown keys are tolerated (simply never match
 *  a row type); values must be finite positive numbers. */
function parseTauByType(
  raw: string | undefined,
): Partial<Record<MemoryType, number>> {
  if (raw === undefined || raw === "") return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`RECENCY_TAU_DAYS_BY_TYPE is not valid JSON: ${raw}`);
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`RECENCY_TAU_DAYS_BY_TYPE must be a JSON object map`);
  }
  const out: Partial<Record<MemoryType, number>> = {};
  for (const [k, v] of Object.entries(parsed)) {
    if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) {
      throw new Error(
        `RECENCY_TAU_DAYS_BY_TYPE["${k}"] must be a finite number > 0`,
      );
    }
    out[k as MemoryType] = v;
  }
  return out;
}

/** `DECAY_EXEMPT` — CSV of `type:<memoryType>` / `source_kind:<kind>` entries, e.g.
 *  `type:entity,source_kind:decision`. Default empty (nothing exempt). */
function parseExempt(raw: string | undefined): {
  types: MemoryType[];
  sourceKinds: string[];
} {
  const types: MemoryType[] = [];
  const sourceKinds: string[] = [];
  for (const entry of (raw ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)) {
    if (entry.startsWith("type:")) {
      const t = entry.slice(5);
      if (!(MEMORY_TYPES as readonly string[]).includes(t)) {
        throw new Error(
          `DECAY_EXEMPT type "${t}" is invalid — use one of: ${MEMORY_TYPES.join(", ")}`,
        );
      }
      types.push(t as MemoryType);
    } else if (entry.startsWith("source_kind:")) {
      const k = entry.slice(12);
      if (k === "") {
        throw new Error(
          `DECAY_EXEMPT entry "${entry}" has an empty source_kind`,
        );
      }
      sourceKinds.push(k);
    } else {
      throw new Error(
        `DECAY_EXEMPT entry "${entry}" is invalid — use type:<t> or source_kind:<k>`,
      );
    }
  }
  return { types, sourceKinds };
}

// The blend/decay knobs the scoring pipeline runs with (src/memory.ts blendScores).
// Code defaults = the pre-wave-7 live behavior (additive 0.7/0.2/0.1, exp, τ30, no
// exemptions) — a bakeoff winner ships as compose env pins ONLY, so rollback = delete
// the pins. τ (RECENCY_TAU_DAYS) is the 1/e constant, NOT the half-life: at age = τ
// the recency weight is e⁻¹ ≈ 36.8%; the true half-life is τ·ln2 ≈ 20.8 days for τ=30.
// (Renamed from RECENCY_HALFLIFE_DAYS in wave-7 — value 30 unchanged, on purpose.)
// TODO(operator/retune): CLOSED by the wave-7 blend/decay bakeoff for the blend weights
// and RRF k — decision record: services/memory-mcp/test/retune.md (wave-7 section).
const blendConfig: BlendConfig = {
  form: parseEnum<BlendForm>(
    "BLEND_FORM",
    process.env.BLEND_FORM,
    ["additive", "multiplicative"],
    "additive",
  ),
  weights: {
    relevance: parseNum(
      "BLEND_W_RELEVANCE",
      process.env.BLEND_W_RELEVANCE,
      0.7,
    ),
    recency: parseNum("BLEND_W_RECENCY", process.env.BLEND_W_RECENCY, 0.2),
    importance: parseNum(
      "BLEND_W_IMPORTANCE",
      process.env.BLEND_W_IMPORTANCE,
      0.1,
    ),
  },
  decay: {
    shape: parseEnum<DecayShape>(
      "DECAY_SHAPE",
      process.env.DECAY_SHAPE,
      ["exp", "power"],
      "exp",
    ),
    tauDays: parseNum("RECENCY_TAU_DAYS", process.env.RECENCY_TAU_DAYS, 30, {
      positive: true,
    }),
    tauDaysByType: parseTauByType(process.env.RECENCY_TAU_DAYS_BY_TYPE),
    powerExponent: parseNum(
      "DECAY_POWER_EXPONENT",
      process.env.DECAY_POWER_EXPONENT,
      0.5,
      { positive: true },
    ),
    exempt: parseExempt(process.env.DECAY_EXEMPT),
  },
};

export const config = {
  databaseUrl: process.env.DATABASE_URL ?? "",
  blendConfig,
  voyageApiKey: process.env.VOYAGE_API_KEY ?? "",
  port: parseInt(process.env.PORT ?? "8080", 10),
  // If unset, the server runs open (fine for localhost-only). Set in prod.
  bearerToken: process.env.MNEMO_TOKEN ?? "",
  defaultProjectId: process.env.DEFAULT_PROJECT_ID ?? "default",
  embedModel: process.env.VOYAGE_EMBED_MODEL ?? "voyage-4-large",
  // Contextual embedder (voyage-context-4) for the Wave-P blue/green swap. Used by
  // embedContextual()/embedContextualSingle(); the legacy embedModel stays for the
  // burn-in embed() path. NOT required in assertServerConfig — unset falls back here.
  // context-4 is a drop-in successor to context-3: same endpoint/shape/1024-int8, cheaper
  // ($0.12 vs $0.18 /M) and higher retrieval quality. Vectors are NOT cross-compatible with
  // context-3, so the whole embedding_v2 column must be re-embedded consistently on a swap.
  contextModel: process.env.VOYAGE_CONTEXT_MODEL ?? "voyage-context-4",
  rerankModel: process.env.VOYAGE_RERANK_MODEL ?? "rerank-2.5-lite",
  // recall pool size handed to the reranker (doc: keep 20–25)
  candidatePool: parseInt(process.env.CANDIDATE_POOL ?? "25", 10),
  // Anthropic key + models for the PAID operator scripts (distill-eval now; consolidate
  // in wave-5). Deliberately NOT in assertServerConfig — the MCP server runs without
  // them; src/llm.ts raises a clear message when a script needs the key and it is
  // absent (AC-108). Haiku-class default: cheap judge/distill work.
  anthropicApiKey: process.env.ANTHROPIC_API_KEY ?? "",
  // Write-time summary gate (src/summarize.ts) — an EXTRA gate beyond the key being
  // present, because compose passes ANTHROPIC_API_KEY through for operator scripts:
  // without this flag, provisioning the key for `consolidate` would silently start
  // spending on every memory_store (write-path-stays-cheap lock, program wave-5
  // design lock). Default off; NOT in assertServerConfig.
  summarizeOnStore: process.env.SUMMARIZE_ON_STORE === "1",
  // `||` (not `??`): compose materializes `${CONSOLIDATE_MODEL:-}` as an EMPTY STRING
  // in-container, and an empty model string 400s at the Anthropic API — empty must
  // fall back exactly like unset (found live, 2026-07-04 backfill all-skip incident).
  consolidateModel: process.env.CONSOLIDATE_MODEL || "claude-haiku-4-5",
  // Eval-candidate distiller model; falls back to the consolidate model.
  distillModel:
    process.env.DISTILL_MODEL ||
    process.env.CONSOLIDATE_MODEL ||
    "claude-haiku-4-5",
  // Origin allowlist (DNS-rebinding guard). Empty = only no-Origin (loopback) passes.
  allowedOrigins: (process.env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
  // Cloudflare Access (Managed OAuth) — validate Cf-Access-Jwt-Assertion on the public path.
  cfAccessTeamDomain: process.env.CF_ACCESS_TEAM_DOMAIN ?? "",
  cfAccessAud: process.env.CF_ACCESS_AUD ?? "",
};

export function assertServerConfig(): void {
  if (!config.databaseUrl) throw new Error("DATABASE_URL required");
  if (!config.voyageApiKey) throw new Error("VOYAGE_API_KEY required");
}
