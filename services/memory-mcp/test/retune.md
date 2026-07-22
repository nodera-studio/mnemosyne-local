# Wave P retune procedure — the embedder-flip gate decision record (AC-024)

After the contextual re-embed (`embedding_v2` via voyage-context-4) and **before** the
query flip goes live, the RRF/blend/rerank constants MUST be re-tuned on a held-out eval
set. Re-embed couples to fusion/rerank: flipping the embedding column without re-tuning
silently regresses recall (the locked caveat, memory `0350c509`). This file is the
decision record the operator fills in after the **live** backfill run.

> The code flip is already landed (`src/memory.ts` reads `embedding_v2`, embeds the query
> via `embedContextualSingle`, and reads the tuned constants — `RRF_K` plus, since
> wave 7, the blend/decay knobs via `config.blendConfig` in `src/config.ts`;
> `RECENCY_HALFLIFE_DAYS` was renamed `RECENCY_TAU_DAYS`, see the wave-7 section). It
> went live at the **gated redeploy** — the constants stay at no-regression defaults
> until a measured retune lands (wave-7 bakeoff below).

## What the operator runs (live — paid Voyage quota)

These are the steps I (the operator) run; they are NOT run by the implementer/CI.

1. **Apply migrations on the live DB** (loopback):
   ```
   cd services/memory-mcp && DATABASE_URL=<live> npm run migrate
   ```
   Applies `003_context_embeddings.sql` (ADD COLUMN embedding_v2). `004` is HOLD-skipped.
2. **Backfill the contextual column** (deliberate, paid):
   ```
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> npm run backfill:context
   ```
   Verify completeness (AC-021):
   ```
   SELECT count(*) FROM memory.memories
   WHERE archived_at IS NULL AND COALESCE(status,'active')='active' AND embedding_v2 IS NULL;
   -- must be 0
   ```
3. **Build the contextual HNSW index** (CONCURRENTLY, own connection):
   ```
   DATABASE_URL=<live> npm run index:v2
   ```
   **Then, after this (and after ANY future mass re-embed of `embedding_v2`), reindex +
   vacuum** — a mass re-embed leaves the HNSW graph full of dead tuples, which silently
   degrades result counts, and VACUUM alone is slow against a bloated index (reindex
   first, then vacuum). Run via psql on the live DB; CONCURRENTLY needs its own
   connection (not the batch migrate runner), takes ~2× the plain build, and keeps
   writes flowing:
   ```sql
   REINDEX INDEX CONCURRENTLY mem_hnsw_v2;
   VACUUM (VERBOSE) memory.memories;
   ```
   (Skip the REINDEX only when `index:v2` just created `mem_hnsw_v2` from scratch on an
   already-backfilled column — a fresh build has no dead tuples; still VACUUM.)
4. **Freeze the eval set (wave-2 v2 gold).** Migrate the title-keyed seeds to stable-id
   v2 gold (`npm run gold:migrate` proposes `recall-eval.v2.json`; a human approves every
   id), expand to ≥30–50 rows from the LIVE corpus (distill/harvest proposals, human
   approval), and freeze the **test** split — flip decisions are scored on it, never on
   the freely-recomputed dev split (AC-109). Shape + policy: `test/eval.md`.
5. **Baseline + candidate runs through the wave-2 eval spine** (replaces the retired
   scalar-baseline mechanism — see the note below). Record a run artifact and freeze it
   as the committed baseline, then the gate does the statistics:
   ```
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> EVAL_RECORD=1 npx vitest run recall-gate
   cp test/runs/<date>-dev.json test/runs/baseline-dev.json   # commit it
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> npx vitest run recall-gate
   ```
   The gate joins per-query nDCG@10 against `baseline-dev.json` by row id and fails on
   a paired-bootstrap 95% CI excluding zero on the regression side, plus a coarse
   pool-layer Recall@25 floor (−0.05 absolute). Fill `BEFORE`/`AFTER` below from the
   two artifacts' aggregates. Full semantics: `test/eval.md`.
6. **Retune constants** on the SAME held-out set (sweep `RRF_K` ∈ {30,40,60,80}, the
   blend weights, confirm rerank-2.5-lite still helps), then set the finalized values in
   `src/memory.ts` and record them below.
7. **Flip gate (AC-023):** flip to v2 ONLY if the v2 run shows no regression vs the v1
   baseline on the gate metrics (paired-bootstrap CI on per-query nDCG@10 + the pool
   Recall@25 floor; a lift is preferred). If it regresses, do NOT flip — investigate
   retune/embedder. (The original scalar "v2 recall@10 ≥ v1 recall@10" phrasing is
   retired — see the wave-2 note below.)
8. After a burn-in window, apply `sql/004_drop_legacy_embedding.sql` (lift the HOLD) and
   remove the legacy `embed()` write from `memory.ts`.

## Recorded numbers — G0, 2026-07-03 (operator record, from the original private deployment)

These numbers were recorded against the original private deployment's own memory
corpus (2124 active memories) and its own frozen 58-row gold set — neither of which
ships in this public template repo (see `test/fixtures/recall-eval.json`'s header for
why a fresh deployment needs its own gold set rather than reusing someone else's).
Kept here only as a worked example of what a G0 recorded-numbers entry looks like once
you've run the flip gate on your own corpus: Corpus: 2124 active memories, fully
backfilled with **voyage-context-4** (AC-021 probe = 0), `mem_hnsw_v2` REINDEXed +
VACUUMed. Gold: frozen v2 `recall-eval.json` (58 rows, 38 dev / 20 test, incl. the
amended-G0 temporal expansion m-051..m-058). Artifact: `test/runs/baseline-dev.json`
(= the committed `2026-07-03-dev.json` run over the 38-row dev split):
**pool Recall@25 0.9474 · nDCG@10 0.7404 · MRR@10 0.7673** — these were THE gate
reference numbers for that deployment. The table below is the earlier 33-row
pre-temporal split, kept because it is the only matched context-3-vs-context-4
comparison on record.

**BEFORE (v1, `embedding`) was never measured**: the scalar `V1_RECALL_AT_10_BASELINE`
mechanism was retired before any v1 number was recorded, and the branch code cannot
exercise the legacy query path (the flip is compiled in). The flip decision therefore
rests on (a) healthy absolute v2 numbers below, (b) the research-grounded expectation
for contextual embeddings, and (c) the intact rollback path (legacy `embedding` column
+ `mem_hnsw` stay until burn-in; `004` stays HOLD).

| Metric (33-row pre-temporal dev split) | context-3 v2 (accidental arm, see note) | context-4 v2 |
| -------------------------------------- | --------------------------------------- | ------------ |
| pool Recall@25                          | 0.9394                                   | 0.9394       |
| nDCG@10                                 | 0.6917                                   | 0.7035       |
| MRR@10                                  | 0.7121                                   | 0.7321       |
| dev-split size                          | 33                                       | 33 (same set) |

> **Accidental context-3 vs context-4 comparison:** a compose env-default drift
> (fixed the same day — the model is now hard-pinned in `compose.yaml`) first ran the
> backfill AND the eval with voyage-context-3, producing a *matched* context-3 pipeline
> measurement before the corrected context-4 re-run of the identical dev split on the
> identical corpus rows. context-4 wins directionally: +0.012 nDCG@10, +0.020 MRR@10,
> equal pool recall. N=33, not statistically gated — recorded as corroboration, not
> proof. Zero-nDCG rows in both arms: m-018, m-023 (known-hard gold).

## Finalized tuned constants (set in `src/memory.ts`)

Kept at the no-regression defaults for the G0 flip: the baseline is healthy and the
step-6 sweep (RRF_K × blend × reranker) is deferred to the wave-5/6 harness machinery
(`eval:compare` + `bakeoff:rerank`), where each arm becomes a recorded, paired-gated
A/B instead of an eyeballed sweep. Constants remain the single source in `src/memory.ts`;
any future retune lands there and re-records here.

| Constant                 | Default (pre-flip) | Tuned (operator) |
| ------------------------ | ------------------ | ---------------- |
| `RRF_K`                  | 60                 | 60 (kept; sweep deferred to wave-5/6 A/B) |
| `BLEND_RELEVANCE`        | 0.7                | 0.7 (kept)       |
| `BLEND_RECENCY`          | 0.2                | 0.2 (kept)       |
| `BLEND_IMPORTANCE`       | 0.1                | 0.1 (kept)       |
| `RECENCY_HALFLIFE_DAYS`  | 30                 | 30 (kept)        |
| reranker                 | rerank-2.5-lite    | rerank-2.5-lite (wave-6 bakeoff decides the 2.5 swap; decision record: [`../../codebase-mcp/test/bakeoff.md`](../../codebase-mcp/test/bakeoff.md#rerank-bakeoff-wave-6-ws5)) |

## Wave-7 blend/decay bakeoff — tuned-constants decision record (closes the retune TODO)

Wave 7 closes the `TODO(operator/retune)` for ALL the scoring knobs (RRF k + the blend
weights): each knob is now a measured, paired-gated A/B through
`npm run bakeoff:blend -- --yes` instead of an eyeballed sweep. Procedure: arm
selection on the DEV split → ONE predeclared two-arm confirmation of a non-A0 winner on
the frozen TEST split (a recorded flip gate — see `eval.md`) → the winner ships as
**compose env pins ONLY** on memory-mcp (code defaults remain A0; rollback = delete the
pins; the prepared pin block is commented in `compose.yaml`).

**Rename note (wave-7):** `RECENCY_HALFLIFE_DAYS` → `RECENCY_TAU_DAYS`, value 30
unchanged — deliberately NOT converted. The constant was always used as
`exp(−age/τ)`, so it is the **1/e time constant τ**, not a half-life: at age = 30 days
the recency weight is e⁻¹ ≈ 36.8% (not 50%); the true half-life is
τ·ln2 ≈ 30 × 0.6931 ≈ 20.8 days. Inserting ln 2 to make the name honest would have
CHANGED scoring behavior; renaming the constant keeps the numbers identical. The knob
now lives in `src/config.ts` (`config.blendConfig`), env-pinnable per the table in the
top-level README; `retrievalConfig()` serializes it as `blend.decay.tauDays`.

**Access-based decay: REJECTED (for now).** `getMemory` MUTATES `access_count` /
`last_accessed_at` on every read (memory.ts:508-509 at review time — the
`UPDATE … SET access_count = access_count + 1, last_accessed_at = now()` in
`getMemory`), so any access-conditioned scoring is state-dependent — an eval run would
change the very state it measures — and self-reinforcing: retrieved → boosted →
retrieved again. Revisit only with a separate shadow-logging design (log accesses out
of band; score offline against the shadow log). Recorded in the bakeoff artifact as
`rejectedForNow: ["access-based decay"]`.

### Decision table (operator fills from the live dev-split run + test confirmation)

Arms are FROZEN in `src/db/bakeoff-blend.ts` (`predeclaredArms()`): A0 control (= live
defaults), A1 per-type exp (semantic/procedural τ90; entity + decision exempt), A2
per-type power, A3 multiplicative, A4 relevance-only, plus the RRF_K axis K20/K120
(k=60 control). Artifact: `test/runs/<date>-bakeoff-blend.json` (seed 42).

**Executed 2026-07-04** (dev split, 38 rows, seed 42; A0 nDCG@10 0.6935, pool
Recall@25 0.9474 — artifact `test/runs/2026-07-04-bakeoff-blend.json`):

| Arm | mean Δ nDCG@10 vs A0 | 95% CI (seed 42) | sign W/L/T (p) | affected | by-type slice | temporal slice | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | +0.0108 | [−0.0218, +0.0436] | 9/4/25 (0.267) | 13 | sem +0.041, proc −0.031 | 0.000 (n=5) | INCONCLUSIVE |
| A2 | +0.0169 | [−0.0142, +0.0475] | 10/2/26 (0.043) | 12 | sem +0.044, proc −0.019 | 0.000 (n=5) | INCONCLUSIVE |
| A3 | −0.0143 | [−0.0551, +0.0246] | 6/8/24 (0.789) | 14 | sem +0.025, proc −0.071 | −0.026 (n=5) | INCONCLUSIVE |
| A4 | −0.0248 | [−0.0654, +0.0135] | 5/10/23 (0.302) | 15 | sem +0.008, proc −0.074 | −0.026 (n=5) | INCONCLUSIVE |
| K20 | 0.0000 | [0, 0] | 0/0/38 (1.0) | 0 | all 0 | 0 | INCONCLUSIVE |
| K120 | 0.0000 | [0, 0] | 0/0/38 (1.0) | 0 | all 0 | 0 | INCONCLUSIVE |

Winner rule (predeclared): win-side CI (ciLow > 0) AND wins > losses AND affected > 0,
highest mean delta wins; none qualifying ⇒ KEEP A0 and leave compose untouched.

**VERDICT: KEEP A0.** No arm qualified. Test-split confirmation: NOT RUN (predeclared —
only a qualifying winner spends the frozen split). Applied pins: NONE (code defaults
stay live). This closes the `TODO(operator/retune)` for the blend weights + RRF k with
a measured record. Readings worth keeping:

- **A2 (per-type power decay) is the one to re-test** when the corpus/gold grow: best
  mean delta, 10W/2L (sign p = 0.043), driven entirely by the semantic slice (+0.044)
  — but the bootstrap CI straddles zero at N=38, which is exactly the ~0.05-detectable
  regime `test/eval.md` warns about. Not p-hackable into a win today by design.
- **A4 (relevance-only) is directionally WORSE (−0.025)**: the recency+importance
  blend earns its keep vs pure rerank ordering — the blend's existence is validated.
- **The RRF-k axis is a non-lever on this corpus**: k ∈ {20, 120} produced deltas of
  exactly zero on every query — pool composition shifts never survived rerank+blend
  into the top-10. `RRF_K` stays a code constant; no env pin warranted.
- A3 (multiplicative) and A4 both hurt the temporal slice (−0.026): additive recency
  is doing real work on "current status" queries.

## Why a recall gate, not eyeball

`recallAtK`/`mrr`/`ndcgAtK`/`pairedBootstrapCI` are unit-tested pure functions
(`recall.helper.test.ts`); the flip gate is encoded as `recall-gate.test.ts` (skips until
`embedding_v2` is populated and a baseline artifact exists). A green eyeball on a few
queries can hide a corpus-wide regression, so the decision is the recorded number, not a
spot-check.

> **Wave-2 note:** the original scalar mechanism (`V1_RECALL_AT_10_BASELINE`, "v2
> recall@10 ≥ a recorded number") is **retired**. The gate now recomputes the dev split
> and compares per-query nDCG@10 against the committed `test/runs/baseline-dev.json`
> with a paired bootstrap (seed 42) — "no significant regression + directional win".
> The frozen **test** split is spent only by flip gates (AC-109). Handbook:
> `test/eval.md`.

## Wave-2 summary-backfill flip gate (token-efficiency plan, AC-812)

The stored-summary feature changes reranker docs (`title\n(summary\n)content`, capped
as a whole at `RERANK_DOC_TRUNCATION`) — but only for rows whose `summary` is non-NULL.
At merge time every corpus row is NULL, docs are byte-identical, and the CI gate stays
green by construction. The PAID backfill is therefore the flip moment; the operator
runs it as a recorded gate:

1. **Confirm the current committed baseline** (`test/runs/baseline-dev.json`
   aggregates): pool Recall@25 0.9474 / nDCG@10 0.7211 / MRR@10 0.7555 (the
   post-750d0e4 merge-point refresh — the file on disk is authoritative).
2. **Backfill the summaries** (PAID — Anthropic quota, one Haiku-class call per
   pending row; resumable, re-run to continue after a crash):
   ```
   ANTHROPIC_API_KEY=<key> DATABASE_URL=<live> \
     npm --prefix services/memory-mcp run backfill:summaries -- --yes
   ```
3. **Live gate re-run, recorded** (PAID — Voyage quota, embeds every dev query):
   ```
   EVAL_RECORD=1 DATABASE_URL=<live> VOYAGE_API_KEY=<live> npx vitest run recall-gate
   ```
   Commit the dated `test/runs/<date>-dev.json` artifact.
4. **Gate green + operator accepts** → freeze it as the new baseline and commit the
   artifact PAIR (wave-4 convention):
   ```
   cp test/runs/<date>-dev.json test/runs/baseline-dev.json
   ```
5. **Regression** → revert the wave-2 `buildRerankDoc` commit (code rollback: docs go
   back to `title\ncontent`); KEEP the paid summaries in place — they are inert once
   the doc builder ignores them, and display surfaces (`formatHits`, `memory_get`)
   remain useful.

### Recorded numbers — summary-backfill flip gate, 2026-07-04 (operator record)

- Backfill: **1,751 / 2,157** active rows summarized (Haiku via `dist/db/backfill-summaries.js
  --yes` in-container; the npm script's `tsx` entry does not exist in the prod image).
  The remaining **406 rows** skipped when the Anthropic account exhausted its credit
  mid-run — resumable, re-run after top-up. Avg content 14,072 chars (8,000 sent),
  avg summary 799 chars.
- Gate (dev split, live corpus, rerank-2.5): **PASS with a lift** —
  BEFORE (committed baseline, pre-summary): pool Recall@25 0.9474 / nDCG@10 0.7231 / MRR@10 0.7612;
  AFTER (`test/runs/2026-07-04-summary-flipgate-dev.json`): pool Recall@25 0.9474 / **nDCG@10 0.7436 (+0.0205)** /
  **MRR@10 0.7868 (+0.0256)**. Pool layer identical by construction (fuseCandidates untouched).
- Frozen: AFTER artifact copied to `baseline-dev.json` (step 4 accepted). SUPERSEDED same-day:
  wave-7 (#12/#14) re-recorded `baseline-dev.json` on the post-consolidation corpus
  (0.9474 / 0.7068 / 0.7262) — that file is authoritative; this record's AFTER artifact
  is preserved under its own filename above.
- Remainder: when the 406 pending rows are backfilled, re-run the (Voyage-only) gate
  against THIS baseline and refresh again only on a further accepted shift.
- Incident: first backfill run skipped ALL rows — compose materializes
  `${CONSOLIDATE_MODEL:-}` as an EMPTY string and `config.ts` used `??`, so the API got
  `model: ""` (400, swallowed by the summarizer's null-on-failure contract). Worked
  around with `CONSOLIDATE_MODEL=claude-haiku-4-5` in `/opt/mnemosyne/.env`; code fix
  (`||` fallback) in PR #11.

## Two-phase search core (Wave 1, retrieval-improvement program)

`searchMemory` is now composition over two exported phases in `src/memory.ts`:

- **`fuseCandidates()`** — the ONE hybrid-recall + RRF SQL path, returning the
  `config.candidatePool` (25) candidate pool with per-arm `bm25_rank`/`vec_rank`.
- **`rerankAndBlend()`** — rerank → 0.7·rel + 0.2·recency + 0.1·importance blend →
  top-limit shaping.

The eval harness scores **Recall@25 at `fuseCandidates`** (did the pool surface the gold
row at all?) and **nDCG@10 at the full pipeline** (did rerank+blend order it well?), both
through the exact code the live handler runs — never a duplicated SQL path (AC-102). A
golden-output pin (`test/search-golden.test.ts`, recorded pre-split) guards the split
itself (AC-101).

All retrieval constants now snapshot via the exported **`retrievalConfig()`**; every eval
run serializes it into its artifact (AC-104), so any recorded number in this file is
reproducible against the exact configuration that produced it. Since wave 7 the blend/
decay knobs live in `src/config.ts` (`config.blendConfig`, env-pinnable — the snapshot's
nested `blend` object + top-level `scoringVersion`); `RRF_K` stays a module const in
`src/memory.ts`. The snapshot follows automatically either way (single source).
