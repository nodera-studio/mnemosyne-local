# Operator runbooks — code graph + contextual embeddings

One place for the **deliberate, gated, paid** ops actions that the code-graph +
contextual-embeddings feature set introduced. None of these run on container boot.
Each spends Voyage API quota (money) and each must be gated on a recall/score check,
not an eyeball. Run them by hand, over loopback, against the live (or staging) DB,
after the relevant image is built.

The two per-wave decision records that these runbooks fill in:

- Memory contextual flip → `services/memory-mcp/test/retune.md`
- Code embedder bake-off → `services/codebase-mcp/test/bakeoff.md`
- Decision-log entity-link policy (ADR, no paid step) → `services/memory-mcp/test/decision-log.md`

> **The HOLD rule.** A migration whose first line begins with `-- HOLD` is intentionally
> NOT applied by `npm run migrate` (the runner skips it). Those are one-way drops gated on
> burn-in — apply them by hand only at the step called out below. Currently held:
> `memory-mcp/sql/004_drop_legacy_embedding.sql` and
> `codebase-mcp/sql/005_drop_bakeoff_scratch.sql`.

---

## Runbook A — memory-mcp contextual blue/green re-embed (Wave P)

Move durable memory recall from the legacy flat `embedding` (voyage-4-large) to
`embedding_v2` (voyage-context-4 contextual, 1024-dim int8). Memory is durable, so this is
a one-time backfill, not a scheduled re-embed. The code flip is already landed
(`src/memory.ts` reads `embedding_v2`, embeds the query contextually, and reads the tuned
constants) — it goes live at the **gated redeploy**, not before.

**Prerequisites:** image built (`docker compose up -d --build memory-mcp`),
`VOYAGE_API_KEY` set, awareness this spends Voyage quota.

1. **Apply migrations** (loopback). Applies `003_context_embeddings.sql` (ADD COLUMN
   `embedding_v2`); `004` is HOLD-skipped; `005_decision_log.sql` applies too.
   ```bash
   DATABASE_URL=<live> npm --prefix services/memory-mcp run migrate
   ```

2. **Backfill the contextual column** (deliberate, paid). Resumable — only targets active
   rows with `embedding_v2 IS NULL`, in batches of ≤500, honoring Retry-After/429.
   ```bash
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> npm --prefix services/memory-mcp run backfill:context
   ```
   Verify completeness (AC-021) — must return 0:
   ```sql
   SELECT count(*) FROM memory.memories
   WHERE archived_at IS NULL AND COALESCE(status,'active')='active' AND embedding_v2 IS NULL;
   ```

3. **Build the contextual HNSW index** CONCURRENTLY (own connection, single statement —
   cannot run in the batch runner):
   ```bash
   DATABASE_URL=<live> npm --prefix services/memory-mcp run index:v2
   ```
   **Post-re-embed hygiene (also after ANY future mass re-embed of `embedding_v2`):**
   a mass re-embed leaves dead tuples in the HNSW graph, silently degrading result
   counts; VACUUM alone is slow against a bloated index, so reindex first. Via psql
   (CONCURRENTLY needs its own connection; ~2× build time; writes keep flowing):
   ```sql
   REINDEX INDEX CONCURRENTLY mem_hnsw_v2;
   VACUUM (VERBOSE) memory.memories;
   ```
   (A `index:v2` build fresh from scratch has no dead tuples — skip the REINDEX then,
   but still VACUUM.)

4. **Freeze the eval set (wave-2 v2 gold).** Migrate the title-keyed seeds to stable-id
   v2 (`npm run gold:migrate` proposes `recall-eval.v2.json`; a human approves every id),
   expand to ≥30–50 rows from the LIVE corpus (distill/harvest proposals, human approval),
   and freeze the **test** split — spent only by flip gates (AC-109).

5. **Measure — record a baseline artifact, then gate** (the live flip-gate run; it embeds
   every dev query, so it spends Voyage quota), from the service dir:
   ```bash
   cd services/memory-mcp
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> EVAL_RECORD=1 npx vitest run recall-gate
   cp test/runs/<date>-dev.json test/runs/baseline-dev.json   # commit it
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> npx vitest run recall-gate
   ```
   The gate joins per-query nDCG@10 against `baseline-dev.json` (paired-bootstrap 95% CI)
   plus a pool Recall@25 floor. Record the BEFORE/AFTER aggregates in `retune.md`; full
   semantics in `services/memory-mcp/test/eval.md`.

6. **Re-tune the fusion constants** on the SAME held-out set BEFORE the flip (re-embed
   couples to fusion/rerank — AC-024): sweep `RRF_K` ∈ {30,40,60,80}, re-check the
   `0.7·rel + 0.2·recency + 0.1·importance` blend and `RECENCY_HALFLIFE_DAYS`, confirm
   rerank-2.5-lite still helps. Set the finalized values in `src/memory.ts`, record in
   `retune.md`.

7. **Flip gate (AC-023).** Redeploy to flip the query path to `embedding_v2` ONLY if the
   gate shows no regression vs the v1 baseline (paired-bootstrap CI on per-query nDCG@10
   plus the pool Recall@25 floor; a lift is preferred). If it regresses, do NOT flip —
   revisit the retune or the embedder, then re-measure.

8. **Burn-in, then drop the legacy column.** After `searchMemory` has served from
   `embedding_v2` with no regression and store/update have been writing both columns,
   remove the legacy `embed()` write from `memory.ts`, then apply the HELD drop:
   ```bash
   psql "$DATABASE_URL" -f services/memory-mcp/sql/004_drop_legacy_embedding.sql
   ```
   This is one-way — the legacy column was the rollback path.

**Decision-log indexes (Wave 6 — paired with this rollout):** after the decision-log
migration (`005_decision_log.sql`, applied in step 1) the active-decision partial HNSW is
built by its own op script:
```bash
DATABASE_URL=<live> npm --prefix services/memory-mcp run index:decision
```

---

## Runbook B — codebase-mcp code-embedder bake-off (Wave 5)

Decide whether to keep the incumbent **voyage-code-3** or swap code search to
**voyage-context-3**, scored on the CODE corpus at Matryoshka 1024 + int8, on Recall@10 +
MRR@10 over a labeled eval set, with the FULL pipeline (RRF + rerank). **Non-destructive:**
the contextual arm is written to a scratch column `codebase.code_chunks.embedding_ctx`;
live `code_search` keeps serving from `embedding` the whole time.

**Prerequisites:** Waves 1–4 merged, the code corpus indexed (`embedding` populated),
`VOYAGE_API_KEY` set, awareness the contextual re-embed spends Voyage quota.

1. **Freeze a real eval set.** Expand `test/fixtures/code-eval.json` from the seed toward
   representative coverage of the live corpora; a human confirms each row's `relevantPaths`.
   Set `_seed: false` and freeze.

2. **Apply the scratch migration** (additive, idempotent; live search unaffected). Applies
   `004_bakeoff_scratch.sql`; `005` is HOLD-skipped:
   ```bash
   DATABASE_URL=<live> npm --prefix services/codebase-mcp run migrate
   ```

3. **Run the bake-off** (scratch-embeds the contextual arm into `embedding_ctx`, scores
   BOTH arms). Resumable — only targets chunks with `embedding_ctx IS NULL`:
   ```bash
   DATABASE_URL=<live> VOYAGE_API_KEY=<live> \
     npm --prefix services/codebase-mcp run bakeoff -- <projectId> [repositoryId]
   ```
   Record the per-arm Recall@10 + MRR@10 in `bakeoff.md`.

4. **Pick the winner — per Recall@10** (MRR@10 breaks ties; AC-030). A negative result
   (keep voyage-code-3, the code specialist) is legitimate — do NOT force a swap.

5. **If voyage-code-3 wins:** record the negative result; drop the scratch column by
   applying the HELD `005_drop_bakeoff_scratch.sql`. No swap, no retune. Done.

6. **If voyage-context-3 wins (swap):** re-tune RRF + rerank BEFORE flipping (AC-031 — same
   coupling tax as Runbook A). Sweep the RRF constant (`60` in `search.ts` + the bake-off
   `searchArm`), re-score the candidate with `scoreArm`, gate the flip on
   tuned-candidate Recall@10 ≥ incumbent baseline. Then flip: truncate the stored vectors
   (NULL out `code_chunks.embedding`), re-embed the live `embedding` with the winner via a
   forced full reindex (`code_reindex force:true`), switch
   `indexer.ts`'s `embedCode` → `embedCodeContextual`, embed the QUERY with the winning
   embedder, set `VOYAGE_CODE_MODEL` in `config.ts` + `compose.yaml` + `.env.example`, and
   apply the retuned RRF constant.
   **Post-re-embed hygiene (after the winning-arm re-embed — and ANY future mass
   re-embed of `code_chunks.embedding`):** reindex + vacuum, same reason as Runbook A
   step 3 (dead HNSW tuples degrade result counts; VACUUM is slow without reindex first):
   ```sql
   REINDEX INDEX CONCURRENTLY chunk_hnsw;
   VACUUM (VERBOSE) codebase.code_chunks;
   ```
   Note: the indexer's chunk-level embed cache (Wave 3, AC-404) reuses stored vectors by
   `content_sha256`, model-blind — so a model flip MUST clear the stored vectors first
   (the truncate in the flip step above) before the forced reindex; a forced run against
   surviving old-model rows would cache-hit and resurrect old-model vectors.

7. **Burn-in, then drop the scratch.** After the new live `embedding` serves `code_search`
   with no regression, apply the HELD `005_drop_bakeoff_scratch.sql` and freeze the
   regression floors (`CODE_RECALL_FLOOR` / `CODE_MRR_FLOOR` in CI).

---

## Why these are gated, paid, deliberate

- **Money + quota.** Every backfill / bake-off re-embed POSTs the corpus to Voyage
  (voyage-context-4 ≈ $0.12/M tokens, first 200M free). A container start hook that
  re-embedded on every deploy would burn quota silently — so there is none.
- **Recall coupling.** Flipping an embedding column without re-tuning RRF + blend + rerank
  regresses recall silently. Both runbooks gate the flip on a recorded recall number, never
  an eyeball.
- **One-way drops.** The legacy/scratch columns are the rollback path; the `-- HOLD` marker
  keeps the migrate runner from dropping them prematurely. Apply the drop only at the
  burn-in step, by hand.
- **Memory embeddings use voyage-context-4** (released 2026-06-29; drop-in successor to
  context-3 — same endpoint/shape/1024-int8, ~33% cheaper, higher retrieval quality). A
  model swap requires re-embedding the WHOLE `embedding_v2` column (v3 and v4 vectors are
  not cross-compatible). The CODE bake-off arm below still names context-3; treat that as a
  separate, not-yet-run measurement.
