# Recall harness — the embedder-flip gate contract

This harness is how Wave P decides whether the voyage-context-3 swap is safe to flip
live. Per the locked decision (AC-023), **the flip is gated on recall against a
held-out eval set, not on eyeball inspection.** This document is that contract.

## Files

- `recall-eval.json` — the labeled set: `{ query, relevantTitles }` rows plus a top-`k`.
  Shipped as a **seed** (`"_seed": true`), NOT a finalized eval set. It uses memory
  **titles**, not ids, because ids are environment-specific and must never be invented.
- `../recall.helper.ts` —
  - `recallAtK(hits, relevant, k)` and `mrr(hits, relevant)` — pure math, unit-tested.
  - `meanRecallAtK` / `meanReciprocalRank` — corpus-level aggregates.
  - `runRecallEval(evalFile, { projectId })` — DB-backed runner that calls the live
    `searchMemory` (`../src/memory.ts`) for each query and scores ranked titles.

## The gate (Wave P flip procedure)

1. **Freeze a real held-out set.** Before the flip, an operator expands the seed in
   `recall-eval.json` to a representative held-out set drawn from the **live** corpus,
   using verbatim memory titles. The seed rows are placeholders that describe the
   shape of the corpus, not guaranteed live titles.
2. **Baseline on the incumbent embedder.** Run `runRecallEval` against the current
   `embedding` column. Record `recallAtK` + `mrr` — this is the bar to beat.
3. **Re-embed + retune (AC-024).** Backfill `embedding_v2`, then retune the RRF `k`,
   the `0.7·rel + 0.2·recency + 0.1·importance` blend, and the reranker on the SAME
   held-out set. Re-embed couples to fusion/rerank — never flip without retuning.
4. **Score the candidate embedder.** Run `runRecallEval` against `embedding_v2`.
5. **Flip only if recall does not regress.** Require candidate `recallAtK` ≥ baseline
   (no regression) and ideally `mrr` ≥ baseline. If recall regresses, do NOT flip —
   investigate the retune or the embedder, then re-score.

## Why the pure functions are unit-tested directly

`recallAtK` / `mrr` are the gate's arithmetic. A silent bug there (off-by-one rank,
wrong empty-set default) would pass a bad flip with a green-looking number. They are
therefore tested against hand-computed values in `../recall.helper.test.ts`
independent of any DB:

- relevant title at rank 1 → `recallAtK = 1`, `mrr = 1.0`
- relevant title at rank 3 → `recallAtK@3 = 1` (but `@2 = 0`), `mrr = 1/3`
- no relevant title in the hits → `recallAtK = 0`, `mrr = 0`
- empty relevant set → both `0` (conservative)

## Running it

- Pure-function tests run anywhere (CI without infra):
  `npm test` in `services/memory-mcp`.
- The DB-backed runner is guarded with `describe.skipIf(!process.env.DATABASE_URL)`,
  so it is **skipped** unless `DATABASE_URL` (loopback Postgres) and `VOYAGE_API_KEY`
  are set. When enabled it embeds each query through Voyage and hits the live search
  path, so it consumes API quota — run it deliberately, not on every CI push.
