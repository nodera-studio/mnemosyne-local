# Retrieval eval harness handbook (wave-2, retrieval-improvement program)

The single reference for how retrieval quality is measured, gated, and grown in BOTH
services (memory-mcp here; codebase-mcp mirrors every piece — see the table at the end).
Program plan: `.claude/plans/2026-07-03-retrieval-improvement-program/` (AC-101…AC-109).

## Two layers, and why

Every eval run scores the SAME live code at two points (AC-104):

| Layer | Code under test | Metric | Question answered |
| --- | --- | --- | --- |
| POOL | `fuseCandidates()` (the one RRF SQL path) | **Recall@25** | Did hybrid recall surface the gold row into the candidate pool at all? |
| FINAL | full `searchMemory()` (fuse → rerank → blend) | **nDCG@10** + MRR | Did rerank+blend order the gold row well? |

The split matters because **the reranker cannot fix recall**: a gold row missing from
the 25-candidate pool is unrecoverable downstream, no matter how good the reranker is.
A drop in pool Recall@25 points at the embedder/fusion; a drop in nDCG@10 with a stable
pool points at rerank/blend. One aggregate number cannot tell those apart.

Runner: `runRecallEval()` in `test/recall.helper.ts` — id-keyed v2 gold, and gold ids
are resolved through their `superseded_by` FORWARD chains before scoring (AC-106), so
consolidation marking a gold memory superseded can never orphan the gold row: a hit on
any successor counts. Wave-5's `npm run consolidate` produces exactly these forward
pointers (loser → winner, chains permitted, never destructive) — gold referencing a
consolidated loser keeps scoring through the winner without any gold edit.

nDCG is scored **one-credit**: each gold key gains at its FIRST occurrence in the top-k
only (crediting duplicates lets DCG exceed the IDCG anchor — nDCG > 1 — silently biasing
the gate), matching the first-occurrence semantics of recall/MRR. The code runner
additionally dedupes chunk-keyed hit paths to their first occurrence before scoring (one
file's chunks can fill several top-10 slots; a duplicate is not a second relevant
document). Numbers recorded before the one-credit fix are NOT comparable to post-fix
numbers — re-record the baseline rather than joining across the change.

## Gold sets: v2 shape, dev/test split policy (AC-109)

`test/fixtures/recall-eval.json` (memory) and `test/fixtures/code-eval.json` (code).
v2 rows: `{ id, query, relevantIds|relevantPaths, archetype?, split, provenance,
approvedBy? }` with a `{version: 2, k, changelog: []}` header. Stable row ids
(`m-001`… / `c-001`…) are the join key for the gate.

- **dev split** — recomputed freely: by the CI gate, by local tuning runs, by anyone.
- **test split** — spent EXCLUSIVELY by flip gates (embedder flips, rerank swaps,
  consolidation acceptance, the wave-7 blend-winner confirmation). Never used for
  tuning, never recomputed casually — that is the p-hacking guard. Every change to the
  test split is recorded in the file's `changelog` header. **Consolidation acceptance
  (AC-304) IS a flip gate**: the pre/post test-split runs of `test/consolidation.md`
  are sanctioned test-split spend; record them via `npm run eval:record -- --yes` and
  compare with `npm run eval:compare`. **The wave-7 blend bakeoff confirmation IS a
  flip gate too (AC-705)**: arm SELECTION runs on the dev split
  (`npm run bakeoff:blend -- --yes`); a non-A0 winner is then confirmed by exactly ONE
  predeclared two-arm run on the frozen test split
  (`… -- --yes --split test --arms A0,<winner>`) BEFORE any compose pin ships. One
  combined confirmation covers BOTH sweep axes (blend arms + RRF k) — never one
  test-split run per axis, and never iterate on the test split. Record the spend in
  this file's changelog and the verdict in `retune.md`.

Some gold rows carry `facet: "temporal"` (recency-sensitive queries, m-051..m-058).
The wave-7 bakeoff slices per-arm deltas over this facet separately — decay-shape arms
are DISCRIMINABLE only when enough temporal rows exist, so the operator pre-checks
≥ 5 temporal rows in the target split before a bakeoff run. The v2 loader tolerates
the extra key everywhere else; scoring is unchanged.

Nothing enters either split without a human approving every gold id/path and recording
`approvedBy` (GRAFT 2). The scripts below only PROPOSE.

## Run artifacts + the baseline convention (AC-104)

**Recording runs** (`EVAL_RECORD=1` on the gate test) write
`test/runs/<ISO-date>-<label>.json` via `writeRunArtifact()`: the full
`retrievalConfig()` snapshot (verbatim — so every number is reproducible against the
exact configuration that produced it), per-query scores for both layers, and the
aggregates. Ordinary gate runs recompute in memory and write nothing — a failing gate
reports the per-query-delta statistics (mean + 95% CI) in its assertion message; test
suites write artifacts to a tmp dir. Artifacts under `test/runs/` are **committed** —
they are the baselines.

`test/runs/baseline-dev.json` is THE committed baseline the CI gate joins against: a
copy of one artifact, chosen by the operator at a flip gate and refreshed **only** at
flip gates. Record one with `EVAL_RECORD=1` on the gate test (below), then
`cp test/runs/<date>-dev.json test/runs/baseline-dev.json` and commit.

## The CI regression gate (AC-105)

`test/recall-gate.test.ts` (memory) / `test/code-recall-gate.test.ts` (code):

1. Recompute the **dev split** through the live two-layer pipeline.
2. Join per-query nDCG@10 against `baseline-dev.json` **by row id** — id drift fails
   loudly (a changed dev split invalidates the baseline).
3. `deltas = fresh − baseline` → `pairedBootstrapCI(deltas, {seed: 42})` (10k
   percentile bootstrap, deterministic). **FAIL** when the 95% CI excludes zero on the
   regression side (`ciHigh < 0`).
4. Coarse pool floor: **FAIL** when mean Recall@25 drops > 0.05 absolute vs baseline.

Semantics: **"no significant regression + directional win"** — at N≈75 rows/service the
bootstrap only detects nDCG deltas ≥ ~0.05; smaller movements are declared noise **by
design**. Do not chase micro-deltas on the dev split; that is how the dev split gets
p-hacked into meaninglessness.

Graceful-skip ladder (CI always lands on a rung; the gate never false-fails): no
`DATABASE_URL` → no `baseline-dev.json` → gold file not yet v2 → corpus not populated
(`embedding_v2` for memory / no indexed chunks for code). The operator runs the SAME
test against the live DB — that IS the flip-gate run (it embeds every dev query, so it
burns Voyage quota there; CI never reaches that rung).

The gate machinery itself is proven fail-able by the `*-gate-sim.test.ts`
test-of-the-test suites (synthetic baseline 0.2 above reality → FAIL; equal → PASS).

### Baseline lifetime — hot-corpus drift (measured 2026-07-04)

On a live memory corpus the committed baseline EXPIRES: one day after the G0 baseline,
the gate failed at mean nDCG@10 delta −0.019 (CI [−0.041, −0.005]) with **zero** pool-
recall change and zero gaining rows — new memories written that day (about the very
program topics the gold covers) legitimately outranked yesterday's gold in the top-10.
The control run proved it: the BASELINE-ERA code re-run on the same-day corpus failed
the gate identically (−0.018), so the regression was corpus drift, not code.

Operational rules that follow:
1. A live-gate FAIL is triaged with the **same-corpus A/B control**: re-run the gate
   from the baseline-era code (the commit `baseline-dev.json` was recorded at) against
   today's corpus. Code is guilty only for the code-vs-control difference.
2. Baselines are REFRESHED at every merge/flip gate (record on the merge-point corpus,
   commit alongside the dated artifact) — the gate protects a wave's review cycle, not
   weeks of corpus evolution. The frozen **test** split is unaffected by refreshes.
3. Rot-prone gold (temporal "current status" rows) lives in the dev split BY POLICY
   (see the gold changelog) so it can be re-pointed when the current state moves.

## Response shaping (WS2)

Tool responses are deliberately shaped for token efficiency, not raw scorer inspection
(research: `.claude/research/2026-07-03-retrieval-improvement-scaling-token-efficiency.md`).
Public search hits are rank-only: callers see rank, identifiers, dates, status, and
snippets, but not raw rerank/blend scores. Internal scored hits remain available inside
`rerankAndBlend()` / `rerankCodeHits()` for harness composition and scoring.

Filters are plain SQL pre-filters in BOTH RRF arms before fusion:

| Tool | Optional filters | Semantics |
| --- | --- | --- |
| `memory_search` | `tags`, `after` | `tags` is ANY-of metadata tag matching; `after` compares `COALESCE(event_date, created_at)` |
| `code_search` | `path`, `extension` | `path` is a substring of the repo-relative path; `extension` matches the file suffix |

If any optional refinement filter yields an empty candidate pool, the tool retries once
without those optional filters and prepends the retry notice. Existing scoping filters
(`type` for memory; `repo`/`language` for code) are preserved on the retry. Truncated
responses append a steering line that recommends smaller targeted searches and names the
available filters as optional refinements. Code hits additionally merge adjacent or
overlapping same-file chunks after top-k reranking, capped by `MAX_MERGED_LINES`
(default 120); k is a cap, so merged responses are not backfilled.

Memory snippets are query-aware since wave 1 of the token-efficiency plan (AC-801..804):
`searchMemory` runs one extra `ts_headline('english', …, plainto_tsquery)` query over
ONLY the final ≤limit hit ids (options in `SNIPPET_HEADLINE_OPTS`: private-use sentinel
selectors `U+E000`/`U+E001`, rendered as markdown-safe `**` for display, ≤2 fragments
joined by `" … "`), so a hit the query matched lexically shows the matching fragments
instead of the document head. The deterministic 180-char whitespace-collapsed prefix
remains the fallback for every hit whose cleaned headline carries no match sentinel
(vector-only hits, stop-words-only queries — literal `**` in stored markdown
deliberately does NOT count as a match) and for ANY headline failure — and stays the
only snippet form `memory_get_recent` produces. This is
display-only: `fuseCandidates` SQL and `rerankAndBlend` scoring are untouched, pool
composition and hit order are provably unchanged (the golden ORDERED-ID pins did not
move), and the headline options are deliberately NOT part of `retrievalConfig()`.
Headline pins in `test/search-golden.test.ts` are recorded on pg17 — the live stack's
Postgres major — and only need re-recording on a major bump.

Since wave 2 of the token-efficiency plan (AC-807..812), `memory.memories.summary`
stores an optional 1–3 sentence dense summary (NULL until populated). Hits and
`memory_get` carry it, `formatHits` renders a `summary:` line when present, and the
reranker doc becomes `title\n(summary\n)content` truncated as a whole to
`RERANK_DOC_TRUNCATION` (`retrievalConfig().rerankDocIncludesSummary`). **The summary
BACKFILL is a FLIP GATE:** rerank docs change for summarized rows, so the corpus-wide
`npm run backfill:summaries -- --yes` run must be followed by the live recall-gate
re-record per `test/retune.md`'s wave-2 runbook (pre-backfill, all summaries are NULL
and docs are byte-identical — the gate stays green at merge by construction). Write-time
summaries stay OFF by default: the double gate is `ANTHROPIC_API_KEY` present AND
`SUMMARIZE_ON_STORE=1`.

## Gold lifecycle: synthetic-now, logs-forever

1. **Seed** (`provenance: "seed-v1"`) — the hand-written rows, migrated to stable ids
   via `npm run gold:migrate` (free; proposes `recall-eval.v2.json` for review).
2. **Distill** (`provenance: "distilled"`) — **PAID**: `npm run distill-eval -- --yes`
   samples the corpus stratified (type/source_kind for memory; language/dir for code),
   asks a Haiku-class model for query candidates per archetype, and writes
   `test/fixtures/eval-candidates-<date>.json` with `approved: false` on every row.
   Human verifies each suggested gold id/path, then merges into the eval file.
3. **Harvest** (`provenance: "log-harvest"`) — free, monthly: `npm run harvest-eval`
   aggregates `memory.search_log` (written fire-and-forget by every live search,
   AC-107) and proposes frequent / zero-hit / low-overlap queries not already in gold,
   with the most-common final ids as hints. Same approval flow. Cleanup note lives in
   its header: `DELETE FROM memory.search_log WHERE created_at < now() - interval '90 days';`

## PAID script inventory (AC-108)

All main-guarded (`import.meta.url`), all `npm run`-gated, none imported by server.ts,
none in CI:

| Script | Service | Cost | Consent |
| --- | --- | --- | --- |
| `npm run distill-eval -- --yes` | both | Anthropic quota (Haiku-class; `ANTHROPIC_API_KEY`, `DISTILL_MODEL` with `CONSOLIDATE_MODEL` fallback) | refuses without `--yes` |
| `npm run backfill:summaries -- --yes` | memory | Anthropic quota (Haiku-class, one call per `summary IS NULL` row; ≈ min(content,8000)/4 + 256 tokens each) | refuses without `--yes`; the run is a FLIP GATE (see retune.md) |
| `npm run backfill:context` | memory | Voyage quota ($0.12/M) | deliberate operator step |
| `npm run bakeoff` | codebase | Voyage quota | deliberate operator step |
| `npm run consolidate -- --yes [--apply]` | memory | Anthropic quota (judge; `CONSOLIDATE_MODEL`, ≈500 tok/pair) — dry-run by DEFAULT, `--apply` flips losers | refuses without `--yes` |
| `npm run eval:record -- --yes [--split test\|dev]` | memory | Voyage quota (embeds every split query — the flip-gate recorder) | refuses without `--yes` |
| `npm run bakeoff:blend -- --yes [--split dev\|test] [--arms A0,…]` | memory | Voyage quota (one embed/query reused across k + one rerank/query per RRF-k; blend arms score OFFLINE from the capture, AC-703) | refuses without `--yes` |
| `npm run eval:compare -- <A> <B>` | memory | free (reads two artifacts) | — |
| `npm run gold:migrate` | memory | free (SQL only) | — |
| `npm run harvest-eval` | memory | free (SQL only) | — |
| gate tests on the LIVE DB | both | Voyage quota (embeds every dev query) | operator-run flip gate |

Tests never call live Voyage/Anthropic — every paid dependency is injectable and mocked.

## codebase-mcp mirror map

| memory-mcp | codebase-mcp |
| --- | --- |
| `test/recall.helper.ts` (runner + math re-exports) | `test/code-eval.helper.ts` → `src/recall-math.ts` |
| `test/recall-gate.test.ts` | `test/code-recall-gate.test.ts` |
| `test/recall-gate-sim.test.ts` | `test/code-recall-gate-sim.test.ts` |
| `test/fixtures/recall-eval.json` (memory ids + supersession resolution) | `test/fixtures/code-eval.json` (file paths — no supersession; paths do not supersede) |
| `test/runs/baseline-dev.json` | same convention |
| `src/db/distill-eval.ts` | `src/db/distill-eval.ts` |
| `src/db/harvest-eval.ts` (search_log) | — (no code search log yet) |
