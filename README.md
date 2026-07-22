# Mnemosyne

**Gives Claude Code durable, cross-session memory and fast semantic + AST-aware
code search — as three small local MCP servers, not a SaaS you sign up for.**
Ask it to remember a decision today, recall it correctly three weeks and forty
sessions later. Ask it "where does X live" and get an answer grounded in an actual
code graph (symbols, call edges, import edges), not a re-read of the whole repo
every time.

Runs **fully local** via Docker Compose, loopback-only (`127.0.0.1`): no account,
no data leaving your machine except embedding/rerank calls to Voyage AI. Built for
Claude Code running on the same machine (an Apple Silicon MacBook Pro/Mac Studio,
or any machine with Docker) — or reachable over your own LAN/Tailscale if you widen
the port bindings yourself.

## What you get

```mermaid
flowchart LR
    CC["Claude Code"] -->|"memory_search, memory_store, ..."| M["memory-mcp<br/>:8080"]
    CC -->|"code_search, code_graph_expand, ..."| K["codebase-mcp<br/>:8081"]
    CC -->|"secret_get, secret_list"| S["secrets-mcp<br/>:8082"]
    M --> PG[("Postgres + pgvector")]
    K --> PG
    S --> PG
    M -.->|"embed + rerank"| V["Voyage AI"]
    K -.->|"embed + rerank"| V
```

- **memory-mcp** — durable memory across sessions: hybrid semantic+lexical recall,
  an entity graph, a typed decision log with supersession tracking, contextual
  embeddings tuned via a real retrieval eval harness (not vibes).
- **codebase-mcp** — semantic + lexical code search PLUS a deterministic AST code
  graph: symbol lookup, call-graph expansion, path tracing between two symbols —
  grounded in your actual parsed source, not a re-read every query.
- **secrets-mcp** — an encrypted secret store scoped to Claude Code's own tool
  calls, isolated on its own DB schema, never touched by the search/embedding path.

## Model choices — picked by measurement, not by vendor claims

Current production pins (`compose.yaml`):

| Job | Model | Why this one |
| --- | --- | --- |
| Memory embeddings | `voyage-context-4` (contextual) | Shipped via a blue/green swap gated on a held-out recall check against the eval harness below — not an eyeball comparison. |
| Code embeddings | `voyage-code-3` | Current incumbent; an active bake-off is scoring `voyage-context-3` as a challenger on the same eval harness before any swap. |
| Reranking (both engines) | `rerank-2.5` | Won a measured bake-off against `rerank-2.5-lite`: no regression on the no-regression gate, a small positive mean nDCG delta, and cost was immaterial at this corpus scale — see `services/codebase-mcp/test/bakeoff.md`. |

The point isn't just "we use good models" — it's that every swap goes through the same
gate: score the candidate against a frozen eval split with the full pipeline (fuse →
rerank → blend), require no statistically significant regression, and only then flip
the pin. `services/memory-mcp/test/eval.md` is the full methodology if you want to run
the same bake-offs against your own corpus before trusting these picks on faith.

```mermaid
flowchart TD
    DEV["Dev split<br/>recomputed freely — CI gate,<br/>local tuning, anyone, anytime"] --> CI["CI regression gate<br/>paired-bootstrap 95% CI must exclude<br/>zero on the regression side"]
    TEST["Frozen test split<br/>spent ONLY by flip gates —<br/>embedder swaps, rerank swaps,<br/>consolidation acceptance"] --> FLIP["ONE predeclared confirmation run<br/>before any pin ships"]
    CI -->|pass| ITER["Safe to keep iterating on dev"]
    FLIP -->|pass| SHIP["Model/config swap ships as<br/>a compose env pin — code defaults<br/>unchanged, rollback = delete the pin"]
```

Splitting dev from test is the p-hacking guard: the dev split gets hammered on
constantly, so it can't be trusted for a real go/no-go call. The test split is touched
exactly once per candidate, by design — see `services/memory-mcp/test/eval.md`'s
"Gold sets: v2 shape, dev/test split policy."

## Quickstart

**Prerequisites:** Docker with Compose V2 (the `docker compose` subcommand — every
command below assumes it, not the standalone `docker-compose` binary) and a Voyage AI
API key. Node.js ≥22 is only needed if you run the operator scripts (migrations,
eval-harness, consolidation, bake-offs) directly on your host rather than through
Docker — bringing the three MCP servers up via `docker compose` doesn't need it.

```bash
git clone https://github.com/nodera-studio/mnemosyne-local.git mnemosyne
cd mnemosyne
cp .env.example .env
# Get a Voyage AI key (embeddings + rerank; free tier available): https://dashboard.voyageai.com
# Edit .env: set VOYAGE_API_KEY, then run:
make secrets                              # fills in the rest of .env (random local secrets)
docker compose up -d postgres redis
docker compose up -d --build memory-mcp codebase-mcp secrets-mcp
docker compose ps                         # wait for all three to report healthy
```

Then point Claude Code at the three servers (add to your MCP config, e.g. via
`claude mcp add` or your `~/.claude.json`):

| Server | URL |
| --- | --- |
| memory | `http://localhost:8080/mcp` |
| codebase | `http://localhost:8081/mcp` |
| secrets | `http://localhost:8082/mcp` |

Smoke-test from inside a Claude Code session: ask it to `memory_store` a test note,
then `memory_search` for it in a fresh session — if it comes back, you're live.
Index a repo with `code_reindex` before using `code_search`/`code_graph_expand`.

## Layout

```
compose.yaml            # service stack (postgres, redis, memory/codebase/secrets MCP, langfuse)
postgres/init/          # first-boot SQL (extensions + schemas)
services/memory-mcp/    # durable cross-session memory engine (Stage 1)
services/codebase-mcp/  # semantic + lexical code search + AST code graph (Stage 2)
services/secrets-mcp/   # encrypted secrets store
docs/                   # operator runbooks (see Runbooks below)
.env.example            # config template; copy to .env and fill in locally
```

## Host

Runs locally via `docker compose` — no dedicated box, no systemd unit. Bring it up from
this directory whenever you want the MCP servers available; there's nothing to deploy.

## Common commands (run locally, in this directory)

```bash
make secrets     # generate .env with strong random secrets (once)
docker compose up -d postgres redis
docker compose up -d --build memory-mcp codebase-mcp
docker compose ps
docker compose logs -f postgres
```

### Per-service migrations (manual — never auto-run on deploy)

Neither MCP service applies its SQL on container boot. Migrations are an explicit,
loopback-only operator step run AFTER the relevant image is built. The runner
(`src/db/migrate.ts` in each service) applies `sql/*.sql` in lexical order and
**skips any file whose first line begins with `-- HOLD`** (those are deliberate,
gated drops — see the runbook).

```bash
# memory-mcp — applies 001..003 + 005..007; SKIPS 004 (HOLD legacy-embedding drop)
DATABASE_URL=<loopback> npm --prefix services/memory-mcp run migrate

# codebase-mcp — applies 001..004 + 006; SKIPS 005 (HOLD bake-off-scratch drop)
DATABASE_URL=<loopback> npm --prefix services/codebase-mcp run migrate
```

Tests: the pre-existing DB suites do NOT self-migrate — before the FIRST test run
against a pristine disposable DB, run `npm run migrate` (both services) against it
(e.g. `DATABASE_URL=postgres://postgres:postgres@localhost:5544/mnemosyne`).

### Deliberate, paid ops actions (operator-run, NOT auto-on-deploy)

These spend Voyage API quota and are gated, one-time, per-rollout steps. They are
**never** invoked by a container start hook. (`distill-eval` in the eval-harness section
below is PAID too — Anthropic quota rather than Voyage — under the same operator-gated,
never-on-deploy posture.) Full step-by-step in
[`docs/Runbooks-CodeGraph-ContextEmbeddings.md`](docs/Runbooks-CodeGraph-ContextEmbeddings.md).

```bash
# memory-mcp — contextual blue/green re-embed (Wave P)
npm --prefix services/memory-mcp run backfill:context   # populate embedding_v2
npm --prefix services/memory-mcp run index:v2           # build mem_hnsw_v2 CONCURRENTLY
npm --prefix services/memory-mcp run index:decision     # build the decision partial HNSW CONCURRENTLY

# codebase-mcp — code-embedder bake-off (Wave 5: voyage-code-3 vs voyage-context-3)
npm --prefix services/codebase-mcp run bakeoff -- <projectId> [repositoryId]

# codebase-mcp — reranker bake-off (Wave 6: rerank-2.5-lite vs rerank-2.5);
# refuses without --yes and writes test/runs/<date>-bakeoff-rerank.json
DATABASE_URL=<live> VOYAGE_API_KEY=<live> \
  npm --prefix services/codebase-mcp run bakeoff:rerank -- --yes <projectId> [repositoryId]
```

### Eval-harness operator scripts (retrieval-improvement program, wave 2)

Gold-set growth is "synthetic-now, logs-forever": every script only PROPOSES candidates —
a human approves every gold id/path before it enters an eval split. Handbook:
[`services/memory-mcp/test/eval.md`](services/memory-mcp/test/eval.md).

```bash
# FREE — convert title-keyed seed gold to stable-id v2 (writes recall-eval.v2.json for review)
DATABASE_URL=<loopback> npm --prefix services/memory-mcp run gold:migrate

# PAID (Anthropic quota, Haiku-class; needs ANTHROPIC_API_KEY, optional DISTILL_MODEL
# with CONSOLIDATE_MODEL as fallback) — LLM-distill eval-query candidates from the
# corpus into test/fixtures/eval-candidates-<date>.json; refuses without --yes
DATABASE_URL=<loopback> npm --prefix services/memory-mcp   run distill-eval -- --yes
DATABASE_URL=<loopback> npm --prefix services/codebase-mcp run distill-eval -- --yes

# PAID (Anthropic quota, Haiku-class; needs ANTHROPIC_API_KEY, optional
# CONSOLIDATE_MODEL) — resumable backfill of memory.memories.summary for existing
# rows; refuses without --yes. FLIP GATE: rerank docs change for summarized rows —
# follow services/memory-mcp/test/retune.md's wave-2 runbook (live gate re-record).
# Write-time summaries stay off unless SUMMARIZE_ON_STORE=1 is ALSO set in .env
# (double gate: key AND flag — the key alone never spends on memory_store).
DATABASE_URL=<live> ANTHROPIC_API_KEY=<key> \
  npm --prefix services/memory-mcp run backfill:summaries -- --yes

# FREE — monthly gold rotation from real memory.search_log usage (frequent/zero-hit/
# low-overlap queries not already in gold); flags: --days 30 --min-count 2 --limit 40.
# Pair it with the log's retention cleanup (append-only, grows unbounded otherwise):
#   DELETE FROM memory.search_log WHERE created_at < now() - interval '90 days';
DATABASE_URL=<loopback> npm --prefix services/memory-mcp run harvest-eval

# PAID (Voyage quota — embeds every dev query) — the flip-gate run: record a run
# artifact, freeze it as the committed baseline, then the CI gate does the statistics
DATABASE_URL=<live> VOYAGE_API_KEY=<live> EVAL_RECORD=1 \
  npm --prefix services/memory-mcp run test -- recall-gate
```

### Consolidation operator scripts (retrieval-improvement program, wave 5)

Memory consolidation is ADD-only and reversible: losers are marked
`status='superseded'` + `superseded_by=<winner>` — content is never edited or
deleted, pinned rows never lose, decision rows never enter. Runbook + decision
record: [`services/memory-mcp/test/consolidation.md`](services/memory-mcp/test/consolidation.md).

```mermaid
flowchart LR
    A["consolidate -- --yes<br/>(DRY-RUN default)"] --> B["LLM judge scores candidate pairs<br/>(assertion-equivalence, not similarity)"]
    B --> C["Report artifact:<br/>test/runs/consolidate-{date}.json<br/>(review surface + undo log)"]
    C --> D{{"Human reviews the<br/>same-day report"}}
    D -->|approve| E["consolidate -- --yes --apply<br/>losers marked superseded<br/>(status flip only — never deleted)"]
    D -->|reject/ignore| F["No changes — nothing expires,<br/>re-run later"]
```

```bash
# PAID (Anthropic quota — the assertion-equivalence judge; needs ANTHROPIC_API_KEY,
# optional CONSOLIDATE_MODEL) — DRY-RUN by default: writes the report artifact
# test/runs/consolidate-<date>.json (review surface + undo log + resumability
# checkpoint), changes ZERO rows. Refuses without --yes.
DATABASE_URL=<live> npm --prefix services/memory-mcp run consolidate -- --yes

# FREE — applies the reviewed same-day dry-run report (never re-judges; refuses
# when no complete same-day report exists — pass --report <path> across a UTC
# day boundary). No CLI path applies un-reviewed verdicts.
DATABASE_URL=<live> npm --prefix services/memory-mcp run consolidate -- --yes --apply

# PAID (Voyage quota — embeds every query of the chosen split) — record a TEST-split
# run artifact at a flip gate (pre/post-consolidation A/B); refuses without --yes
DATABASE_URL=<live> npm --prefix services/memory-mcp run eval:record -- --yes --label pre-consolidation

# FREE — paired comparison of two run artifacts (mean nDCG delta + bootstrap 95% CI)
npm --prefix services/memory-mcp run eval:compare -- test/runs/<A>.json test/runs/<B>.json
```

### Blend/decay bake-off (retrieval-improvement program, wave 7)

Blend arms are scored OFFLINE from one captured fuse+rerank pass per query (AC-703), so
the Voyage spend is one embed per query plus one rerank per query per RRF-k value —
never per arm. Defaults to the dev split; the frozen test split is spent ONCE, on the
predeclared two-arm confirmation of the winner. Decision record + pin procedure:
[`services/memory-mcp/test/retune.md`](services/memory-mcp/test/retune.md).

```bash
# PAID (Voyage quota) — 5 predeclared blend arms (A0 control, per-type exp/power decay,
# multiplicative, relevance-only) + the RRF_K axis (K20/K120; k=60 is the control);
# refuses without --yes and writes test/runs/<date>-bakeoff-blend.json (seed 42)
DATABASE_URL=<live> VOYAGE_API_KEY=<live> \
  npm --prefix services/memory-mcp run bakeoff:blend -- --yes

# The ONE sanctioned frozen-test-split confirmation of a non-A0 winner (a flip gate):
DATABASE_URL=<live> VOYAGE_API_KEY=<live> \
  npm --prefix services/memory-mcp run bakeoff:blend -- --yes --split test --arms A0,<winner>
```

The winner ships as **compose env pins only** on memory-mcp — code defaults remain A0,
so rollback = delete the pins (see the commented block in `compose.yaml`):

| Env key (memory-mcp) | Default | Meaning |
| --- | --- | --- |
| `BLEND_FORM` | `additive` | `additive` (legacy weighted sum) or `multiplicative` (`rel × (1 + w·recency + w·importance)`) |
| `BLEND_W_RELEVANCE` / `BLEND_W_RECENCY` / `BLEND_W_IMPORTANCE` | `0.7` / `0.2` / `0.1` | blend weights |
| `DECAY_SHAPE` | `exp` | `exp` = e^(−age/τ); `power` = 1/(1 + age/τ)^exponent |
| `RECENCY_TAU_DAYS` | `30` | τ, the 1/e time constant (NOT the half-life — that is τ·ln2 ≈ 20.8 days at τ=30); renamed from `RECENCY_HALFLIFE_DAYS` in wave 7, value unchanged |
| `DECAY_POWER_EXPONENT` | `0.5` | exponent for the `power` shape |
| `RECENCY_TAU_DAYS_BY_TYPE` | `{}` | JSON map of per-type τ overrides, e.g. `{"semantic":90,"procedural":90}` |
| `DECAY_EXEMPT` | (empty) | CSV of `type:<t>` / `source_kind:<k>` rows exempt from decay (recency = 1.0), e.g. `type:entity,source_kind:decision` |

## MCP tools

Both `memory_search` and `code_search` share the same two-layer retrieval shape — the
split that lets the eval harness tell "the embedder missed it" apart from "the reranker
ordered it badly," which one aggregate number can't:

```mermaid
flowchart LR
    Q["memory_search / code_search"] --> FUSE["fuseCandidates()<br/>hybrid RRF: semantic (vector) + lexical (BM25/ts_rank)<br/>scored as pool Recall@25"]
    FUSE --> RERANK["rerankAndBlend() / rerankCodeHits()<br/>rerank-2.5 + recency/importance blend<br/>scored as nDCG@10 + MRR"]
    RERANK --> HITS["Rank-only hits back to Claude<br/>(no raw scores exposed — token-efficient)"]
```

### memory-mcp (durable memory)

`memory_search` · `memory_store` · `memory_get` · `memory_get_recent` ·
`memory_update` · `memory_list` · `memory_delete` · `memory_get_entity` ·
`memory_decision_chain`

- **`memory_search`** returns rank-only compact hits with
  `createdAt`, `eventDate`, `effectiveDate` (`eventDate ?? createdAt`), and `status`.
  Each hit's `snippet` is query-aware: a `ts_headline` fragment when the hit matched
  lexically (detected via private-use sentinel selectors, never naive `**` scanning —
  literal `**` in stored markdown must not false-positive), else the deterministic
  180-char prefix. `summary` is an optional 1–3 sentence dense summary (NULL until
  populated by the opt-in write-time summarizer or the `backfill:summaries` operator
  script) — see [`services/memory-mcp/test/eval.md`](services/memory-mcp/test/eval.md)
  for the full contract. Optional refinements: `type`, `tags` (ANY-of metadata tags), and
  `after` (ISO timestamp compared to `COALESCE(event_date, created_at)`). If `tags`/`after`
  match no candidates, the tool retries once without those optional filters and prepends
  the retry notice.
- **`memory_get_recent`** returns the same public hit fields as `memory_search` and keeps
  newest-created ordering (`created_at DESC`).
- **`memory_get`** includes `status`, `event_date`, and `superseded_by` in its JSON. It
  prepends `superseded — see <successor id>` for superseded rows with a successor, or
  `status=<status>` for other non-active rows; the row is still returned. Content is
  budgeted by `maxChars` (default 6000 chars ≈ 1500 tokens; max 200000): oversized
  content is cut with explicit `truncated: true`, `totalChars`, and a re-fetch note.
  Pass `full=true` (or `maxChars=0`) for the complete body; `metadata` is never budgeted.
- **`memory_store`** now accepts a typed **decision-log** shape (only meaningful when
  `sourceKind='decision'`): `decisionProject`, `decisionStatus`
  (`active`|`superseded`|`deferred`), `decidedAt`, `supersedesId` (the prior decision
  THIS one replaces), `decidedIn`, `relatedIds[]`. Non-decision rows leave these NULL.
- **`memory_decision_chain`** (new) — walks the supersession chain backward from a
  decision via `supersedes_id` (this decision → the one it superseded → …) using a
  recursive CTE, returning the ordered lineage with `decision_status` + `decided_at`.

### codebase-mcp (code search + AST graph)

`code_search` · `code_get_file` · `code_index_status` · `code_reindex` ·
`code_graph_expand` · `code_symbol_lookup` · `code_trace_path`

Four different ways to query the SAME deterministic AST graph, not four separate
indexes:

```mermaid
flowchart TD
    IDX["Indexer (tree-sitter)<br/>extracts symbols + call/import edges<br/>per file, on every reindex"] --> GRAPH[("codebase.symbols +<br/>codebase.symbol_edges")]
    GRAPH --> CS["code_search<br/>semantic + lexical hit list"]
    GRAPH --> CGE["code_graph_expand<br/>callers/callees from a seed,<br/>depth-capped recursive traversal"]
    GRAPH --> CSL["code_symbol_lookup<br/>resolve a name to its definition(s)"]
    GRAPH --> CTP["code_trace_path<br/>follow call edges downstream —<br/>route → handler → service → repo"]
```

- **`code_search`** returns rank-only hits. Optional refinements: `repo`, `path`
  (substring of repo-relative path), `extension` (with or without leading dot), and
  `language`. If `path`/`extension` match no candidates, the tool retries once without
  those optional filters while preserving `repo`/`language`, and prepends the retry
  notice. When top-k contains adjacent/overlapping chunks of the same file, results merge
  into a single line span capped by `MAX_MERGED_LINES` (default 120), annotated as
  `(spans N chunks)`; merged responses are not backfilled.
- **`code_graph_expand`** (new) — given a symbol name, a `file`(+`line`), or a
  `code_search` `chunkId`, returns its definition plus callers/callees from the AST call
  graph as structured rows `{name, file:line, depth, edge_type}`. `direction`
  (`callees`|`callers`|`both`, default `both`), `depth` (default 4, hard cap 10), `limit`
  (default 50). Falls back to a file read when the graph has no edges for the seed
  (un-graphed repo / non-TS file / unknown name). Name-matched (ambiguous) hops are
  labeled "(name-matched, may be ambiguous)".
- **`code_symbol_lookup`** (new) — resolve a symbol name to its definition(s)
  `{name, kind, file:line}`; multiple matches are returned with an ambiguity note.
- **`code_trace_path`** (new) — follow call edges downstream from a seed and return the
  ordered chain `{name, file:line, edge_type, depth}` (route → handler → service → repo).
  Pass `to` to stop at a target name. Edges are by-name, so this returns A resolved path,
  not a proven-unique one.
- **`code_index_status`** (enriched) — in addition to per-repo file/chunk counts and last
  index time, it now surfaces the latest `codebase.index_runs` row: `phase` (terminal
  `done`/`error`), files done/total, chunks total, current file, error text, and
  start/finish times — so a failed detached index is no longer invisible.
- **`code_reindex`** (enriched) — `force:true` bypasses the `content_sha256` skip so an
  ALREADY-indexed corpus gets its code graph built on first rollout. Forced runs are
  graph-only for unchanged files (no re-embed, so no Voyage quota burn).

## Code graph + contextual embeddings

Capabilities landed on the Stage 1/2 engines:

**Deterministic AST code graph (codebase-mcp).** Indexing now extracts symbols and call /
import edges with tree-sitter (TypeScript) into `codebase.symbols` + `codebase.symbol_edges`
(a hardened schema with `project_id`, a `file_id` FK, and a uniqueness key — migration
`003_graph_constraints.sql`). Edges are resolved by name in a post-walk phase under a
per-repo advisory lock; name-ambiguous edges are labeled, not hidden (scip-typescript was
deferred for the Node 18/20 vs node:22 runtime mismatch). Re-indexing a file replaces its
symbols/edges via a `file_id`-keyed delete+reinsert. The `index_runs` lifecycle table
(`002_index_jobs.sql`) makes the detached indexer observable through `code_index_status`.
The native tree-sitter binding is a runtime dependency that survives the build→runtime
Docker copy (AC-012). The three graph tools above query a recursive-CTE traversal with a
depth cap, a limit, and a path-array cycle guard, and fall back to a file read when the
graph is empty.

**Contextual embeddings (memory-mcp).** Memory embeddings move from the legacy flat
`embedding` (voyage-4-large) to `embedding_v2` (voyage-context-4 contextual, 1024-dim int8
`halfvec`) via a **blue/green** swap: a parallel column is added (`003_context_embeddings.sql`),
backfilled out of band (`backfill:context`), indexed CONCURRENTLY (`index:v2`), then the
query path flips — **gated on a recall check** against a held-out eval set, never an eyeball.
The RRF `k`, the `0.7·rel + 0.2·recency + 0.1·importance` blend, and the reranker are
re-tuned BEFORE the flip (re-embed couples to fusion/rerank). The legacy column is dropped
only after burn-in (`004_drop_legacy_embedding.sql`, HOLD). A code-embedder **bake-off**
(Wave 5) scores voyage-code-3 vs voyage-context-3 on the code corpus into a scratch column
(`004_bakeoff_scratch.sql`) without touching live search, and picks per Recall@10.

```mermaid
flowchart TD
    A["Add parallel column<br/>embedding_v2 (003_context_embeddings.sql)<br/>— existing search untouched"] --> B["Backfill out of band<br/>(backfill:context)"]
    B --> C["Index CONCURRENTLY<br/>(index:v2)"]
    C --> D{{"Gate: recall check against a<br/>held-out eval set — never an eyeball"}}
    D -->|regresses| RETUNE["Retune RRF k / blend weights /<br/>reranker first — re-embed<br/>couples to fusion+rerank"]
    RETUNE --> D
    D -->|no regression| FLIP["Flip the query path to embedding_v2"]
    FLIP --> BURN["Burn-in period<br/>(both columns still written —<br/>rollback stays possible)"]
    BURN --> DROP["Drop the legacy column<br/>(004_drop_legacy_embedding.sql,<br/>held until burn-in ends)"]
```

**Decision-log metadata (memory-mcp).** Decisions are structured columns on
`memory.memories` (`005_decision_log.sql`) — a sparse typed shape for `source_kind='decision'`
rows, NOT an LLM-extracted temporal KG. Supersession walks `supersedes_id` (backward) via a
recursive CTE; the live `entities`/`entity_edges` graph is never mutated by a decision write.

**Retrieval eval harness (both services — retrieval-improvement waves 1–3).** Search is
split into two exported phases (`fuseCandidates`/`rerankAndBlend` in memory;
`fuseCodeCandidates`/`rerankCodeHits` in codebase) so evals score the exact live code at
two layers: pool Recall@25 at the fuse phase, nDCG@10 + MRR at the full pipeline. Gold
sets are stable-id v2 with **dev/test splits** (the test split is spent only by flip
gates); the CI regression gate recomputes the dev split and joins per-query nDCG@10
against the committed `test/runs/baseline-dev.json`, failing on a paired-bootstrap 95%
CI that excludes zero on the regression side. Run artifacts (with a verbatim
`retrievalConfig()` snapshot) are written only under `EVAL_RECORD=1`. Every live
`memory_search` is logged fire-and-forget to `memory.search_log` (006) for the monthly
gold harvest. Supporting perf work: covering edge indexes (codebase 006) + app-side BFS
for graph depth ≥ 5 (the recursive CTE, with `SET LOCAL work_mem`, serves depth ≤ 4),
staged edge resolution, and a chunk-level embed cache keyed on `content_sha256`.
Handbook: [`services/memory-mcp/test/eval.md`](services/memory-mcp/test/eval.md).

### Decision records (per-wave, fill from the live run)

- Code bake-off procedure + recorded result → [`services/codebase-mcp/test/bakeoff.md`](services/codebase-mcp/test/bakeoff.md)
- Memory retune / flip-gate decision record → [`services/memory-mcp/test/retune.md`](services/memory-mcp/test/retune.md)
- Decision-log entity-link policy (ADR) → [`services/memory-mcp/test/decision-log.md`](services/memory-mcp/test/decision-log.md)

## Migration inventory

| Service | File | Wave | Applied by runner? |
| --- | --- | --- | --- |
| codebase-mcp | `001_codebase.sql` | base | yes |
| codebase-mcp | `002_index_jobs.sql` | 1 | yes |
| codebase-mcp | `003_graph_constraints.sql` | 2 | yes |
| codebase-mcp | `004_bakeoff_scratch.sql` | 5 | yes |
| codebase-mcp | `005_drop_bakeoff_scratch.sql` | 5 | **HOLD** — skipped (apply post-decision) |
| codebase-mcp | `006_edge_covering_indexes.sql` | retrieval wave 3 | yes |
| memory-mcp | `001_memory.sql` | base | yes |
| memory-mcp | `002_migration.sql` | base | yes |
| memory-mcp | `003_context_embeddings.sql` | P | yes |
| memory-mcp | `004_drop_legacy_embedding.sql` | P | **HOLD** — skipped (apply post burn-in) |
| memory-mcp | `005_decision_log.sql` | 6 | yes |
| memory-mcp | `006_search_log.sql` | retrieval wave 2 | yes |
| memory-mcp | `007_summary.sql` | retrieval token-efficiency | yes |

The two `CREATE INDEX CONCURRENTLY` indexes (`mem_hnsw_v2`, the decision partial HNSW)
cannot run inside the batch runner's transaction, so they are built by standalone op
scripts: `npm run index:v2` and `npm run index:decision`. Codebase `006` builds its
covering edge indexes plain (not CONCURRENTLY — deliberate, the table is small) and
drops the old single-column `symedge_from`/`symedge_to`, which `001` no longer creates.

## Status

Memory and codebase engines are complete and in daily use: durable hybrid recall,
an entity graph, a code graph with symbol/call-graph tools, and a retrieval eval
harness that gates embedder/rerank/blend changes against measured regressions
rather than eyeballing search results. Optimization work continues — the eval
harness (`services/memory-mcp/test/eval.md`) is the live source of truth for
what's tuned and what's next.
