# Decision-log metadata — entity-link policy (Wave 6, AC-042)

ADR-style note. Companion to `sql/005_decision_log.sql`, `src/memory.ts`
(`storeMemory` decision fields + `decisionChain`), and the `memory_store` /
`memory_decision_chain` tools in `src/server.ts`.

## Decision

Decisions are recorded as **structured decision-log metadata** on
`memory.memories` (a sparse typed shape for `source_kind='decision'` rows) —
**NOT** as an LLM-extracted temporal knowledge graph. This implements the locked
decision in memory `19643000` ("structured decision-log metadata … NOT an
LLM-extracted graph") and the deferral of the memories temporal KG.

## How decisions relate (columns are canonical)

A decision relates to other memories **via columns only**:

- `supersedes_id uuid` (self-FK → `memory.memories.id`) — the **backward** chain
  link. Canonical direction: *this decision replaces that one*. The recursive CTE
  in `decisionChain()` walks this column.
- `superseded_by uuid` (pre-existing, `002_migration.sql:9`) — the **forward**
  inverse, kept in sync by `storeMemory` on supersession for back-compat. Not the
  CTE's walk axis.
- `related_ids uuid[]` — non-supersession sibling relations (GIN-indexed).

The decision lifecycle lives on `decision_status` (`active|superseded|deferred`),
a **separate axis** from the memory-lifecycle `status` column
(`active|superseded|archived|closed`, `002:4`). Reusing `status` would conflate
the two; the migration comment + this note disambiguate.

## The live entity graph is NOT mutated (AC-042)

`memory.entities` and `memory.entity_edges` are LIVE and populated. Storing a
decision **creates no `entities` row and no `entity_edges` row** — the write path
(`storeMemory`) only touches `memory.memories`. The column-based relations above
are the canonical way decisions relate.

### Optional entity link (operator-only, off by default)

An operator MAY, out of band, add an `entities` row + an `entity_edges` edge to
link a decision to a tracked project/Linear issue/repo. This is explicitly
**optional** and never automatic. The default decision write leaves the entity
graph byte-for-byte unchanged — `SELECT count(*) FROM memory.entity_edges` is
identical before and after storing decisions (asserted in
`test/decision-log.test.ts`, AC-042).
