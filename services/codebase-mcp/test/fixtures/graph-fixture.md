# Golden graph fixture — `sample-repo/`

This is the deterministic golden input for the codebase-mcp AST extractor (Wave 3)
and the traversal CTE / `code_graph_expand` tool (Wave 4). It is intentionally tiny
(8 files) so every expected symbol, edge, and depth expansion below can be
hand-verified against the source. `start_line` values are 1-based and reflect the
**formatted** fixture files (Prettier runs on write) — if the fixtures are reformatted
and a line moves, update the matching row here.

The chain `route → handler → service → repo` is the AC-008 route→handler→service→repo
path. `ambiguous.ts` + `format-alt.ts` are the AC-010 by-name ambiguity case.
`cycle-a.ts` ↔ `cycle-b.ts` is the AC-008 path-array cycle guard.

## Expected symbols (`codebase.symbols`)

`kind` is `function` for `export function …` and `interface` for `export interface …`.
`start_line` is the line of the declaration.

| name           | kind      | file            | start_line |
| -------------- | --------- | --------------- | ---------- |
| `route`        | function  | `route.ts`      | 5          |
| `handleGetUser`| function  | `handler.ts`    | 5          |
| `getUser`      | function  | `service.ts`    | 4          |
| `findUser`     | function  | `repo.ts`       | 9          |
| `UserRow`      | interface | `repo.ts`       | 4          |
| `format`       | function  | `ambiguous.ts`  | 6          |
| `useFormat`    | function  | `ambiguous.ts`  | 10         |
| `format`       | function  | `format-alt.ts` | 4          |
| `alpha`        | function  | `cycle-a.ts`    | 5          |
| `beta`         | function  | `cycle-b.ts`    | 5          |

Total: **10 symbols** (note `format` appears twice — the ambiguity case).

## Expected edges (`codebase.symbol_edges`)

`kind` is `call` (a function-call site) or `import` (an `import … from` statement).
`from → to` is by symbol name; the resolver matches a call/import target to a symbol
in the repo. Where more than one symbol shares the target name, the edge is
**name-matched / ambiguous** (AC-010) and must be labeled
`(name-matched, may be ambiguous)` in tool output.

### `call` edges

| from            | to             | ambiguous? | site line             |
| --------------- | -------------- | ---------- | --------------------- |
| `route`         | `handleGetUser`| no         | `route.ts:6`          |
| `handleGetUser` | `getUser`      | no         | `handler.ts:6`        |
| `getUser`       | `findUser`     | no         | `service.ts:5`        |
| `useFormat`     | `format`       | **yes**    | `ambiguous.ts:12`     |
| `alpha`         | `beta`         | no         | `cycle-a.ts:7`        |
| `beta`          | `alpha`        | no         | `cycle-b.ts:7`        |

`useFormat → format` is ambiguous: two `format` symbols exist (`ambiguous.ts:6`,
`format-alt.ts:4`). The extractor may emit either target (or both) but MUST mark it
name-matched.

**Raw extractor output is a SUPERSET of this table.** The extractor emits a by-name
edge for EVERY call site with an enclosing symbol, including calls to built-in /
external methods that have no symbol in this repo — concretely `format → trim`
(`ambiguous.ts:7`) and `format → toFixed` (`format-alt.ts:5`). Those two are correct
raw output but resolve to NO repo symbol, so the indexer's post-walk resolution drops
them and they never become `symbol_edges` rows. The 6 rows above are exactly the call
edges whose `toName` resolves to a symbol in the repo (i.e. what lands in
`symbol_edges`). The extractor unit test asserts the 6 resolvable edges are present and
that the 2 built-in-method edges resolve to zero symbols; the indexer-graph test asserts
the final `symbol_edges` set equals the 6 above (modulo the ambiguous `format` fan-out).

### `import` edges (file → file, by imported symbol)

| from file       | imports         | from file       |
| --------------- | --------------- | --------------- |
| `route.ts`      | `handleGetUser` | `handler.ts`    |
| `handler.ts`    | `getUser`       | `service.ts`    |
| `handler.ts`    | `UserRow`       | `repo.ts`       |
| `service.ts`    | `findUser`,`UserRow` | `repo.ts`  |
| `cycle-a.ts`    | `beta`          | `cycle-b.ts`    |
| `cycle-b.ts`    | `alpha`         | `cycle-a.ts`    |

## Expected `code_graph_expand` for the `route` seed (traversal-depth fixture)

Seed = symbol `route` (`route.ts:5`). Direction = callees (downstream of `route`).
Rows are `{name, file, line, depth, edge_type}`. Depth 0 is the seed (def). Each
deeper level follows `call` edges one hop further. The chain is linear, so each
depth adds exactly one symbol.

### depth = 1

| name            | file         | line | depth | edge_type |
| --------------- | ------------ | ---- | ----- | --------- |
| `route`         | `route.ts`   | 5    | 0     | def       |
| `handleGetUser` | `handler.ts` | 5    | 1     | call      |

### depth = 2

| name            | file         | line | depth | edge_type |
| --------------- | ------------ | ---- | ----- | --------- |
| `route`         | `route.ts`   | 5    | 0     | def       |
| `handleGetUser` | `handler.ts` | 5    | 1     | call      |
| `getUser`       | `service.ts` | 4    | 2     | call      |

### depth = 4

| name            | file         | line | depth | edge_type |
| --------------- | ------------ | ---- | ----- | --------- |
| `route`         | `route.ts`   | 5    | 0     | def       |
| `handleGetUser` | `handler.ts` | 5    | 1     | call      |
| `getUser`       | `service.ts` | 4    | 2     | call      |
| `findUser`      | `repo.ts`    | 9    | 3     | call      |

The chain bottoms out at `findUser` (depth 3): `repo.ts` is the terminal node and
has no outgoing `call` edges, so depth 4 returns the same 4 rows as depth 3. This
verifies the traversal terminates cleanly at a leaf rather than padding empty rows.

## Cycle-guard expectation (`alpha`/`beta`)

Seed = `alpha` (`cycle-a.ts:5`), callees, depth ≥ 4. Edges form a cycle
(`alpha → beta → alpha → …`). A correct path-array cycle guard visits each symbol
once per acyclic path and stops re-entering a symbol already on the current path:

| name    | file         | line | depth | edge_type |
| ------- | ------------ | ---- | ----- | --------- |
| `alpha` | `cycle-a.ts` | 5    | 0     | def       |
| `beta`  | `cycle-b.ts` | 5    | 1     | call      |

At depth 2 the only callee of `beta` is `alpha`, which is already on the path
(`alpha → beta → alpha`), so the guard stops. Output must be **finite** (no infinite
loop / stack overflow) — exactly these 2 rows for the `alpha` cycle seed.
