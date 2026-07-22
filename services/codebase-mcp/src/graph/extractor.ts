// Deterministic AST extractor (Wave 3, Phase 1c). Parses a single TS/TSX file with
// tree-sitter and emits symbols + by-name call/import edges + the file's import list.
//
// Scope (v1): TypeScript + TSX ONLY. JS/JSX fall through with no graph data — the
// same "no language → no graph" behavior the chunker already has. scip-typescript is
// deferred, so edges are resolved BY NAME repo-wide in the indexer's post-walk pass,
// NOT here; this module knows nothing about other files. Names that resolve to more
// than one target are labeled ambiguous at read time (Wave 4 / AC-010).
//
// The parser instances are constructed once per language and reused across every file
// in a run (constructing a Parser per file is slow on large repos). Parsing is wrapped
// so a syntactically broken file yields EMPTY graph data instead of crashing the index.

import { createRequire } from "node:module";
import { extname } from "node:path";

const require = createRequire(import.meta.url);
// tree-sitter ships a native binding; load via createRequire to keep ESM happy.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const Parser = require("tree-sitter") as TreeSitterParserCtor;
// tree-sitter-typescript exports { typescript, tsx } via `export =`.
const TypeScript = require("tree-sitter-typescript") as {
  typescript: TreeSitterLanguage;
  tsx: TreeSitterLanguage;
};

// ── Minimal structural types over the native binding (it ships its own .d.ts, but
//    we keep a narrow local surface so the extractor compiles under strict mode and
//    is not coupled to the binding's full type export). ──────────────────────────
type TreeSitterLanguage = { name: string };
interface SyntaxNode {
  type: string;
  text: string;
  startPosition: { row: number; column: number };
  namedChildCount: number;
  parent: SyntaxNode | null;
  namedChild(index: number): SyntaxNode | null;
  childForFieldName(field: string): SyntaxNode | null;
}
interface Tree {
  rootNode: SyntaxNode;
}
interface ParserInstance {
  setLanguage(lang: TreeSitterLanguage): void;
  parse(input: string): Tree;
}
type TreeSitterParserCtor = new () => ParserInstance;

// ── Public result shapes (plain serializable objects — the indexer writes them via pg). ──

export type SymbolKind =
  | "function"
  | "class"
  | "interface"
  | "type"
  | "enum"
  | "method";

export interface SymbolRow {
  name: string;
  kind: SymbolKind;
  filePath: string;
  startLine: number; // 1-based
}

export interface ByNameEdge {
  fromName: string | null; // null = file-level (no enclosing symbol) — dropped for call edges
  fromFile: string; // the file the edge originates in (for file-scoped import resolution)
  toName: string;
  kind: "call" | "import";
  module?: string; // import module specifier (import edges only)
  siteLine: number; // 1-based line of the call/import site
}

export interface ImportRow {
  name: string; // imported binding name (local alias if aliased)
  module: string; // module specifier
}

export interface ExtractResult {
  symbols: SymbolRow[];
  edges: ByNameEdge[];
  imports: ImportRow[];
}

// ── Parser pool: one instance per language, reused across files. ──────────────────
let tsParser: ParserInstance | null = null;
let tsxParser: ParserInstance | null = null;

function parserFor(language: "typescript" | "tsx"): ParserInstance {
  if (language === "tsx") {
    if (!tsxParser) {
      tsxParser = new Parser();
      tsxParser.setLanguage(TypeScript.tsx);
    }
    return tsxParser;
  }
  if (!tsParser) {
    tsParser = new Parser();
    tsParser.setLanguage(TypeScript.typescript);
  }
  return tsParser;
}

/** Map a path to the tree-sitter grammar to use, or null when out of scope (v1: TS/TSX only). */
export function graphLanguageFor(path: string): "typescript" | "tsx" | null {
  const ext = extname(path).toLowerCase();
  if (ext === ".tsx") return "tsx";
  if (ext === ".ts") {
    // .d.ts has no runtime symbols worth graphing, but the grammar still parses it
    // fine; emitting its interfaces/types is harmless and occasionally useful.
    return "typescript";
  }
  return null;
}

// Declaration node types that define a named symbol, mapped to our kind vocabulary.
const DECL_KIND: Record<string, SymbolKind> = {
  function_declaration: "function",
  generator_function_declaration: "function",
  class_declaration: "class",
  abstract_class_declaration: "class",
  interface_declaration: "interface",
  type_alias_declaration: "type",
  enum_declaration: "enum",
  method_definition: "method",
};

// Is this declarator a callable symbol (`const fn = () => …`) we emitted as a symbol,
// vs. a plain value binding (`const user = handleGetUser(id)`) that is NOT a scope?
// Only the former should attribute enclosed call sites — otherwise a call inside a
// value initializer would wrongly attribute to the variable instead of the function.
function isCallableDeclarator(node: SyntaxNode): boolean {
  const value = node.childForFieldName("value");
  return (
    !!value &&
    (value.type === "arrow_function" ||
      value.type === "function_expression" ||
      value.type === "function")
  );
}

/** The nearest ancestor declaration's symbol name (for attributing a call edge). */
function enclosingSymbolName(node: SyntaxNode): string | null {
  let cur: SyntaxNode | null = node.parent;
  while (cur) {
    if (DECL_KIND[cur.type]) {
      const nameNode = cur.childForFieldName("name");
      if (nameNode) return nameNode.text;
    } else if (
      cur.type === "variable_declarator" &&
      isCallableDeclarator(cur)
    ) {
      const nameNode = cur.childForFieldName("name");
      if (nameNode) return nameNode.text;
    }
    cur = cur.parent;
  }
  return null;
}

/** Resolve a call_expression's callee to a bare name (identifier or member property). */
function calleeName(callExpr: SyntaxNode): string | null {
  const fn = callExpr.childForFieldName("function");
  if (!fn) return null;
  if (fn.type === "identifier") return fn.text;
  if (fn.type === "member_expression") {
    // obj.method(...) → attribute the call to `method` (last segment).
    const prop = fn.childForFieldName("property");
    if (prop) return prop.text;
  }
  return null;
}

/** Depth-first walk; calls `visit` on every named node. */
function walk(node: SyntaxNode, visit: (n: SyntaxNode) => void): void {
  visit(node);
  for (let i = 0; i < node.namedChildCount; i++) {
    const child = node.namedChild(i);
    if (child) walk(child, visit);
  }
}

/** Collect imported binding names + module from an `import_statement`. */
function collectImports(
  stmt: SyntaxNode,
  filePath: string,
  line: number,
  imports: ImportRow[],
  edges: ByNameEdge[],
): void {
  const source = stmt.childForFieldName("source");
  if (!source) return;
  // `source` is a string node; strip the surrounding quotes.
  const moduleSpec = source.text.replace(/^['"`]|['"`]$/g, "");

  const clause = stmt.namedChild(0);
  if (!clause || clause.type !== "import_clause") return;

  for (let i = 0; i < clause.namedChildCount; i++) {
    const child = clause.namedChild(i);
    if (!child) continue;
    if (child.type === "identifier") {
      // default import: `import Foo from '…'`
      pushImport(
        child.text,
        child.text,
        moduleSpec,
        filePath,
        imports,
        edges,
        line,
      );
    } else if (child.type === "namespace_import") {
      // `import * as ns from '…'` — the local binding is the identifier after `as`.
      const id = child.namedChild(child.namedChildCount - 1);
      if (id)
        pushImport(
          id.text,
          id.text,
          moduleSpec,
          filePath,
          imports,
          edges,
          line,
        );
    } else if (child.type === "named_imports") {
      for (let j = 0; j < child.namedChildCount; j++) {
        const spec = child.namedChild(j);
        if (!spec || spec.type !== "import_specifier") continue;
        // Edge target is the ORIGINAL exported name (so it resolves to the
        // declaring symbol); the local binding (alias if present) is what the
        // file actually imports for the code_chunks.imports list.
        const nameNode = spec.childForFieldName("name");
        const aliasNode = spec.childForFieldName("alias");
        if (!nameNode) continue;
        const exportedName = nameNode.text;
        const localName = aliasNode ? aliasNode.text : exportedName;
        pushImport(
          localName,
          exportedName,
          moduleSpec,
          filePath,
          imports,
          edges,
          line,
        );
      }
    }
  }
}

function pushImport(
  localName: string,
  exportedName: string,
  moduleSpec: string,
  filePath: string,
  imports: ImportRow[],
  edges: ByNameEdge[],
  line: number,
): void {
  // import list uses the LOCAL binding (alias if aliased); the edge targets the
  // EXPORTED name so it resolves to the declaring symbol repo-wide.
  imports.push({ name: localName, module: moduleSpec });
  edges.push({
    fromName: null,
    fromFile: filePath,
    toName: exportedName,
    kind: "import",
    module: moduleSpec,
    siteLine: line,
  });
}

/**
 * Extract symbols + by-name edges + the import list from one file's content.
 * Returns empty data for non-TS/TSX paths and for any parse failure (a broken file
 * must not crash the whole repo index).
 */
export function extractGraph(content: string, filePath: string): ExtractResult {
  const empty: ExtractResult = { symbols: [], edges: [], imports: [] };
  const language = graphLanguageFor(filePath);
  if (!language) return empty;

  let tree: Tree;
  try {
    tree = parserFor(language).parse(content);
  } catch {
    return empty;
  }

  const symbols: SymbolRow[] = [];
  const edges: ByNameEdge[] = [];
  const imports: ImportRow[] = [];

  try {
    walk(tree.rootNode, (n) => {
      // ── Symbols ──
      const kind = DECL_KIND[n.type];
      if (kind) {
        const nameNode = n.childForFieldName("name");
        if (nameNode) {
          symbols.push({
            name: nameNode.text,
            kind,
            filePath,
            startLine: n.startPosition.row + 1,
          });
        }
      }

      // ── Exported const arrow functions: `export const fn = () => …` ──
      // The chunker treats `const` as a symbol kind; here we only emit a `function`
      // symbol when the initializer is an arrow/function expression (a real callable).
      if (n.type === "variable_declarator") {
        const value = n.childForFieldName("value");
        const nameNode = n.childForFieldName("name");
        if (
          nameNode &&
          value &&
          (value.type === "arrow_function" ||
            value.type === "function_expression" ||
            value.type === "function")
        ) {
          symbols.push({
            name: nameNode.text,
            kind: "function",
            filePath,
            startLine: n.startPosition.row + 1,
          });
        }
      }

      // ── Call edges (by name) ──
      if (n.type === "call_expression") {
        const to = calleeName(n);
        const from = enclosingSymbolName(n);
        // Drop calls with no enclosing symbol (file-level/global) — v1 has no
        // file pseudo-symbol; those calls are not attributable to a graph node.
        if (to && from) {
          edges.push({
            fromName: from,
            fromFile: filePath,
            toName: to,
            kind: "call",
            siteLine: n.startPosition.row + 1,
          });
        }
      }

      // ── Import edges + import list ──
      if (n.type === "import_statement") {
        collectImports(n, filePath, n.startPosition.row + 1, imports, edges);
      }
    });
  } catch {
    // A walk failure (e.g. a pathological tree) degrades to whatever we collected.
    return { symbols, edges, imports };
  }

  return { symbols, edges, imports };
}
