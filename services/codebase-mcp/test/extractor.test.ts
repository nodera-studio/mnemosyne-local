// Wave 3 — extractor golden test (AC-004 / AC-010). Runs the tree-sitter extractor over
// the Wave-0 sample-repo/ fixture and asserts the emitted symbols + by-name edges match
// the golden expectations in fixtures/graph-fixture.md. No DB — pure AST output.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { extractGraph, type ExtractResult } from "../src/graph/extractor.js";

const here = dirname(fileURLToPath(import.meta.url));
const dir = join(here, "fixtures", "sample-repo");

const FILES = [
  "route.ts",
  "handler.ts",
  "service.ts",
  "repo.ts",
  "ambiguous.ts",
  "format-alt.ts",
  "cycle-a.ts",
  "cycle-b.ts",
];

function extractAll(): {
  byFile: Record<string, ExtractResult>;
  symbols: { name: string; kind: string; file: string; line: number }[];
  calls: { from: string | null; to: string; file: string; line: number }[];
  imports: { to: string; module?: string; file: string }[];
} {
  const byFile: Record<string, ExtractResult> = {};
  const symbols: { name: string; kind: string; file: string; line: number }[] =
    [];
  const calls: {
    from: string | null;
    to: string;
    file: string;
    line: number;
  }[] = [];
  const imports: { to: string; module?: string; file: string }[] = [];
  for (const f of FILES) {
    const r = extractGraph(readFileSync(join(dir, f), "utf8"), f);
    byFile[f] = r;
    for (const s of r.symbols)
      symbols.push({ name: s.name, kind: s.kind, file: f, line: s.startLine });
    for (const e of r.edges) {
      if (e.kind === "call")
        calls.push({
          from: e.fromName,
          to: e.toName,
          file: f,
          line: e.siteLine,
        });
      else imports.push({ to: e.toName, module: e.module, file: f });
    }
  }
  return { byFile, symbols, calls, imports };
}

describe("extractor golden output (AC-004)", () => {
  const all = extractAll();

  it("emits exactly the 10 golden symbols with correct kind + start_line", () => {
    const expected = [
      { name: "route", kind: "function", file: "route.ts", line: 5 },
      { name: "handleGetUser", kind: "function", file: "handler.ts", line: 5 },
      { name: "getUser", kind: "function", file: "service.ts", line: 4 },
      { name: "UserRow", kind: "interface", file: "repo.ts", line: 4 },
      { name: "findUser", kind: "function", file: "repo.ts", line: 9 },
      { name: "format", kind: "function", file: "ambiguous.ts", line: 6 },
      { name: "useFormat", kind: "function", file: "ambiguous.ts", line: 10 },
      { name: "format", kind: "function", file: "format-alt.ts", line: 4 },
      { name: "alpha", kind: "function", file: "cycle-a.ts", line: 5 },
      { name: "beta", kind: "function", file: "cycle-b.ts", line: 5 },
    ];
    // Order-independent set comparison.
    const key = (s: {
      name: string;
      kind: string;
      file: string;
      line: number;
    }) => `${s.file}:${s.name}:${s.kind}:${s.line}`;
    const got = new Set(all.symbols.map(key));
    for (const e of expected) expect(got).toContain(key(e));
    expect(all.symbols).toHaveLength(10);
  });

  it("two same-named `format` symbols exist (the AC-010 ambiguity case)", () => {
    const formats = all.symbols.filter((s) => s.name === "format");
    expect(formats).toHaveLength(2);
    expect(new Set(formats.map((s) => s.file))).toEqual(
      new Set(["ambiguous.ts", "format-alt.ts"]),
    );
  });

  it("emits the 6 resolvable call edges (route chain + cycle + ambiguous use)", () => {
    const goldenCalls = [
      { from: "route", to: "handleGetUser", file: "route.ts", line: 6 },
      { from: "handleGetUser", to: "getUser", file: "handler.ts", line: 6 },
      { from: "getUser", to: "findUser", file: "service.ts", line: 5 },
      { from: "useFormat", to: "format", file: "ambiguous.ts", line: 12 },
      { from: "alpha", to: "beta", file: "cycle-a.ts", line: 7 },
      { from: "beta", to: "alpha", file: "cycle-b.ts", line: 7 },
    ];
    const key = (c: {
      from: string | null;
      to: string;
      file: string;
      line: number;
    }) => `${c.from}->${c.to}@${c.file}:${c.line}`;
    const got = new Set(all.calls.map(key));
    for (const g of goldenCalls) expect(got).toContain(key(g));
  });

  it("route → handler → service → repo chain is present (AC-008 seed chain)", () => {
    const has = (from: string, to: string) =>
      all.calls.some((c) => c.from === from && c.to === to);
    expect(has("route", "handleGetUser")).toBe(true);
    expect(has("handleGetUser", "getUser")).toBe(true);
    expect(has("getUser", "findUser")).toBe(true);
  });

  it("A ↔ B cycle is present (alpha→beta, beta→alpha)", () => {
    const has = (from: string, to: string) =>
      all.calls.some((c) => c.from === from && c.to === to);
    expect(has("alpha", "beta")).toBe(true);
    expect(has("beta", "alpha")).toBe(true);
  });

  it("built-in method calls (trim/toFixed) are emitted but resolve to no repo symbol", () => {
    // The extractor emits these raw; they are a superset over the golden resolved set.
    expect(all.calls.some((c) => c.to === "trim")).toBe(true);
    expect(all.calls.some((c) => c.to === "toFixed")).toBe(true);
    // No symbol named trim/toFixed exists, so the post-walk resolver drops them.
    expect(all.symbols.some((s) => s.name === "trim")).toBe(false);
    expect(all.symbols.some((s) => s.name === "toFixed")).toBe(false);
  });

  it("`route → handleGetUser` attributes to the function, not the `const user` declarator", () => {
    const routeCall = all.calls.find(
      (c) => c.to === "handleGetUser" && c.file === "route.ts",
    );
    expect(routeCall?.from).toBe("route");
  });

  it("emits import edges with the imported (exported) name + module, and the import list", () => {
    // route.ts imports handleGetUser from ./handler.js
    const routeImports = all.imports.filter((i) => i.file === "route.ts");
    expect(routeImports).toEqual([
      { to: "handleGetUser", module: "./handler.js", file: "route.ts" },
    ]);
    // service.ts imports findUser + UserRow from ./repo.js
    const svc = extractGraph(
      readFileSync(join(dir, "service.ts"), "utf8"),
      "service.ts",
    );
    expect(svc.imports.map((i) => i.name).sort()).toEqual([
      "UserRow",
      "findUser",
    ]);
  });

  it("returns empty graph for non-TS/TSX files (out-of-scope languages)", () => {
    const r = extractGraph("def foo():\n    pass\n", "thing.py");
    expect(r).toEqual({ symbols: [], edges: [], imports: [] });
  });

  it("returns partial/empty graph for a syntactically broken file (no crash)", () => {
    const r = extractGraph("export function broken( { { {", "broken.ts");
    // Must not throw; tree-sitter is error-tolerant so it may still find the name.
    expect(Array.isArray(r.symbols)).toBe(true);
    expect(Array.isArray(r.edges)).toBe(true);
  });

  it("parses TSX (the tsx grammar) without error", () => {
    const tsx = `import { Btn } from "./btn.js";
export function View() {
  return <div onClick={() => Btn()} />;
}`;
    const r = extractGraph(tsx, "View.tsx");
    expect(r.symbols.some((s) => s.name === "View")).toBe(true);
    expect(r.imports.some((i) => i.name === "Btn")).toBe(true);
  });
});
