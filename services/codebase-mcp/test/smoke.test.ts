import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(here, "fixtures");
const sampleRepoDir = join(fixturesDir, "sample-repo");

const SAMPLE_FILES = [
  "route.ts",
  "handler.ts",
  "service.ts",
  "repo.ts",
  "ambiguous.ts",
  "format-alt.ts",
  "cycle-a.ts",
  "cycle-b.ts",
];

describe("golden graph fixture", () => {
  it("every sample-repo file exists", () => {
    for (const f of SAMPLE_FILES) {
      expect(existsSync(join(sampleRepoDir, f)), `${f} missing`).toBe(true);
    }
  });

  it("graph-fixture.md parses and documents the route chain + edges", () => {
    const md = readFileSync(join(fixturesDir, "graph-fixture.md"), "utf8");
    expect(md.length).toBeGreaterThan(0);
    // The documented chain and its acceptance hooks must be present.
    for (const token of [
      "route",
      "handleGetUser",
      "getUser",
      "findUser",
      "name-matched",
      "cycle",
    ]) {
      expect(md, `graph-fixture.md must mention ${token}`).toContain(token);
    }
  });

  it("the route -> handler -> service -> repo chain uses real import + call syntax", () => {
    const route = readFileSync(join(sampleRepoDir, "route.ts"), "utf8");
    expect(route).toContain("./handler.js");
    expect(route).toContain("handleGetUser(");

    const service = readFileSync(join(sampleRepoDir, "service.ts"), "utf8");
    expect(service).toContain("./repo.js");
    expect(service).toContain("findUser(");
  });

  it("contains the ambiguity case (two `format` functions)", () => {
    const ambiguous = readFileSync(join(sampleRepoDir, "ambiguous.ts"), "utf8");
    const alt = readFileSync(join(sampleRepoDir, "format-alt.ts"), "utf8");
    expect(ambiguous).toContain("export function format(");
    expect(alt).toContain("export function format(");
  });

  it("contains the A <-> B cycle", () => {
    const a = readFileSync(join(sampleRepoDir, "cycle-a.ts"), "utf8");
    const b = readFileSync(join(sampleRepoDir, "cycle-b.ts"), "utf8");
    expect(a).toContain("./cycle-b.js");
    expect(b).toContain("./cycle-a.js");
    expect(a).toContain("beta(");
    expect(b).toContain("alpha(");
  });
});
