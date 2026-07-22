// distill-eval operator script — codebase twin (wave-2 Step 7). PURE tests: pool and
// LLM are injected mocks (AC-108: zero live calls). Verifies the paid gate, the
// (language, top-level dir) stratified sampling SQL, and approved:false on every row.

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import {
  COST_NOTE,
  distillEval,
  guardPaidRun,
  type SampledChunk,
} from "../src/db/distill-eval.js";

const tmp = mkdtempSync(join(tmpdir(), "distill-eval-code-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const SAMPLE: SampledChunk[] = [
  {
    file_path: "services/codebase-mcp/src/auth.ts",
    language: "typescript",
    symbol_name: "verifyBearer",
    excerpt: "export function verifyBearer(token: string) { ... }",
  },
  {
    file_path: "services/memory-mcp/src/db/pool.ts",
    language: "typescript",
    symbol_name: null,
    excerpt: "export const pool = new Pool({ connectionString });",
  },
];

describe("distill-eval (codebase) — paid gate", () => {
  it("refuses without --yes, passes with it", () => {
    expect(guardPaidRun([])).toBe(COST_NOTE);
    expect(guardPaidRun(["--yes"])).toBeNull();
  });
});

describe("distill-eval (codebase) — end-to-end with injected deps", () => {
  it("samples by (language, top dir), prompts the injected model, writes path-keyed hints", async () => {
    const pool = {
      query: vi.fn(async (_t: string, _p: unknown[]) => ({ rows: SAMPLE })),
    };
    const complete = vi.fn(
      async (_system: string, _user: string) =>
        `[{"query":"bearer token verification","suggestedGold":["${SAMPLE[0].file_path}"],"archetype":"exact-symbol"}]`,
    );
    const outPath = join(tmp, "candidates.json");

    const { candidates } = await distillEval({
      pool,
      complete,
      projectId: "proj-y",
      perStratum: 4,
      outPath,
      log: () => {},
    });

    const [sql, params] = pool.query.mock.calls[0]!;
    expect(sql).toMatch(
      /PARTITION BY COALESCE\(language, ''\), split_part\(file_path, '\/', 1\)/,
    );
    expect(params).toEqual(["proj-y", 4]);

    const [, user] = complete.mock.calls[0]!;
    expect(user).toContain("services/codebase-mcp/src/auth.ts");
    expect(user).toContain("verifyBearer");

    expect(candidates).toHaveLength(1);
    expect(candidates[0].suggestedGold).toEqual([SAMPLE[0].file_path]);
    const written = JSON.parse(readFileSync(outPath, "utf8")) as {
      candidates: Array<{ approved: boolean; provenance: string }>;
    };
    expect(written.candidates[0].approved).toBe(false);
    expect(written.candidates[0].provenance).toBe("distilled");
  });

  it("fails loudly on an unindexed corpus instead of prompting the model", async () => {
    const pool = { query: vi.fn(async () => ({ rows: [] })) };
    const complete = vi.fn(async () => "[]");
    await expect(
      distillEval({ pool, complete, projectId: "empty", log: () => {} }),
    ).rejects.toThrow(/no indexed chunks/);
    expect(complete).not.toHaveBeenCalled();
  });
});
