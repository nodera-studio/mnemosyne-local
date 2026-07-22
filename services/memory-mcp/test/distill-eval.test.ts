// distill-eval operator script (wave-2 Step 7) — PURE tests: pool and LLM are both
// injected mocks (AC-108: zero live calls). Verifies the paid gate, the stratified
// sampling SQL, candidate parsing/dedupe, and that every written row is approved:false.

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it, vi } from "vitest";
import {
  COST_NOTE,
  dedupeCandidates,
  distillEval,
  guardPaidRun,
  parseCandidates,
  type SampledMemory,
} from "../src/db/distill-eval.js";

const tmp = mkdtempSync(join(tmpdir(), "distill-eval-mem-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const SAMPLE: SampledMemory[] = [
  {
    id: "11111111-0000-4000-8000-000000000001",
    type: "semantic",
    source_kind: "decision",
    title: "Voyage embedding model decision",
    excerpt: "We picked voyage-context-4 because...",
  },
  {
    id: "11111111-0000-4000-8000-000000000002",
    type: "procedural",
    source_kind: null,
    title: "Deploy runbook",
    excerpt: "Redeploy via docker compose up --build...",
  },
];

function mockPool(rows: SampledMemory[]) {
  return {
    query: vi.fn(async (_text: string, _params: unknown[]) => ({ rows })),
  };
}

describe("distill-eval (memory) — paid gate", () => {
  it("refuses without --yes, with the cost note", () => {
    expect(guardPaidRun([])).toBe(COST_NOTE);
    expect(guardPaidRun(["--limit", "10"])).toMatch(/PAID operator script/);
    expect(guardPaidRun(["--limit", "10"])).toMatch(/--yes/);
  });

  it("passes with --yes", () => {
    expect(guardPaidRun(["--yes"])).toBeNull();
  });
});

describe("distill-eval (memory) — parsing + dedupe", () => {
  it("extracts the JSON array from fenced/prosey output and forces the fixed fields", () => {
    const raw =
      'Sure! Here are the candidates:\n```json\n[{"query":"where is the deploy runbook","suggestedGold":["a"],"archetype":"conceptual-where-how"},{"query":"","suggestedGold":[]},{"query":"q2","suggestedGold":"not-an-array","archetype":7}]\n```';
    const parsed = parseCandidates(raw);
    expect(parsed).toHaveLength(2); // the empty-query row is dropped
    expect(parsed[0]).toEqual({
      query: "where is the deploy runbook",
      suggestedGold: ["a"],
      archetype: "conceptual-where-how",
      provenance: "distilled",
      approved: false,
    });
    expect(parsed[1].suggestedGold).toEqual([]); // non-array coerced to []
    expect(parsed[1].archetype).toBe("unknown"); // non-string coerced
  });

  it("throws a clear error when the output has no JSON array", () => {
    expect(() => parseCandidates("I cannot help with that")).toThrow(
      /no JSON array/,
    );
  });

  it("dedupes on normalized query text, first occurrence wins", () => {
    const rows = parseCandidates(
      '[{"query":"How to deploy","suggestedGold":["a"],"archetype":"x"},{"query":"  how to DEPLOY ","suggestedGold":["b"],"archetype":"y"}]',
    );
    const deduped = dedupeCandidates(rows);
    expect(deduped).toHaveLength(1);
    expect(deduped[0].suggestedGold).toEqual(["a"]);
  });
});

describe("distill-eval (memory) — end-to-end with injected deps", () => {
  it("samples stratified, prompts the injected model, writes approved:false candidates", async () => {
    const pool = mockPool(SAMPLE);
    const complete = vi.fn(
      async (_system: string, _user: string) =>
        `[{"query":"which embedding model did we pick","suggestedGold":["${SAMPLE[0].id}"],"archetype":"decision-recall"},
          {"query":"how do I redeploy the stack","suggestedGold":["${SAMPLE[1].id}"],"archetype":"conceptual-where-how"}]`,
    );
    const outPath = join(tmp, "candidates.json");

    const { candidates, path } = await distillEval({
      pool,
      complete,
      projectId: "proj-x",
      perStratum: 5,
      outPath,
      log: () => {},
    });

    // Stratified sampling SQL, scoped to the project.
    expect(pool.query).toHaveBeenCalledTimes(1);
    const [sql, params] = pool.query.mock.calls[0]!;
    expect(sql).toMatch(/PARTITION BY type, COALESCE\(source_kind, ''\)/);
    expect(sql).toMatch(/row_number\(\) OVER/);
    expect(params).toEqual(["proj-x", 5]);

    // The model saw the sampled rows (ids + titles) and the archetype instructions.
    expect(complete).toHaveBeenCalledTimes(1);
    const [system, user] = complete.mock.calls[0]!;
    expect(system).toMatch(/conceptual-where-how/);
    expect(user).toContain(SAMPLE[0].id);
    expect(user).toContain("Deploy runbook");

    // Written file: every row approved:false + provenance distilled (AC-109).
    expect(path).toBe(outPath);
    const written = JSON.parse(readFileSync(outPath, "utf8")) as {
      _note: string;
      candidates: Array<{ approved: boolean; provenance: string }>;
    };
    expect(written.candidates).toHaveLength(2);
    for (const c of written.candidates) {
      expect(c.approved).toBe(false);
      expect(c.provenance).toBe("distilled");
    }
    expect(written._note).toMatch(/approvedBy/);
    expect(candidates).toHaveLength(2);
  });

  it("fails loudly on an empty corpus instead of prompting the model", async () => {
    const pool = mockPool([]);
    const complete = vi.fn(async () => "[]");
    await expect(
      distillEval({ pool, complete, projectId: "empty", log: () => {} }),
    ).rejects.toThrow(/no active memories/);
    expect(complete).not.toHaveBeenCalled(); // no quota spent on nothing
  });
});
