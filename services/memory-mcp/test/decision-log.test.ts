// Wave 6 — decision-log metadata. DB-backed (needs the disposable test DB), with the
// Voyage embedder fully MOCKED via a stubbed fetch (NO live quota). Skipped gracefully
// without DATABASE_URL.
//
//   AC-040 — storing source_kind='decision' enforces + persists the typed shape (project,
//            decision_status ∈ {active,superseded,deferred}, decided_at, supersedes_id,
//            decided_in, related_ids); an invalid decision_status is rejected.
//   AC-041 — supersession resolves via the recursive CTE on supersedes_id; the superseded
//            row flips to decision_status='superseded' + superseded_by=<new id>; a
//            malformed self-cycle terminates (cycle guard).
//   AC-042 — storing decisions does NOT add memory.entities / entity_edges rows.

import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import pg from "pg";

const HAS_DB = !!process.env.DATABASE_URL;

// ── Mock the Voyage embedder (both endpoints) so storeMemory writes deterministic int8
// vectors without touching the network. /embeddings (flat) → legacy embedding;
// /contextualizedembeddings (nested) → embedding_v2.
function fakeFlatResponse(n: number, dim = 1024) {
  return {
    data: Array.from({ length: n }, () => ({
      embedding: Array.from({ length: dim }, () => 1),
    })),
  };
}
function fakeContextualResponse(n: number, dim = 1024) {
  return {
    data: Array.from({ length: n }, (_, docIdx) => ({
      index: docIdx,
      data: [{ index: 0, embedding: Array.from({ length: dim }, () => 1) }],
    })),
  };
}

function mockVoyageFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init: { body: string }) => {
      const body = JSON.parse(init.body);
      let response: unknown;
      if (url.includes("/contextualizedembeddings")) {
        response = fakeContextualResponse(body.inputs.length);
      } else if (url.includes("/embeddings")) {
        response = fakeFlatResponse(body.input.length);
      } else {
        response = {};
      }
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        json: async () => response,
        text: async () => JSON.stringify(response),
      } as unknown as Response;
    }),
  );
}

describe.skipIf(!HAS_DB)(
  "decision-log (Wave 6, DB-backed, mocked Voyage)",
  () => {
    let pool: pg.Pool;
    const projectId = `decision-test-${Date.now()}`;
    // Imported AFTER env is set so config picks up DATABASE_URL/VOYAGE_API_KEY.
    let mem: typeof import("../src/memory.js");

    beforeAll(async () => {
      process.env.VOYAGE_API_KEY ??= "test-key";
      mem = await import("../src/memory.js");
      pool = new pg.Pool({
        connectionString: process.env.DATABASE_URL,
        max: 4,
      });
    });

    afterAll(async () => {
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        projectId,
      ]);
      await pool.end();
    });

    beforeEach(() => {
      mockVoyageFetch();
    });
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("AC-040 — stores a decision with the full typed shape", async () => {
      const related = [
        "11111111-1111-1111-1111-111111111111",
        "22222222-2222-2222-2222-222222222222",
      ];
      const r = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Use voyage-context-3",
        content: "Swap memory embeddings to contextual.",
        sourceKind: "decision",
        decisionProject: "mnemosyne",
        decisionStatus: "active",
        decidedIn: "wave-P",
        relatedIds: related,
      });

      const { rows } = await pool.query(
        `SELECT decision_project, decision_status, decided_at, supersedes_id,
              decided_in, related_ids
       FROM memory.memories WHERE id = $1`,
        [r.id],
      );
      const row = rows[0];
      expect(row.decision_project).toBe("mnemosyne");
      expect(row.decision_status).toBe("active");
      expect(row.decided_at).toBeInstanceOf(Date); // defaulted to now()
      expect(row.supersedes_id).toBeNull();
      expect(row.decided_in).toBe("wave-P");
      expect(row.related_ids).toEqual(related);
    });

    it("AC-040 — non-decision memory leaves the decision columns NULL", async () => {
      const r = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "A plain fact",
        content: "Not a decision.",
        sourceKind: "research",
        // decision fields supplied but must be ignored for non-decision rows
        decisionProject: "should-be-ignored",
        decisionStatus: "active",
      });
      const { rows } = await pool.query(
        `SELECT decision_project, decision_status, decided_at FROM memory.memories WHERE id = $1`,
        [r.id],
      );
      expect(rows[0].decision_project).toBeNull();
      expect(rows[0].decision_status).toBeNull();
      expect(rows[0].decided_at).toBeNull();
    });

    it("AC-040 — rejects an invalid decision_status (code-level, before the DB CHECK)", async () => {
      await expect(
        mem.storeMemory({
          projectId,
          type: "semantic",
          title: "bad status",
          content: "x",
          sourceKind: "decision",
          // @ts-expect-error — deliberately invalid value to exercise the guard
          decisionStatus: "bogus",
        }),
      ).rejects.toThrow(/invalid decisionStatus/);
    });

    it("AC-041 — supersession flips the prior row + decisionChain resolves [B, A]", async () => {
      const a = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Decision A",
        content: "original decision",
        sourceKind: "decision",
        decisionStatus: "active",
        decisionProject: "p",
      });
      const b = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Decision B",
        content: "replaces A",
        sourceKind: "decision",
        decisionStatus: "active",
        decisionProject: "p",
        supersedesId: a.id,
      });

      // A is now superseded, with superseded_by pointing forward to B (inverse pointer).
      const { rows: aRows } = await pool.query(
        `SELECT decision_status, superseded_by FROM memory.memories WHERE id = $1`,
        [a.id],
      );
      expect(aRows[0].decision_status).toBe("superseded");
      expect(aRows[0].superseded_by).toBe(b.id);

      // decisionChain(B) walks backward B → A via supersedes_id.
      const chain = await mem.decisionChain(b.id);
      expect(chain.map((c) => c.id)).toEqual([b.id, a.id]);
      expect(chain[0].depth).toBe(0);
      expect(chain[1].depth).toBe(1);
      expect(chain[1].decision_status).toBe("superseded");
    });

    it("AC-041 — a 3-link chain (C→B→A) resolves in order", async () => {
      const a = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "A3",
        content: "a",
        sourceKind: "decision",
        decisionStatus: "active",
      });
      const b = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "B3",
        content: "b",
        sourceKind: "decision",
        decisionStatus: "active",
        supersedesId: a.id,
      });
      const c = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "C3",
        content: "c",
        sourceKind: "decision",
        decisionStatus: "active",
        supersedesId: b.id,
      });
      const chain = await mem.decisionChain(c.id);
      expect(chain.map((x) => x.id)).toEqual([c.id, b.id, a.id]);
    });

    it("AC-041 — a malformed self-cycle terminates (cycle guard)", async () => {
      // Build a cycle directly in SQL (the write path won't create one): X.supersedes_id = X.
      const { rows } = await pool.query<{ id: string }>(
        `INSERT INTO memory.memories
         (project_id, type, title, content, source_kind, decision_status)
       VALUES ($1,'semantic','cycle','x','decision','active') RETURNING id`,
        [projectId],
      );
      const x = rows[0].id;
      await pool.query(
        `UPDATE memory.memories SET supersedes_id = $1 WHERE id = $1`,
        [x],
      );
      // Without the path-array guard this would infinite-loop; it must terminate and
      // return X exactly once.
      const chain = await mem.decisionChain(x);
      expect(chain.map((c) => c.id)).toEqual([x]);
    });

    // Regression (review Fix 4) — the supersession UPDATE is scoped by id + project_id +
    // source_kind='decision', so a new decision cannot flip a row in a DIFFERENT project
    // or a NON-decision memory. Both cases must be a no-op on the target row.
    it("Fix 4 — supersedesId pointing at ANOTHER project's row does not flip it", async () => {
      const otherProject = `${projectId}-other`;
      // A decision in a different project.
      const foreign = await mem.storeMemory({
        projectId: otherProject,
        type: "semantic",
        title: "Foreign decision",
        content: "belongs to another project",
        sourceKind: "decision",
        decisionStatus: "active",
      });
      // A new decision in OUR project tries to supersede the foreign one.
      await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Cross-project supersede attempt",
        content: "should not flip the foreign row",
        sourceKind: "decision",
        decisionStatus: "active",
        supersedesId: foreign.id,
      });
      const { rows } = await pool.query<{
        decision_status: string;
        superseded_by: string | null;
      }>(
        `SELECT decision_status, superseded_by FROM memory.memories WHERE id = $1`,
        [foreign.id],
      );
      expect(rows[0].decision_status).toBe("active"); // unchanged
      expect(rows[0].superseded_by).toBeNull(); // not flipped
      // Cleanup: clear the cross-project supersedes_id pointer (mem_supersedes_fk) before
      // dropping the foreign row, then remove the foreign project's rows.
      await pool.query(
        `UPDATE memory.memories SET supersedes_id = NULL WHERE supersedes_id = $1`,
        [foreign.id],
      );
      await pool.query(`DELETE FROM memory.memories WHERE project_id = $1`, [
        otherProject,
      ]);
    });

    it("Fix 4 — supersedesId pointing at a NON-decision memory does not flip it", async () => {
      // A plain (non-decision) memory in the SAME project.
      const plain = await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Plain research note",
        content: "not a decision row",
        sourceKind: "research",
      });
      // A new decision tries to supersede the non-decision memory.
      await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Supersede a non-decision",
        content: "should be a no-op on the plain row",
        sourceKind: "decision",
        decisionStatus: "active",
        supersedesId: plain.id,
      });
      const { rows } = await pool.query<{
        decision_status: string | null;
        superseded_by: string | null;
      }>(
        `SELECT decision_status, superseded_by FROM memory.memories WHERE id = $1`,
        [plain.id],
      );
      // Non-decision rows keep NULL decision columns; the supersede UPDATE must not touch them.
      expect(rows[0].decision_status).toBeNull();
      expect(rows[0].superseded_by).toBeNull();
    });

    it("AC-042 — storing decisions does NOT touch entities / entity_edges", async () => {
      const before = await pool.query<{ e: string; ed: string }>(
        `SELECT (SELECT count(*) FROM memory.entities)     AS e,
              (SELECT count(*) FROM memory.entity_edges) AS ed`,
      );
      await mem.storeMemory({
        projectId,
        type: "semantic",
        title: "Decision with no entity",
        content: "relates via columns only",
        sourceKind: "decision",
        decisionStatus: "active",
        relatedIds: ["33333333-3333-3333-3333-333333333333"],
      });
      const after = await pool.query<{ e: string; ed: string }>(
        `SELECT (SELECT count(*) FROM memory.entities)     AS e,
              (SELECT count(*) FROM memory.entity_edges) AS ed`,
      );
      expect(after.rows[0].e).toBe(before.rows[0].e);
      expect(after.rows[0].ed).toBe(before.rows[0].ed);
    });
  },
);

// Pure (no-DB) guard so the file always runs at least one assertion in CI.
describe("decision constants (no DB)", () => {
  it("DECISION_STATUSES is the three-value lifecycle", async () => {
    const mem = await import("../src/memory.js");
    expect(mem.DECISION_STATUSES).toEqual(["active", "superseded", "deferred"]);
  });
});
