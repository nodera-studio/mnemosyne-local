import { describe, expect, it } from "vitest";
import {
  formatIndexStatus,
  type ChunkAggRow,
  type IndexRunRow,
} from "../src/status-format.js";

const run = (over: Partial<IndexRunRow> = {}): IndexRunRow => ({
  repository_id: "demo",
  phase: "done",
  files_done: 8,
  files_total: 8,
  chunks_total: 8,
  current_file: null,
  error: null,
  started_at: "2026-06-23T19:00:00.000Z",
  finished_at: "2026-06-23T19:00:05.000Z",
  ...over,
});

const chunk = (over: Partial<ChunkAggRow> = {}): ChunkAggRow => ({
  repository_id: "demo",
  files: 8,
  chunks: 8,
  last_indexed: "2026-06-23T19:00:05.000Z",
  ...over,
});

describe("formatIndexStatus (AC-003)", () => {
  it("appends a Last run line to an indexed repo's chunk aggregate", () => {
    const text = formatIndexStatus([chunk()], [run()]);
    expect(text).toContain("demo: 8 files, 8 chunks");
    expect(text).toContain("Last run: done (8/8 files, 8 chunks)");
  });

  it("surfaces a run with no chunks yet (in-flight first index)", () => {
    const text = formatIndexStatus(
      [],
      [
        run({
          phase: "embedding",
          files_done: 2,
          finished_at: null,
          current_file: "src/a.ts",
        }),
      ],
    );
    expect(text).toContain("demo: 0 chunks (not yet indexed)");
    expect(text).toContain("Last run: embedding (2/8 files");
    expect(text).toContain("current=src/a.ts");
    expect(text).toContain("running");
  });

  it("shows the error text for a failed run", () => {
    const text = formatIndexStatus(
      [],
      [
        run({
          phase: "error",
          error: "boom",
          finished_at: "2026-06-23T19:00:02.000Z",
        }),
      ],
    );
    expect(text).toContain("Last run: error");
    expect(text).toContain('error="boom"');
  });

  it("falls back to the empty message when nothing is indexed", () => {
    expect(formatIndexStatus([], [])).toBe("no repos indexed yet");
  });

  it("renders chunk aggregates without a run row unchanged (no regression)", () => {
    const text = formatIndexStatus([chunk()], []);
    expect(text).toBe(
      "demo: 8 files, 8 chunks (last 2026-06-23T19:00:05.000Z)",
    );
    expect(text).not.toContain("Last run");
  });
});
