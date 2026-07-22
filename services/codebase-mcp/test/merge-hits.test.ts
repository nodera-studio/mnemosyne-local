import { describe, expect, it } from "vitest";
import { mergeAdjacentHits } from "../src/merge-hits.js";
import { formatHits } from "../src/search.js";
import type { MergeableCodeHit } from "../src/merge-hits.js";

function hit(
  chunkId: string,
  filePath: string,
  startLine: number,
  endLine: number,
  extra: Partial<MergeableCodeHit> = {},
): MergeableCodeHit {
  return {
    chunkId,
    repositoryId: "repo",
    filePath,
    startLine,
    endLine,
    symbolName: null,
    language: "typescript",
    snippet: `snippet ${chunkId}`,
    ...extra,
  };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("mergeAdjacentHits", () => {
  it("merges two overlapping chunks of one file to the union range", () => {
    const hits = [
      hit("leader", "src/a.ts", 1, 60, {
        symbolName: "first",
        snippet: "leader snippet",
      }),
      hit("second", "src/a.ts", 49, 108, {
        symbolName: "second",
        snippet: "second snippet",
      }),
    ];

    expect(mergeAdjacentHits(hits, 120)).toEqual([
      {
        ...hits[0],
        startLine: 1,
        endLine: 108,
        mergedCount: 2,
      },
    ]);
  });

  it("formats the merged union range with the merged-count note", () => {
    const merged = mergeAdjacentHits(
      [hit("a", "src/a.ts", 1, 60), hit("b", "src/a.ts", 49, 108)],
      120,
    );

    expect(formatHits(merged).split("\n")[0]).toBe(
      "1. src/a.ts:1-108 (spans 2 chunks)  [a]",
    );
  });

  it("chain-merges until the span cap splits the next chunk", () => {
    const hits = [
      hit("a", "src/a.ts", 1, 40),
      hit("b", "src/a.ts", 41, 80),
      hit("c", "src/a.ts", 81, 120),
    ];

    expect(mergeAdjacentHits(hits, 80)).toEqual([
      { ...hits[0], startLine: 1, endLine: 80, mergedCount: 2 },
      hits[2],
    ]);
  });

  it("never merges chunks from different files", () => {
    const hits = [
      hit("a", "src/a.ts", 1, 60),
      hit("b", "src/b.ts", 49, 108),
    ];

    expect(mergeAdjacentHits(hits, 120)).toEqual(hits);
  });

  it("never merges same-path chunks from different repositories", () => {
    const hits = [
      hit("a", "src/config.ts", 1, 60, { repositoryId: "repo-a" }),
      hit("b", "src/config.ts", 49, 108, { repositoryId: "repo-b" }),
    ];

    expect(mergeAdjacentHits(hits, 120)).toEqual(hits);
  });

  it("preserves the original rank order of survivor hits", () => {
    const hits = [
      hit("b1", "src/b.ts", 100, 120),
      hit("a1", "src/a.ts", 1, 60),
      hit("c1", "src/c.ts", 5, 8),
      hit("a2", "src/a.ts", 49, 108),
      hit("a3", "src/a.ts", 200, 220),
    ];

    expect(mergeAdjacentHits(hits, 120).map((h) => h.chunkId)).toEqual([
      "b1",
      "a1",
      "c1",
      "a3",
    ]);
  });

  it("merges at the exact span cap and does not merge one line beyond it", () => {
    const exact = [
      hit("a", "src/a.ts", 1, 60),
      hit("b", "src/a.ts", 61, 120),
    ];
    const tooWide = [
      hit("a", "src/a.ts", 1, 60),
      hit("b", "src/a.ts", 61, 121),
    ];

    expect(mergeAdjacentHits(exact, 120)).toEqual([
      { ...exact[0], startLine: 1, endLine: 120, mergedCount: 2 },
    ]);
    expect(mergeAdjacentHits(tooWide, 120)).toEqual(tooWide);
  });

  it("property: output line ranges within a file never overlap when mergeable components fit the cap", () => {
    // When a mergeable component exceeds the span cap, survivors can overlap:
    // adjacent chunks are built with a 12-line overlap, and cap-split survivors
    // may retain that bounded overlap. The never-overlap property applies when
    // each mergeable component fits under the cap.
    for (let seed = 1; seed <= 100; seed++) {
      const rnd = mulberry32(seed);
      const hits: MergeableCodeHit[] = [];
      for (const filePath of ["src/a.ts", "src/b.ts", "src/c.ts"]) {
        let cursor = 1;
        for (let i = 0; i < 12; i++) {
          const len = 5 + Math.floor(rnd() * 20);
          const gap = Math.floor(rnd() * 8) - 3;
          const startLine = Math.max(1, cursor + gap);
          const endLine = startLine + len - 1;
          hits.push(hit(`${filePath}:${i}`, filePath, startLine, endLine));
          cursor = endLine + 1;
        }
      }

      const shuffled = hits
        .map((h) => ({ h, key: rnd() }))
        .sort((a, b) => a.key - b.key)
        .map(({ h }) => h);
      const merged = mergeAdjacentHits(shuffled, 10_000);

      for (const filePath of ["src/a.ts", "src/b.ts", "src/c.ts"]) {
        const ranges = merged
          .filter((h) => h.filePath === filePath)
          .sort((a, b) => a.startLine - b.startLine);
        for (let i = 1; i < ranges.length; i++) {
          expect(ranges[i].startLine).toBeGreaterThan(ranges[i - 1].endLine);
        }
      }
    }
  });

  it("bounds capped survivor overlap to the chunk overlap", () => {
    const merged = mergeAdjacentHits(
      [
        hit("a", "src/a.ts", 1, 60),
        hit("b", "src/a.ts", 49, 108),
        hit("c", "src/a.ts", 97, 156),
      ],
      120,
    ).sort((a, b) => a.startLine - b.startLine);

    expect(merged).toHaveLength(2);
    const overlap =
      Math.min(merged[0].endLine, merged[1].endLine) -
      Math.max(merged[0].startLine, merged[1].startLine) +
      1;
    expect(overlap).toBeGreaterThan(0);
    expect(overlap).toBeLessThanOrEqual(12);
  });
});
