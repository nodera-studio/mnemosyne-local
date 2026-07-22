// a-sibling.ts — a dead-end branch off root. `aSibling` calls nothing in the chain and is
// NOT on the path to the target. Its file name sorts BEFORE mid.ts and root.ts, so in the
// old depth-ordered slice it appeared at depth 1 ahead of `mid` and was wrongly included.
export function aSibling(n: number): number {
  return n - 1;
}
