// root.ts — the seed. It calls TWO callees: `aSibling` (a dead-end leaf) and `mid`
// (which leads to the target). The sibling is on a DIFFERENT branch than the target, so a
// real predecessor path root→mid→target must NOT include aSibling. A depth-ordered slice
// of the flattened expansion WOULD wrongly include aSibling (it sorts before mid at depth 1).
import { aSibling } from "./a-sibling.js";
import { mid } from "./mid.js";

export function root(n: number): number {
  return aSibling(n) + mid(n);
}
