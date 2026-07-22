// cycle-a.ts — half of the A <-> B cycle (AC-008 path-array cycle guard).
// alpha() calls beta() in cycle-b.ts, which calls back into alpha().
import { beta } from "./cycle-b.js";

export function alpha(n: number): number {
  if (n <= 0) return 0;
  return beta(n - 1);
}
