// cycle-b.ts — other half of the A <-> B cycle (AC-008 path-array cycle guard).
// beta() calls alpha() in cycle-a.ts, closing the cycle alpha -> beta -> alpha.
import { alpha } from "./cycle-a.js";

export function beta(n: number): number {
  if (n <= 0) return 0;
  return alpha(n - 1);
}
