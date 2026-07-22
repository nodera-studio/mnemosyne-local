// mid.ts — the real middle hop on root → mid → target. `mid` calls `target`.
import { target } from "./target.js";

export function mid(n: number): number {
  return target(n);
}
