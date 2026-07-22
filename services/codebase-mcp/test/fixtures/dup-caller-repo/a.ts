// a.ts — file A defines `foo`, and A.foo is the ONLY `foo` that calls `bar`.
// The call-edge FROM join must anchor on file_path too, so this edge originates
// only from a.ts's `foo`, not from b.ts's same-named `foo`.
export function bar(x: number): number {
  return x + 1;
}

export function foo(x: number): number {
  return bar(x);
}
