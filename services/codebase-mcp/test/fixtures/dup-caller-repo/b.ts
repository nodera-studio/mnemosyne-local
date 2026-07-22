// b.ts — file B ALSO defines `foo`, but B.foo does NOT call `bar`. Before the
// file-scoped FROM join fix, a.foo's call to bar over-connected to B.foo too
// (sf.name = 'foo' matched both files), producing a spurious b.foo -> bar edge.
export function foo(x: number): number {
  return x * 2;
}
