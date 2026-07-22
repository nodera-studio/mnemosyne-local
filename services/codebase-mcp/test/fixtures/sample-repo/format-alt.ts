// format-alt.ts — the second, same-named `format` that makes the by-name edge
// from ambiguous.ts -> format ambiguous (AC-010).

export function format(value: number): string {
  return value.toFixed(2);
}
