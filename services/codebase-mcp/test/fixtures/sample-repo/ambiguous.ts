// ambiguous.ts — AC-010 case: two functions named `format` defined in the repo
// (one here, one in format-alt.ts). A by-name edge resolver cannot tell which
// `format` a caller meant, so any edge resolved to `format` must be labeled
// "(name-matched, may be ambiguous)".

export function format(value: string): string {
  return value.trim();
}

export function useFormat(value: string): string {
  // Call to `format` — ambiguous: matches both ambiguous.ts and format-alt.ts.
  return format(value);
}
