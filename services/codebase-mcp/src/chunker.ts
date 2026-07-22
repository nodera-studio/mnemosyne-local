import { createHash } from 'node:crypto';
import { extname } from 'node:path';

export interface Chunk {
  content: string;
  startLine: number;
  endLine: number;
  symbolName: string | null;
  symbolKind: string | null;
  contentSha256: string;
}

const EXT_LANG: Record<string, string> = {
  '.ts': 'typescript', '.tsx': 'typescript', '.js': 'javascript', '.jsx': 'javascript',
  '.mjs': 'javascript', '.cjs': 'javascript', '.py': 'python', '.go': 'go', '.rs': 'rust',
  '.java': 'java', '.rb': 'ruby', '.php': 'php', '.c': 'c', '.h': 'c', '.cpp': 'cpp',
  '.hpp': 'cpp', '.cs': 'csharp', '.sql': 'sql', '.sh': 'bash', '.md': 'markdown',
  '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml', '.toml': 'toml', '.css': 'css',
  '.scss': 'scss', '.html': 'html', '.vue': 'vue', '.svelte': 'svelte', '.astro': 'astro',
};

export function languageFor(path: string): string | null {
  return EXT_LANG[extname(path).toLowerCase()] ?? null;
}

const sha256 = (s: string) => createHash('sha256').update(s).digest('hex');
export const fileSha256 = sha256;

const SYMBOL_RE =
  /\b(?:export\s+)?(?:default\s+)?(?:async\s+)?(function|class|interface|type|enum|const|def|struct|impl|fn|func|module)\s+([A-Za-z0-9_$]+)/;

function guessSymbol(lines: string[]): { name: string | null; kind: string | null } {
  for (const l of lines) {
    const m = l.match(SYMBOL_RE);
    if (m) return { kind: m[1], name: m[2] };
  }
  return { name: null, kind: null };
}

/**
 * Pragmatic chunker: sliding line windows with overlap, language-agnostic.
 * Good enough for semantic code search; tree-sitter AST chunking is a planned upgrade.
 */
export function chunkFile(
  content: string,
  opts = { windowLines: 60, overlapLines: 12, maxChars: 4000 },
): Chunk[] {
  const lines = content.split('\n');
  if (lines.length === 0) return [];
  const chunks: Chunk[] = [];
  const step = Math.max(1, opts.windowLines - opts.overlapLines);
  for (let start = 0; start < lines.length; start += step) {
    const slice = lines.slice(start, start + opts.windowLines);
    let text = slice.join('\n');
    if (text.trim().length > 0) {
      if (text.length > opts.maxChars) text = text.slice(0, opts.maxChars);
      const { name, kind } = guessSymbol(slice);
      chunks.push({
        content: text,
        startLine: start + 1,
        endLine: Math.min(start + opts.windowLines, lines.length),
        symbolName: name,
        symbolKind: kind,
        contentSha256: sha256(text),
      });
    }
    if (start + opts.windowLines >= lines.length) break;
  }
  return chunks;
}
