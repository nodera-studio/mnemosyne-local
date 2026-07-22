#!/usr/bin/env node
// SessionEnd hook: distill a finished Claude Code session into durable memories.
// Primary summarizer = `claude -p --model claude-haiku-4-5` (off interactive quota,
// uses the Agent SDK credit). Re-entrancy-guarded so the nested claude call can't
// recurse. Failures never block session end.

import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

if (process.env.MNEMO_SUMMARIZING === '1') process.exit(0); // re-entrancy guard

const MCP = process.env.MEMORY_MCP_URL || 'http://127.0.0.1:8080/mcp';
const MODEL = process.env.SUMMARIZER_MODEL || 'claude-haiku-4-5';

let payload = {};
try { payload = JSON.parse(readFileSync(0, 'utf8')); } catch { /* no stdin */ }
const transcriptPath = payload.transcript_path;
const sessionId = payload.session_id || 'unknown';
if (!transcriptPath) process.exit(0);

// Flatten the transcript JSONL into role-tagged text.
let convo = '';
try {
  for (const line of readFileSync(transcriptPath, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let e;
    try { e = JSON.parse(line); } catch { continue; }
    const msg = e.message;
    if (!msg) continue;
    const text = Array.isArray(msg.content)
      ? msg.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n')
      : typeof msg.content === 'string' ? msg.content : '';
    if (text.trim()) convo += `\n[${msg.role || e.type}] ${text}\n`;
  }
} catch { process.exit(0); }

if (convo.length < 200) process.exit(0);
convo = convo.slice(-24000); // cap input tokens

const prompt = `You are a memory distiller. From the coding session transcript below, extract durable, reusable memories worth remembering across FUTURE sessions: decisions made, durable facts, user preferences, repeatable procedures, and important entities (people/projects/issues/repos). Ignore chit-chat, transient debugging, and anything ephemeral. Do NOT store raw transcript.

Output ONLY a JSON array (no prose, no code fences) of objects:
{"type":"episodic"|"semantic"|"procedural"|"entity","title":string,"content":string,"importance":number between 0 and 1}
Max 8 items. If nothing durable, output [].

TRANSCRIPT:
${convo}`;

let out = '';
try {
  out = execFileSync('claude', ['-p', '--model', MODEL], {
    input: prompt,
    env: { ...process.env, MNEMO_SUMMARIZING: '1' },
    maxBuffer: 10 * 1024 * 1024,
    timeout: 120_000,
  }).toString();
} catch (e) {
  process.stderr.write(`summarizer: claude -p failed: ${e.message}\n`);
  process.exit(0);
}

let memories = [];
try {
  const m = out.match(/\[[\s\S]*\]/);
  if (m) memories = JSON.parse(m[0]);
} catch { /* unparseable */ }
if (!Array.isArray(memories) || memories.length === 0) process.exit(0);

const TYPES = ['episodic', 'semantic', 'procedural', 'entity'];
let stored = 0;
for (const mem of memories) {
  if (!mem || !mem.title || !mem.content) continue;
  const body = {
    jsonrpc: '2.0', id: 1, method: 'tools/call',
    params: {
      name: 'memory_store',
      arguments: {
        type: TYPES.includes(mem.type) ? mem.type : 'semantic',
        title: String(mem.title).slice(0, 200),
        content: String(mem.content).slice(0, 4000),
        importance: typeof mem.importance === 'number' ? mem.importance : 0.5,
        tags: ['auto-summary', `session:${sessionId}`],
      },
    },
  };
  try {
    await fetch(MCP, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json, text/event-stream' },
      body: JSON.stringify(body),
    });
    stored++;
  } catch { /* best-effort */ }
}
process.stderr.write(`summarizer: stored ${stored}/${memories.length} memories from session ${sessionId}\n`);
