import express from "express";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { config, assertServerConfig } from "./config.js";
import { requireBearer, type MnemoAuth } from "./auth.js";
import { originGuard } from "./origin.js";
import * as mem from "./memory.js";

const MEMORY_TYPES = ["episodic", "semantic", "procedural", "entity"] as const;

export const TOOL_DESCRIPTIONS = {
  memory_search:
    "Hybrid semantic and lexical search over durable cross-session memory. Use when: recalling decisions, facts, incidents, or procedures by topic; finding the most relevant memory when exact metadata is unknown; optionally narrowing broad searches with type, tags (ANY-of), or after. Do NOT use when: enumerating by kind/status/tag (memory_list) or browsing newest memories first (memory_get_recent). Example: query='postgres pool sizing', tags=['infra'], after='2026-06-01T00:00:00Z'.",
  memory_store:
    "Persist a distilled durable memory, not a raw transcript. Use when: the user explicitly asks to remember something; a durable decision, fact, or reusable procedure was established; storing sourceKind/tags will help later enumeration. Do NOT use when: fetching existing memory (memory_get), searching prior memory (memory_search), or recording temporary working notes. Example: store a semantic decision with sourceKind='decision' and tags=['retrieval'].",
  memory_get:
    "Fetch the JSON body for one known memory id, budgeted to maxChars of content (default 6000 chars, ~1500 tokens); oversized content is cut with explicit truncated/totalChars markers and a re-fetch note. Use when: memory_search, memory_get_recent, memory_list, or a prior response returned the exact id; auditing status, metadata, event_date, or supersession fields; reading content beyond a snippet — pass full=true (or maxChars=0) for the complete body. Do NOT use when: discovering memories by topic (memory_search) or enumerating sets by metadata (memory_list). Example: id='00000000-0000-4000-8000-000000000001', full=true.",
  memory_get_recent:
    "Return recently created active memories for this project. Use when: checking what was stored most recently; browsing a short recent timeline; optionally narrowing by memory type. Do NOT use when: searching by meaning or topic (memory_search) or counting/listing by sourceKind/status/tag (memory_list). Example: limit=10, type='semantic'.",
  memory_update:
    "Edit fields or lifecycle status on an existing memory. Use when: correcting title/content/importance; marking a memory resolved, superseded, archived, or closed; pinning or unpinning a known id. Do NOT use when: storing a new memory (memory_store) or soft-deleting only (memory_delete). Example: id='<memory id>', status='resolved'.",
  memory_list:
    "Structured enumeration and counts by sourceKind, type, status, or tag. Use when: listing open tech debts, shipped implementations, decisions, or counts; filtering exact metadata; reviewing active vs resolved totals. Do NOT use when: recalling by meaning/topic (memory_search) or fetching one known id (memory_get). Example: sourceKind='tech-debt', status='active', tag='retrieval'.",
  memory_delete:
    "Soft-archive one known memory id so it is excluded from search. Use when: a specific memory is obsolete, duplicated, or should be hidden from active retrieval; preserving recoverability matters; the id is already known. Do NOT use when: changing lifecycle status without archiving (memory_update) or removing an unknown set (memory_list first). Example: id='<memory id>'.",
  memory_get_entity:
    "Fetch one tracked entity and its relationship edges. Use when: looking up a known person, project, issue, repo, or entity id; inspecting stored relationships; resolving entity metadata rather than free-form memories. Do NOT use when: searching memory prose (memory_search) or listing memories by sourceKind/status (memory_list). Example: nameOrId='mnemosyne'.",
  memory_decision_chain:
    "Resolve a decision memory's supersession lineage backward through supersedes_id. Use when: auditing what a known decision replaced; explaining decision history; checking active/superseded decision context. Do NOT use when: searching for decisions by topic (memory_search) or listing all decisions (memory_list). Example: id='<decision memory id>'.",
} as const;

export function buildServer(projectId: string): McpServer {
  const server = new McpServer({ name: "memory-mcp-server", version: "1.0.0" });

  server.registerTool(
    "memory_search",
    {
      title: "Search long-term memory",
      description: TOOL_DESCRIPTIONS.memory_search,
      inputSchema: {
        query: z.string().describe("What to recall"),
        type: z
          .enum(MEMORY_TYPES)
          .optional()
          .describe(
            "Optional strict scoping — kept on the zero-result retry, never auto-dropped; restrict to one memory type",
          ),
        tags: z
          .array(z.string())
          .optional()
          .describe(
            "Optional refinement — auto-dropped (with a notice) if it would zero the results; ANY matching tag narrows the pool",
          ),
        after: z
          .string()
          .datetime()
          .optional()
          .describe(
            "Optional refinement — auto-dropped (with a notice) if it would zero the results; ISO timestamp for eventDate/createdAt cutoff",
          ),
        limit: z.number().int().min(1).max(20).default(5),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ query, type, tags, after, limit }) => {
      try {
        const result = await mem.searchMemory({
          projectId,
          query,
          type,
          tags,
          after,
          limit,
        });
        return {
          content: [
            { type: "text", text: mem.formatHits(result.hits, result) },
          ],
        };
      } catch (e) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `memory_search failed: ${(e as Error).message}`,
            },
          ],
        };
      }
    },
  );

  server.registerTool(
    "memory_store",
    {
      title: "Store a distilled memory",
      description: TOOL_DESCRIPTIONS.memory_store,
      inputSchema: {
        type: z
          .enum(MEMORY_TYPES)
          .describe(
            "episodic=what happened, semantic=durable fact/decision, procedural=how-to, entity=person/project/issue",
          ),
        title: z.string().describe("Short headline"),
        content: z.string().describe("The distilled fact, markdown ok"),
        importance: z.number().min(0).max(1).default(0.5),
        sourceKind: z
          .string()
          .optional()
          .describe(
            "Category for enumeration: tech-debt, research, plan, audit, decision, implementation, requirements, copywrite",
          ),
        tags: z.array(z.string()).optional(),
        metadata: z
          .record(z.any())
          .optional()
          .describe(
            "Structured fields, e.g. tech-debt: {debtId, severity, category, file}",
          ),
        // Decision-log typed fields (AC-040) — only meaningful when sourceKind='decision'.
        decisionProject: z
          .string()
          .optional()
          .describe("Decision: the project/scope this decision belongs to"),
        decisionStatus: z
          .enum(mem.DECISION_STATUSES)
          .optional()
          .describe(
            "Decision lifecycle: active (default) | superseded | deferred",
          ),
        decidedAt: z
          .string()
          .optional()
          .describe("Decision: ISO timestamp it was decided (defaults to now)"),
        supersedesId: z
          .string()
          .optional()
          .describe(
            "Decision: id of the prior decision THIS one replaces (marks that one superseded)",
          ),
        decidedIn: z
          .string()
          .optional()
          .describe("Decision: where it was decided, e.g. session/PR/wave"),
        relatedIds: z
          .array(z.string())
          .optional()
          .describe("Decision: ids of related decisions/memories"),
      },
      annotations: { readOnlyHint: false, idempotentHint: false },
    },
    async ({
      type,
      title,
      content,
      importance,
      sourceKind,
      tags,
      metadata,
      decisionProject,
      decisionStatus,
      decidedAt,
      supersedesId,
      decidedIn,
      relatedIds,
    }) => {
      try {
        const r = await mem.storeMemory({
          projectId,
          type,
          title,
          content,
          importance,
          sourceKind,
          metadata: { ...(metadata ?? {}), ...(tags ? { tags } : {}) },
          decisionProject,
          decisionStatus,
          decidedAt,
          supersedesId,
          decidedIn,
          relatedIds,
        });
        return {
          content: [
            {
              type: "text",
              // AC-301: an exact-content duplicate short-circuits to the existing row
              // (no insert, no embed spend) — say so instead of pretending to store.
              text: r.duplicate
                ? `Already stored as ${r.id} (exact duplicate).`
                : `stored ${type} memory "${r.title}" (id: ${r.id})`,
            },
          ],
        };
      } catch (e) {
        return {
          isError: true,
          content: [
            {
              type: "text",
              text: `memory_store failed: ${(e as Error).message}`,
            },
          ],
        };
      }
    },
  );

  server.registerTool(
    "memory_get",
    {
      title: "Fetch a memory by id",
      description: TOOL_DESCRIPTIONS.memory_get,
      inputSchema: {
        id: z.string().describe("Memory id from a search result"),
        maxChars: z
          .number()
          .int()
          .min(0)
          .max(200000)
          .default(mem.MEMORY_GET_DEFAULT_MAX_CHARS)
          .describe(
            "Response budget in characters (~4 chars/token). 0 = no limit.",
          ),
        full: z
          .boolean()
          .default(false)
          .describe("Return the complete content regardless of maxChars"),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ id, maxChars, full }) => {
      const row = await mem.getMemory(id);
      if (!row)
        return {
          isError: true,
          content: [{ type: "text", text: `no memory ${id}` }],
        };
      // AC-805/806: budget the content field (never the serialized JSON) unless the
      // caller lifts the cap with full=true or maxChars=0.
      const shaped = mem.budgetMemoryBody(row, full ? 0 : maxChars);
      const status = row.status;
      const supersededBy = row.superseded_by;
      let banner = "";
      if (status === "superseded" && supersededBy) {
        banner = `superseded — see ${supersededBy}\n`;
      } else if (status && status !== "active") {
        banner = `status=${status}\n`;
      }
      return {
        content: [
          { type: "text", text: `${banner}${JSON.stringify(shaped, null, 2)}` },
        ],
      };
    },
  );

  server.registerTool(
    "memory_get_recent",
    {
      title: "Recent memories",
      description: TOOL_DESCRIPTIONS.memory_get_recent,
      inputSchema: {
        type: z.enum(MEMORY_TYPES).optional(),
        limit: z.number().int().min(1).max(20).default(10),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ type, limit }) => {
      const hits = await mem.getRecent({ projectId, type, limit });
      return { content: [{ type: "text", text: mem.formatHits(hits) }] };
    },
  );

  server.registerTool(
    "memory_update",
    {
      title: "Edit / re-score a memory",
      description: TOOL_DESCRIPTIONS.memory_update,
      inputSchema: {
        id: z.string(),
        title: z.string().optional(),
        content: z.string().optional(),
        importance: z.number().min(0).max(1).optional(),
        pinned: z.boolean().optional(),
        status: z
          .string()
          .optional()
          .describe(
            'active | resolved | superseded | archived | closed — e.g. set "resolved" when a tech-debt item is fixed',
          ),
      },
      annotations: { readOnlyHint: false, idempotentHint: true },
    },
    async ({ id, ...fields }) => {
      const r = await mem.updateMemory(id, fields);
      if (!r)
        return {
          isError: true,
          content: [{ type: "text", text: `no memory ${id}` }],
        };
      return { content: [{ type: "text", text: `updated ${r.id}` }] };
    },
  );

  server.registerTool(
    "memory_list",
    {
      title: "List / enumerate memories (structured, not search)",
      description: TOOL_DESCRIPTIONS.memory_list,
      inputSchema: {
        sourceKind: z
          .string()
          .optional()
          .describe(
            "e.g. tech-debt, research, plan, audit, decision, implementation",
          ),
        type: z.enum(MEMORY_TYPES).optional(),
        status: z
          .string()
          .optional()
          .describe("active (default) | resolved | superseded | all"),
        tag: z.string().optional(),
        limit: z.number().int().min(1).max(100).default(30),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ sourceKind, type, status, tag, limit }) => {
      const r = await mem.listMemories({
        projectId,
        sourceKind,
        type,
        status,
        tag,
        limit,
      });
      const header = `${r.rows.length} shown · ${r.active} active / ${r.resolved} resolved / ${r.total} total${sourceKind ? ` [${sourceKind}]` : ""}`;
      const body = r.rows
        .map(
          (m) =>
            `• [${m.status}] ${m.title}  (${m.source_kind ?? m.type}, id: ${m.id})`,
        )
        .join("\n");
      return {
        content: [{ type: "text", text: `${header}\n${body || "(none)"}` }],
      };
    },
  );

  server.registerTool(
    "memory_delete",
    {
      title: "Archive (soft-delete) a memory",
      description: TOOL_DESCRIPTIONS.memory_delete,
      inputSchema: { id: z.string() },
      annotations: {
        readOnlyHint: false,
        idempotentHint: true,
        destructiveHint: true,
      },
    },
    async ({ id }) => {
      const ok = await mem.archiveMemory(id);
      return {
        content: [
          {
            type: "text",
            text: ok ? `archived ${id}` : `nothing to archive for ${id}`,
          },
        ],
      };
    },
  );

  server.registerTool(
    "memory_get_entity",
    {
      title: "Fetch an entity + relationships",
      description: TOOL_DESCRIPTIONS.memory_get_entity,
      inputSchema: { nameOrId: z.string().describe("Entity name or id") },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ nameOrId }) => {
      const row = await mem.getEntity({ projectId, nameOrId });
      if (!row)
        return {
          isError: true,
          content: [{ type: "text", text: `no entity "${nameOrId}"` }],
        };
      return {
        content: [{ type: "text", text: JSON.stringify(row, null, 2) }],
      };
    },
  );

  server.registerTool(
    "memory_decision_chain",
    {
      title: "Resolve a decision's supersession chain",
      description: TOOL_DESCRIPTIONS.memory_decision_chain,
      inputSchema: {
        id: z
          .string()
          .describe("Decision (memory) id to resolve the chain from"),
      },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ id }) => {
      const chain = await mem.decisionChain(id);
      if (chain.length === 0)
        return {
          isError: true,
          content: [{ type: "text", text: `no decision ${id}` }],
        };
      return {
        content: [{ type: "text", text: JSON.stringify(chain, null, 2) }],
      };
    },
  );

  return server;
}

function bootstrap(): void {
  assertServerConfig();
  const app = express();
  app.use(express.json({ limit: "4mb" }));

  app.get("/healthz", (_req, res) => {
    res.json({ ok: true, service: "memory-mcp-server" });
  });

  app.use(originGuard);
  app.use(requireBearer);

  app.post("/mcp", async (req, res) => {
    const auth = (req as express.Request & { auth?: MnemoAuth }).auth;
    const projectId = auth?.extra.projectId ?? config.defaultProjectId;
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    res.on("close", () => void transport.close());
    const server = buildServer(projectId);
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  });

  app.listen(config.port, "0.0.0.0", () => {
    console.error(`memory-mcp-server listening on :${config.port}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  bootstrap();
}
