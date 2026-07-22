import express from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import { config, assertServerConfig } from './config.js';
import { requireBearer, type MnemoAuth } from './auth.js';
import { originGuard } from './origin.js';
import { pool } from './db/pool.js';

function buildServer(projectId: string): McpServer {
  const server = new McpServer({ name: 'secrets-mcp-server', version: '1.0.0' });

  server.registerTool(
    'secret_set',
    {
      title: 'Store/update an encrypted secret',
      description:
        'Encrypt + store a secret (pgcrypto, box-local key). Overwrites if the name exists. ' +
        'Values are NEVER embedded, searched, or logged.',
      inputSchema: {
        name: z.string().describe('Secret name, e.g. STRIPE_SECRET_KEY'),
        value: z.string().describe('The secret value (stored encrypted at rest)'),
        description: z.string().optional(),
      },
      annotations: { readOnlyHint: false, idempotentHint: true },
    },
    async ({ name, value, description }) => {
      try {
        await pool.query(
          `INSERT INTO secrets.secrets (name, project_id, value_enc, description, updated_at)
           VALUES ($1,$2, pgp_sym_encrypt($3,$4), $5, now())
           ON CONFLICT (name) DO UPDATE SET
             value_enc = pgp_sym_encrypt($3,$4),
             description = COALESCE($5, secrets.secrets.description),
             updated_at = now()`,
          [name, projectId, value, config.masterKey, description ?? null],
        );
        return { content: [{ type: 'text', text: `stored secret "${name}"` }] };
      } catch (e) {
        return { isError: true, content: [{ type: 'text', text: `secret_set failed: ${(e as Error).message}` }] };
      }
    },
  );

  server.registerTool(
    'secret_get',
    {
      title: 'Decrypt and return a secret value',
      description: 'Return the decrypted value of a stored secret by name. Use when you need a credential to do a task.',
      inputSchema: { name: z.string() },
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async ({ name }) => {
      try {
        const { rows } = await pool.query<{ value: string }>(
          `SELECT pgp_sym_decrypt(value_enc, $2) AS value FROM secrets.secrets WHERE name=$1`,
          [name, config.masterKey],
        );
        if (rows.length === 0) return { isError: true, content: [{ type: 'text', text: `no secret "${name}"` }] };
        return { content: [{ type: 'text', text: rows[0].value }] };
      } catch (e) {
        return { isError: true, content: [{ type: 'text', text: `secret_get failed: ${(e as Error).message}` }] };
      }
    },
  );

  server.registerTool(
    'secret_list',
    {
      title: 'List secret names (never values)',
      description: 'List stored secret names + descriptions. Never returns values.',
      inputSchema: {},
      annotations: { readOnlyHint: true, openWorldHint: false },
    },
    async () => {
      const { rows } = await pool.query<{ name: string; description: string | null }>(
        `SELECT name, description FROM secrets.secrets WHERE project_id=$1 ORDER BY name`,
        [projectId],
      );
      const text = rows.length
        ? rows.map((r) => `- ${r.name}${r.description ? ` — ${r.description}` : ''}`).join('\n')
        : 'no secrets stored';
      return { content: [{ type: 'text', text }] };
    },
  );

  server.registerTool(
    'secret_delete',
    {
      title: 'Delete a secret',
      description: 'Permanently delete a stored secret by name.',
      inputSchema: { name: z.string() },
      annotations: { readOnlyHint: false, idempotentHint: true, destructiveHint: true },
    },
    async ({ name }) => {
      const { rowCount } = await pool.query(`DELETE FROM secrets.secrets WHERE name=$1`, [name]);
      return { content: [{ type: 'text', text: (rowCount ?? 0) > 0 ? `deleted "${name}"` : `no secret "${name}"` }] };
    },
  );

  return server;
}

assertServerConfig();
const app = express();
app.use(express.json({ limit: '1mb' }));
app.get('/healthz', (_req, res) => { res.json({ ok: true, service: 'secrets-mcp-server' }); });
app.use(originGuard);
app.use(requireBearer);
app.post('/mcp', async (req, res) => {
  const auth = (req as express.Request & { auth?: MnemoAuth }).auth;
  const projectId = auth?.extra.projectId ?? config.defaultProjectId;
  const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined, enableJsonResponse: true });
  res.on('close', () => void transport.close());
  const server = buildServer(projectId);
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});
app.listen(config.port, '0.0.0.0', () => {
  console.error(`secrets-mcp-server listening on :${config.port}`);
});
