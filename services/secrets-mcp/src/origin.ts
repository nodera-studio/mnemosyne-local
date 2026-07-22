import type { Request, Response, NextFunction } from 'express';
import { config } from './config.js';

/**
 * DNS-rebinding guard (MCP spec requires Origin validation on Streamable HTTP).
 * Mounted BEFORE auth, AFTER /healthz.
 *
 * - Browser/cross-site client sends an `Origin`: allow ONLY if it exactly matches
 *   the configured allowlist (ALLOWED_ORIGINS); otherwise 403.
 * - No `Origin` header (server-to-server / non-browser MCP clients — Anthropic's
 *   cloud, mcp-remote, Claude Code over loopback): intentionally ALLOWED here.
 *   These requests are never trusted on Origin alone — the downstream auth/OAuth
 *   gate is what actually protects them. Do NOT 403 the no-Origin case or you break
 *   every legitimate non-browser client, including Claude Code on localhost.
 *
 * secrets-mcp ships with an EMPTY allowlist (loopback-only): any present Origin is
 * rejected and only no-Origin loopback traffic passes.
 */
export function originGuard(req: Request, res: Response, next: NextFunction): void {
  const origin = req.headers.origin;
  if (origin !== undefined) {
    if (!config.allowedOrigins.includes(origin)) {
      res.status(403).json({ error: 'origin_not_allowed' });
      return;
    }
  }
  next();
}
