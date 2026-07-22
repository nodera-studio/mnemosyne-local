import type { Request, Response, NextFunction } from 'express';
import { config } from './config.js';

export interface MnemoAuth {
  token: string;
  clientId: string;
  scopes: string[];
  extra: { projectId: string };
}

/** Bearer auth — open when MNEMO_TOKEN unset (localhost-only); required when set. */
export function requireBearer(req: Request, res: Response, next: NextFunction): void {
  const auth: MnemoAuth = {
    token: 'local',
    clientId: 'local',
    scopes: ['secrets'],
    extra: { projectId: config.defaultProjectId },
  };
  if (config.bearerToken) {
    const presented = (req.headers.authorization ?? '').replace(/^Bearer\s+/i, '');
    if (presented !== config.bearerToken) {
      res.status(401).json({ error: 'invalid_token' });
      return;
    }
    auth.token = presented;
    auth.clientId = 'mnemo';
  }
  (req as Request & { auth?: MnemoAuth }).auth = auth;
  next();
}
