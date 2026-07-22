import type { Request, Response, NextFunction } from 'express';
import { config } from './config.js';
import { verifyAccessJwt } from './cfAccess.js';

export interface MnemoAuth {
  token: string;
  clientId: string;
  scopes: string[];
  extra: { projectId: string };
}

/** Bearer auth — open when MNEMO_TOKEN unset (localhost-only); required when set. */
export async function requireBearer(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  // Public (Cloudflare Access) path: a signed Cf-Access-Jwt-Assertion is present.
  const cfJwt = req.headers['cf-access-jwt-assertion'];
  if (typeof cfJwt === 'string' && cfJwt.length > 0) {
    const verified = await verifyAccessJwt(cfJwt);
    if (!verified) {
      res.status(401).json({ error: 'invalid_access_token' });
      return;
    }
    (req as Request & { auth?: MnemoAuth }).auth = {
      token: cfJwt,
      clientId: 'cf-access',
      scopes: ['codebase'],
      extra: { projectId: config.defaultProjectId },
    };
    next();
    return;
  }

  const auth: MnemoAuth = {
    token: 'local',
    clientId: 'local',
    scopes: ['codebase'],
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
