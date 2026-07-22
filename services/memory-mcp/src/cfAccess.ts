import { createRemoteJWKSet, jwtVerify } from 'jose';
import { config } from './config.js';

// Validate a Cloudflare Access JWT (Cf-Access-Jwt-Assertion). CF injects this header
// on every request it forwards from the Access-protected (Managed OAuth) edge — even
// when the MCP client holds an opaque token. We verify signature against the team's
// JWKS plus issuer + audience (the app's AUD tag). Returns null on any failure.
let jwksCache: ReturnType<typeof createRemoteJWKSet> | null = null;
function jwks() {
  if (!config.cfAccessTeamDomain) return null;
  if (!jwksCache) {
    jwksCache = createRemoteJWKSet(
      new URL(`https://${config.cfAccessTeamDomain}/cdn-cgi/access/certs`),
    );
  }
  return jwksCache;
}

export async function verifyAccessJwt(token: string): Promise<{ sub: string } | null> {
  const ks = jwks();
  if (!ks || !config.cfAccessAud) return null;
  try {
    const { payload } = await jwtVerify(token, ks, {
      issuer: `https://${config.cfAccessTeamDomain}`,
      audience: config.cfAccessAud,
    });
    return { sub: String(payload.sub ?? payload.email ?? 'cf-access') };
  } catch {
    return null;
  }
}
