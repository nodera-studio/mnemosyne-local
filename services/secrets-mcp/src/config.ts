export const config = {
  databaseUrl: process.env.DATABASE_URL ?? "",
  masterKey: process.env.SECRETS_MASTER_KEY ?? "",
  port: parseInt(process.env.PORT ?? "8082", 10),
  bearerToken: process.env.MNEMO_TOKEN ?? "",
  defaultProjectId: process.env.DEFAULT_PROJECT_ID ?? "default",
  // Origin allowlist. secrets-mcp keeps this EMPTY (loopback-only): any present
  // Origin is rejected; only no-Origin loopback traffic passes.
  allowedOrigins: (process.env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean),
};

export function assertServerConfig(): void {
  if (!config.databaseUrl) throw new Error("DATABASE_URL required");
  if (!config.masterKey)
    throw new Error("SECRETS_MASTER_KEY required (box-local key)");
}
