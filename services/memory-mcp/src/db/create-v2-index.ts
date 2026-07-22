// Standalone op script: build the HNSW index for the Wave-P contextual column.
//
// CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so it is kept OUT
// of the batch migrate runner (which sends each sql/*.sql file as one pool.query) and
// run here as a single, standalone statement on its own connection. `pg` sends a lone
// statement in autocommit mode (no implicit BEGIN/COMMIT wrap), which is exactly what
// CONCURRENTLY requires.
//
// Run it AFTER `npm run migrate` has applied 003 (the ADD COLUMN), and ideally AFTER
// the backfill has populated embedding_v2 (so the index build sees real data). It is
// idempotent (IF NOT EXISTS) and resumable: if a previous CONCURRENTLY build was
// interrupted it may leave an INVALID index — drop it (DROP INDEX IF EXISTS
// memory.mem_hnsw_v2) and re-run.
//
//   DATABASE_URL=... npm run index:v2

import pg from "pg";
import { config } from "../config.js";

const STATEMENT = `CREATE INDEX CONCURRENTLY IF NOT EXISTS mem_hnsw_v2
  ON memory.memories USING hnsw (embedding_v2 halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64)`;

async function main(): Promise<void> {
  // A dedicated single-connection client (not the shared pool) — keeps the long
  // CONCURRENTLY build on one backend and avoids any pool-level transaction wrapping.
  const client = new pg.Client({ connectionString: config.databaseUrl });
  await client.connect();
  try {
    process.stdout.write("building mem_hnsw_v2 (CONCURRENTLY) ... ");
    await client.query(STATEMENT);
    console.log("ok");
  } finally {
    await client.end();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
