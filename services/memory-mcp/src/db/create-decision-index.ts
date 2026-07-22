// Standalone op script (Wave 6): build the PARTIAL HNSW index for active decisions.
//
// CREATE INDEX CONCURRENTLY cannot run inside a transaction block, so — exactly like the
// Wave-P create-v2-index.ts — it is kept OUT of the batch migrate runner (which sends each
// sql/*.sql file as one pool.query) and run here as a single, standalone statement on its
// own connection. `pg` sends a lone statement in autocommit mode (no implicit BEGIN/COMMIT
// wrap), which is what CONCURRENTLY requires.
//
// The index is PARTIAL — only the active decision rows (source_kind='decision' AND
// decision_status='active') — so the decision corpus is a tiny, hot subset and non-decision
// rows never enter it. It vectors over embedding_v2 (the post-Wave-P canonical column).
//
// Run it AFTER `npm run migrate` has applied 005 (the ADD COLUMNs). It is idempotent
// (IF NOT EXISTS) and resumable: an interrupted CONCURRENTLY build may leave an INVALID
// index — drop it (DROP INDEX IF EXISTS memory.mem_decision_hnsw) and re-run.
//
//   DATABASE_URL=... npm run index:decision

import pg from "pg";
import { config } from "../config.js";

const STATEMENT = `CREATE INDEX CONCURRENTLY IF NOT EXISTS mem_decision_hnsw
  ON memory.memories USING hnsw (embedding_v2 halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE source_kind = 'decision' AND decision_status = 'active'`;

async function main(): Promise<void> {
  // A dedicated single-connection client (not the shared pool) — keeps the long
  // CONCURRENTLY build on one backend and avoids any pool-level transaction wrapping.
  const client = new pg.Client({ connectionString: config.databaseUrl });
  await client.connect();
  try {
    process.stdout.write("building mem_decision_hnsw (CONCURRENTLY) ... ");
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
