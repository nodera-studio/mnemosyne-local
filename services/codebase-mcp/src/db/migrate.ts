import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { pool } from "./pool.js";

// dist/db/migrate.js  -> ../../sql ;  src/db/migrate.ts (tsx) -> ../../sql
const sqlDir = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "sql");

async function main(): Promise<void> {
  const files = readdirSync(sqlDir)
    .filter((f) => f.endsWith(".sql"))
    .sort();
  for (const f of files) {
    const text = readFileSync(join(sqlDir, f), "utf8");
    // HOLD migrations (e.g. 005_drop_bakeoff_scratch.sql) are ready-but-not-applied:
    // they are gated on a deliberate operator action and must NOT run in the batch
    // runner. They opt out with a `-- HOLD` marker on the first line.
    if (/^\s*--\s*HOLD\b/im.test(text.split("\n")[0] ?? "")) {
      console.log(`skipping ${f} (HOLD — apply manually)`);
      continue;
    }
    process.stdout.write(`applying ${f} ... `);
    await pool.query(text);
    console.log("ok");
  }
  await pool.end();
  console.log("migrations complete");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
