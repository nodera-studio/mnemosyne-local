import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "test/**/*.test.ts"],
    // DB-backed tests share ONE Postgres; the golden/gate suites seed and truncate
    // fixture rows (and apply migrations idempotently in beforeAll). Running test
    // files in parallel races on shared rows and catalog objects. Serialize files —
    // the suite is small and fast, so the cost is negligible. (AC-601: mirrors
    // codebase-mcp's vitest.config.ts.)
    fileParallelism: false,
  },
});
