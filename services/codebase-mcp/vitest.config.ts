import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "test/**/*.test.ts"],
    // DB-backed tests share ONE Postgres; some DROP/CREATE the codebase schema in
    // beforeAll while others CREATE EXTENSION/apply migrations. Running test files in
    // parallel races on shared catalog objects (e.g. the pgvector type). Serialize
    // files — the suite is small and fast, so the cost is negligible.
    fileParallelism: false,
  },
});
