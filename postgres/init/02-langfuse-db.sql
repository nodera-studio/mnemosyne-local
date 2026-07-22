-- Runs once, on first cluster initialization (fresh pgdata only).
-- Creates the database that Langfuse v2 owns; Langfuse runs its own Prisma migrations
-- into it on first boot. For an ALREADY-initialized cluster (the deployed box), this
-- file does NOT re-run — use `make langfuse-db` instead.
SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
