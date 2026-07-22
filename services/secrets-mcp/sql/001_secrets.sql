-- Encrypted secrets store. Isolated schema: NO embeddings, never searched by the
-- memory engine. Values are pgp_sym_encrypt'd with a master key that lives ONLY
-- on the box (key file / env), never in the DB or backups. Idempotent.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS secrets;

CREATE TABLE IF NOT EXISTS secrets.secrets (
  name        text PRIMARY KEY,
  project_id  text NOT NULL DEFAULT 'default',
  value_enc   bytea NOT NULL,                       -- pgp_sym_encrypt(value, master_key)
  description text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS secrets_proj ON secrets.secrets (project_id);
