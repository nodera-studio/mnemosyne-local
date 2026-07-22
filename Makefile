.PHONY: secrets secrets-langfuse up down ps logs psql redis-cli verify langfuse-db langfuse-up

# Generate .env with strong random secrets (idempotent — won't clobber existing).
secrets:
	@if [ -f .env ]; then echo ".env already exists — not overwriting base secrets"; else \
		printf 'POSTGRES_DB=mnemosyne\nPOSTGRES_USER=mnemo\nPOSTGRES_PASSWORD=%s\nREDIS_PASSWORD=%s\n' \
			"$$(openssl rand -hex 24)" "$$(openssl rand -hex 24)" > .env; \
		chmod 600 .env; echo ".env created (chmod 600)"; fi
	@$(MAKE) --no-print-directory secrets-langfuse

# Append Langfuse v2 secrets to .env (idempotent — only adds keys that are missing).
# Safe to run on the already-deployed box; existing values are preserved.
secrets-langfuse:
	@touch .env; chmod 600 .env; \
	add() { grep -q "^$$1=" .env || { echo "$$1=$$2" >> .env; echo "  + $$1"; }; }; \
	add LANGFUSE_DB langfuse; \
	add LANGFUSE_NEXTAUTH_URL http://localhost:3000; \
	add LANGFUSE_NEXTAUTH_SECRET "$$(openssl rand -hex 32)"; \
	add LANGFUSE_SALT "$$(openssl rand -hex 16)"; \
	add LANGFUSE_ENCRYPTION_KEY "$$(openssl rand -hex 32)"; \
	add LANGFUSE_INIT_ORG_ID mnemosyne; \
	add LANGFUSE_INIT_PROJECT_ID mnemosyne-evals; \
	add LANGFUSE_INIT_PROJECT_PUBLIC_KEY "pk-lf-$$(uuidgen | tr 'A-Z' 'a-z')"; \
	add LANGFUSE_INIT_PROJECT_SECRET_KEY "sk-lf-$$(uuidgen | tr 'A-Z' 'a-z')"; \
	add LANGFUSE_INIT_USER_EMAIL dev@example.com; \
	add LANGFUSE_INIT_USER_PASSWORD "$$(openssl rand -hex 12)"; \
	add LANGFUSE_INIT_USER_NAME Mnemosyne; \
	echo "Langfuse secrets ensured in .env"

up:
	docker compose up -d postgres redis

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f

# psql into the running cluster using .env creds.
psql:
	@set -a; . ./.env; set +a; \
	docker compose exec -e PGPASSWORD=$$POSTGRES_PASSWORD postgres \
		psql -U $$POSTGRES_USER -d $$POSTGRES_DB

redis-cli:
	@set -a; . ./.env; set +a; \
	docker compose exec redis redis-cli -a $$REDIS_PASSWORD

# Verify pgvector + schemas are live.
verify:
	@set -a; . ./.env; set +a; \
	docker compose exec -e PGPASSWORD=$$POSTGRES_PASSWORD postgres \
		psql -U $$POSTGRES_USER -d $$POSTGRES_DB -c \
		"SELECT extname, extversion FROM pg_extension WHERE extname='vector'; \
		 SELECT nspname FROM pg_namespace WHERE nspname IN ('memory','codebase');"

# Create the langfuse database in the ALREADY-running cluster (init scripts only run on
# fresh pgdata). Idempotent. Run once before the first `make langfuse-up` on an existing box.
langfuse-db:
	@set -a; . ./.env; set +a; LFDB="$${LANGFUSE_DB:-langfuse}"; \
	if docker compose exec -e PGPASSWORD=$$POSTGRES_PASSWORD -T postgres \
		psql -U $$POSTGRES_USER -d $$POSTGRES_DB -tAc \
		"SELECT 1 FROM pg_database WHERE datname='$$LFDB'" | grep -q 1; then \
		echo "langfuse DB '$$LFDB' already exists"; \
	else \
		docker compose exec -e PGPASSWORD=$$POSTGRES_PASSWORD -T postgres \
			createdb -U $$POSTGRES_USER "$$LFDB" && echo "created langfuse DB '$$LFDB'"; \
	fi

# Bring up the eval plane (run after `make secrets-langfuse` + `make langfuse-db`).
# Langfuse runs its own Prisma migrations into the langfuse DB on first boot (~30-60s).
langfuse-up:
	docker compose up -d langfuse
