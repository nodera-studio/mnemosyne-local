#!/usr/bin/env bash
# Nightly backup of the Mnemosyne memory schema (pg_dump, gzip, 14-day rotation).
set -euo pipefail
cd /opt/mnemosyne
set -a; . ./.env; set +a
TS=$(date -u +%Y%m%d-%H%M%S)
OUT="/opt/mnemosyne/backups/memory-${TS}.sql.gz"
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -n memory </dev/null \
  | gzip -9 > "$OUT"
SZ=$(stat -c%s "$OUT")
if [ "$SZ" -lt 100000 ]; then echo "backup too small ($SZ B) — keeping but flagging"; fi
# rotate: keep newest 14 .sql.gz
ls -1t /opt/mnemosyne/backups/memory-*.sql.gz 2>/dev/null | tail -n +15 | xargs -r rm -f
echo "$(date -u +%FT%TZ) backup ok: $OUT ($((SZ/1024/1024))MB), $(ls /opt/mnemosyne/backups/memory-*.sql.gz | wc -l) kept"
