#!/usr/bin/env bash
set -euo pipefail

# 初始化本地 PostgreSQL 持久化记忆库（deltaforce_agent）
# 依赖：postgresql 服务已安装并启动、pgvector 扩展已安装

DB_USER="${1:-deltaforce_agent}"
DB_PASSWORD="${2:-deltaforce_agent}"
DB_NAME="${3:-deltaforce_agent}"

echo "[1/3] 创建用户（若不存在）: ${DB_USER}"
runuser -u postgres -- psql -v ON_ERROR_STOP=1 -c "DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}') THEN
    EXECUTE format('CREATE USER %I WITH PASSWORD %L', '${DB_USER}', '${DB_PASSWORD}');
  END IF;
END
\$\$;"

echo "[2/3] 创建数据库（若不存在）: ${DB_NAME}"
runuser -u postgres -- psql -v ON_ERROR_STOP=1 -c "DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname='${DB_NAME}') THEN
    EXECUTE format('CREATE DATABASE %I OWNER %I', '${DB_NAME}', '${DB_USER}');
  END IF;
END
\$\$;"

echo "[3/3] 启用 vector 扩展"
runuser -u postgres -- psql -d "${DB_NAME}" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "完成。可在 .env 中配置："
echo "MEMORY_PERSISTENT_DSN=postgresql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}"
