"""Auto-split from persistent_memory_store.py."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

class PersistentSchemaMixin:
    def _ensure_schema(self) -> None:
        if self._schema_ready or not self.enabled:
            return
        conn = self._connect()
        if conn is None:
            return
        with conn:
            with conn.cursor() as cur:
                vector_available = False
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                except Exception:
                    # 扩展通常由 DBA 预装；失败不阻断业务
                    logger.warning("vector 扩展创建失败，继续尝试使用已有扩展", exc_info=False)
                try:
                    cur.execute("SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector');")
                    row = cur.fetchone()
                    vector_available = bool(row[0]) if row else False
                except Exception:
                    vector_available = False
                if not vector_available:
                    logger.warning("未检测到 vector 扩展，将以非向量模式运行长期记忆", exc_info=False)

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        meta JSONB NOT NULL DEFAULT '{}'::jsonb,
                        PRIMARY KEY (user_id, session_id)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_turns (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        turn_index INT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
                        content TEXT NOT NULL,
                        tool_name TEXT,
                        quality_score REAL NOT NULL DEFAULT 1.0,
                        is_failed BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_summaries (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        merge_count INT NOT NULL,
                        summary_text TEXT NOT NULL,
                        source_turn_start INT NOT NULL DEFAULT 0,
                        source_turn_end INT NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS memory_facts (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL DEFAULT 'default_user',
                        session_id TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        fact_value TEXT NOT NULL,
                        fact_type TEXT NOT NULL,
                        keywords TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                        confidence REAL NOT NULL DEFAULT 0.7,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        ttl_until TIMESTAMPTZ,
                        source_turn_id BIGINT,
                        embedding {"VECTOR(" + str(self.vector_dim) + ")" if vector_available else "TEXT"},
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                if vector_available:
                    try:
                        cur.execute(
                            """
                            SELECT udt_name
                            FROM information_schema.columns
                            WHERE table_name = 'memory_facts'
                              AND column_name = 'embedding'
                            LIMIT 1;
                            """
                        )
                        row = cur.fetchone()
                        current_udt = str(row[0]) if row and row[0] is not None else ""
                    except Exception:
                        current_udt = ""
                    if current_udt and current_udt != "vector":
                        try:
                            cur.execute(
                                f"""
                                ALTER TABLE memory_facts
                                ALTER COLUMN embedding TYPE VECTOR({self.vector_dim})
                                USING CASE
                                    WHEN embedding IS NULL THEN NULL
                                    ELSE embedding::vector
                                END;
                                """
                            )
                        except Exception:
                            logger.warning("embedding 列迁移到 vector 失败，回退非向量模式", exc_info=False)
                            vector_available = False
                # ---- schema migration: add user_id columns to old tables ----
                cur.execute(
                    """
                    ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE chat_sessions SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE chat_sessions ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_turns ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE chat_turns SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE chat_turns ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE memory_summaries ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE memory_summaries SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE memory_summaries ALTER COLUMN user_id SET NOT NULL;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE memory_facts ADD COLUMN IF NOT EXISTS user_id TEXT;
                    UPDATE memory_facts SET user_id = 'default_user' WHERE user_id IS NULL OR user_id = '';
                    ALTER TABLE memory_facts ALTER COLUMN user_id SET NOT NULL;
                    """
                )

                # ---- primary key migration for chat_sessions ----
                cur.execute(
                    """
                    DO $$
                    DECLARE pkey_name TEXT;
                    BEGIN
                      SELECT conname INTO pkey_name
                      FROM pg_constraint
                      WHERE conrelid = 'chat_sessions'::regclass
                        AND contype = 'p'
                      LIMIT 1;
                      IF pkey_name IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE chat_sessions DROP CONSTRAINT %I', pkey_name);
                      END IF;
                      ALTER TABLE chat_sessions ADD PRIMARY KEY (user_id, session_id);
                    EXCEPTION WHEN duplicate_object THEN
                      NULL;
                    END $$;
                    """
                )

                # ---- remove old uniqueness constraints ----
                cur.execute(
                    """
                    DO $$
                    DECLARE cname TEXT;
                    BEGIN
                      SELECT conname INTO cname
                      FROM pg_constraint
                      WHERE conrelid = 'memory_summaries'::regclass
                        AND contype = 'u'
                        AND pg_get_constraintdef(oid) LIKE '%(session_id, merge_count)%'
                      LIMIT 1;
                      IF cname IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE memory_summaries DROP CONSTRAINT %I', cname);
                      END IF;
                    END $$;
                    """
                )
                cur.execute(
                    """
                    DO $$
                    DECLARE cname TEXT;
                    BEGIN
                      SELECT conname INTO cname
                      FROM pg_constraint
                      WHERE conrelid = 'memory_facts'::regclass
                        AND contype = 'u'
                        AND pg_get_constraintdef(oid) LIKE '%(session_id, fact_key)%'
                      LIMIT 1;
                      IF cname IS NOT NULL THEN
                        EXECUTE format('ALTER TABLE memory_facts DROP CONSTRAINT %I', cname);
                      END IF;
                    END $$;
                    """
                )

                # ---- create indexes after migration ----
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_turns_user_session_turn ON chat_turns(user_id, session_id, turn_index DESC);"
                )
                cur.execute(
                    "DROP INDEX IF EXISTS idx_memory_summaries_session_turn_end_uniq;"
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX idx_memory_summaries_session_turn_end_uniq
                    ON memory_summaries(user_id, session_id, source_turn_end);
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_session_active ON memory_facts(user_id, session_id, is_active);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_facts_user_session_key ON memory_facts(user_id, session_id, fact_key);"
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_facts_user_ttl ON memory_facts(user_id, session_id, ttl_until);")

        self._vector_sql_ready = bool(vector_available)
        self._schema_ready = True
