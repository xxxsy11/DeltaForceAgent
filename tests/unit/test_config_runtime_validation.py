from config import GraphRAGConfig, validate_runtime_config


def test_config_import_time_allows_missing_runtime_secrets():
    cfg = GraphRAGConfig(
        neo4j_password="",
        memory_persistent_enabled=False,
        memory_persistent_dsn="",
    )
    assert cfg.neo4j_password == ""
    assert cfg.memory_persistent_enabled is False


def test_validate_runtime_config_requires_neo4j_password():
    cfg = GraphRAGConfig(
        neo4j_password="",
        memory_persistent_enabled=False,
        memory_persistent_dsn="",
    )
    try:
        validate_runtime_config(cfg)
    except ValueError as exc:
        assert "NEO4J_PASSWORD" in str(exc)
    else:
        raise AssertionError("validate_runtime_config 应在缺少 NEO4J_PASSWORD 时失败")


def test_validate_runtime_config_requires_persistent_dsn_when_enabled():
    cfg = GraphRAGConfig(
        neo4j_password="secret",
        memory_persistent_enabled=True,
        memory_persistent_dsn="",
    )
    try:
        validate_runtime_config(cfg)
    except ValueError as exc:
        assert "MEMORY_PERSISTENT_DSN" in str(exc)
    else:
        raise AssertionError("validate_runtime_config 应在缺少 MEMORY_PERSISTENT_DSN 时失败")


def test_validate_runtime_config_accepts_complete_runtime_config():
    cfg = GraphRAGConfig(
        neo4j_password="secret",
        memory_persistent_enabled=True,
        memory_persistent_dsn="postgresql://user:pass@localhost:5432/db",
        langsmith_enabled=False,
    )
    validate_runtime_config(cfg)
