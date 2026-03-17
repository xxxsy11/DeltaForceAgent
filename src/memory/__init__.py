"""Session and persistent memory components."""

__all__ = ["SessionMemoryManager", "PersistentMemoryStore"]


def __getattr__(name: str):
    if name == "SessionMemoryManager":
        from .session_memory import SessionMemoryManager

        return SessionMemoryManager
    if name == "PersistentMemoryStore":
        from .persistent import PersistentMemoryStore

        return PersistentMemoryStore
    raise AttributeError(f"module 'memory' has no attribute {name!r}")
