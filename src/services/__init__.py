"""
业务服务层
"""

__all__ = ["RAGService", "DFPriceService"]


def __getattr__(name: str):
    if name == "RAGService":
        from .rag_service import RAGService

        return RAGService
    if name == "DFPriceService":
        from .df_price_service import DFPriceService

        return DFPriceService
    raise AttributeError(f"module 'services' has no attribute {name!r}")
