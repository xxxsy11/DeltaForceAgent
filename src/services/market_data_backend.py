"""Market data backend abstraction.

Open-source repository只保留工具层和业务编排。
真实市场数据接入需要在外部提供私有 backend 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Protocol


class MarketDataBackend(Protocol):
    def query(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query market data by operation key.

        Expected return shape:
        {
            "success": bool,
            "data": Any,
            "error": str,
            "status_code": int,
            "endpoint": str,
            "tried": list,
        }
        """


@dataclass
class DisabledMarketDataBackend:
    reason: str

    def query(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "operation": operation,
            "params": params,
            "error": self.reason,
            "tried": [{"operation": operation, "error": self.reason}],
        }


def load_market_data_backend(config: Any) -> MarketDataBackend:
    module_path = str(getattr(config, "df_market_backend_module", "") or "").strip()
    class_name = str(getattr(config, "df_market_backend_class", "MarketDataBackendImpl") or "").strip()

    if not module_path:
        return DisabledMarketDataBackend(
            reason=(
                "市场数据后端未配置：请通过私有扩展实现并设置 "
                "DF_MARKET_BACKEND_MODULE / DF_MARKET_BACKEND_CLASS。"
            )
        )

    try:
        module = import_module(module_path)
    except Exception as exc:
        return DisabledMarketDataBackend(reason=f"加载市场数据后端失败: {exc}")

    backend_cls = getattr(module, class_name, None)
    if backend_cls is None:
        return DisabledMarketDataBackend(reason=f"市场数据后端类不存在: {module_path}.{class_name}")

    try:
        backend = backend_cls(config=config)
    except TypeError:
        try:
            backend = backend_cls()
        except Exception as exc:
            return DisabledMarketDataBackend(reason=f"初始化市场数据后端失败: {exc}")
    except Exception as exc:
        return DisabledMarketDataBackend(reason=f"初始化市场数据后端失败: {exc}")

    if not callable(getattr(backend, "query", None)):
        return DisabledMarketDataBackend(reason="市场数据后端缺少 query(operation, params) 方法")
    return backend
