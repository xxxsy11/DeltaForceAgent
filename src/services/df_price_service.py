"""Delta Force 市场价格服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .market_data_backend import MarketDataBackend, load_market_data_backend


@dataclass
class DFPriceService:
    backend: MarketDataBackend
    latest_price_operation: str = "latest_price"
    history_price_operation: str = "history_price"
    object_lookup_operation: str = "object_lookup"
    place_profit_rank_operation: str = "place_profit_rank"
    place_profit_history_operation: str = "place_profit_history"
    object_lookup_limit: int = 3000
    _object_id_cache: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config) -> "DFPriceService":
        return cls(
            backend=load_market_data_backend(config),
            latest_price_operation=str(
                getattr(config, "df_market_latest_price_operation", "latest_price")
            ),
            history_price_operation=str(
                getattr(config, "df_market_history_price_operation", "history_price")
            ),
            object_lookup_operation=str(
                getattr(config, "df_market_object_lookup_operation", "object_lookup")
            ),
            place_profit_rank_operation=str(
                getattr(config, "df_market_place_profit_rank_operation", "place_profit_rank")
            ),
            place_profit_history_operation=str(
                getattr(config, "df_market_place_profit_history_operation", "place_profit_history")
            ),
            object_lookup_limit=int(getattr(config, "df_market_object_lookup_limit", 3000)),
        )

    @staticmethod
    def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in params.items():
            if value is None:
                continue
            text = str(value).strip()
            if text == "":
                continue
            normalized[key] = text
        return normalized

    @staticmethod
    def _normalize_object_id(params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(params)
        if "id" not in result and "objectId" in result:
            result["id"] = result["objectId"]
        if "id" in result and "objectId" not in result:
            result["objectId"] = result["id"]
        return result

    @staticmethod
    def _normalize_object_name(params: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(params)
        if "objectName" not in result and "name" in result:
            result["objectName"] = result["name"]
        return result

    @staticmethod
    def _iter_dict_nodes(payload: Any):
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from DFPriceService._iter_dict_nodes(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from DFPriceService._iter_dict_nodes(item)

    @staticmethod
    def _pick_id(node: Dict[str, Any], min_length: int = 1) -> Optional[str]:
        for key in ("id", "objectId", "objectID"):
            value = node.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text and len(text) >= min_length:
                return text
        return None

    @staticmethod
    def _pick_name(node: Dict[str, Any]) -> str:
        for key in ("objectName", "name"):
            value = node.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    @staticmethod
    def _norm_name(name: str) -> str:
        return "".join(str(name).lower().split())

    @staticmethod
    def _pick_latest_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return [x for x in data["items"] if isinstance(x, dict)]
            if isinstance(payload.get("items"), list):
                return [x for x in payload["items"] if isinstance(x, dict)]
        return []

    @staticmethod
    def _latest_payload_total(payload: Any) -> int:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                total = data.get("totalCount")
                if total is None:
                    total = data.get("total")
                if total is not None:
                    try:
                        return int(total)
                    except Exception:
                        return 0
        return 0

    def _request_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = self.backend.query(operation=operation, params=params)
        except Exception as exc:
            return {
                "success": False,
                "params": params,
                "error": f"后端调用异常: {exc}",
                "tried": [{"operation": operation, "error": str(exc)}],
            }

        if not isinstance(payload, dict):
            return {
                "success": False,
                "params": params,
                "error": "后端返回格式异常",
                "tried": [{"operation": operation, "error": "invalid response type"}],
            }

        if payload.get("success"):
            return {
                "success": True,
                "endpoint": payload.get("endpoint") or operation,
                "status_code": payload.get("status_code", 200),
                "params": params,
                "data": payload.get("data"),
            }

        tried = payload.get("tried") or [{"operation": operation, "error": payload.get("error", "unknown")}]
        return {
            "success": False,
            "params": params,
            "error": payload.get("error", "市场数据后端调用失败"),
            "tried": tried,
        }

    def resolve_object_id(self, object_name: str) -> Dict[str, Any]:
        target = str(object_name or "").strip()
        if not target:
            return {"success": False, "error": "objectName 为空"}

        cache_key = self._norm_name(target)
        if cache_key in self._object_id_cache:
            return {"success": True, **self._object_id_cache[cache_key], "cached": True}

        params = {
            "objectName": target,
            "limit": str(max(1000, int(self.object_lookup_limit or 1000))),
        }
        result = self._request_operation(self.object_lookup_operation, params)
        if not result.get("success"):
            return {
                "success": False,
                "error": f"未能根据 objectName 匹配到交易物品ID: {target}",
                "tried": result.get("tried", []),
            }

        payload = result.get("data")
        items = self._pick_latest_items(payload)
        total_count = self._latest_payload_total(payload)

        if total_count and isinstance(items, list) and len(items) < total_count:
            full_result = self._request_operation(
                self.object_lookup_operation,
                {"objectName": target, "limit": str(total_count)},
            )
            if full_result.get("success"):
                payload = full_result.get("data")

        norm_target = self._norm_name(target)
        exact_match = None
        partial_match = None
        for node in self._iter_dict_nodes(payload):
            oid = self._pick_id(node, min_length=8)
            if not oid:
                continue
            name = self._pick_name(node)
            norm_name = self._norm_name(name)
            candidate = {
                "id": oid,
                "objectName": name or target,
                "endpoint": result.get("endpoint", self.object_lookup_operation),
            }
            if norm_name and norm_name == norm_target:
                exact_match = candidate
                break
            if norm_name and (norm_target in norm_name or norm_name in norm_target):
                partial_match = partial_match or candidate

        best = exact_match or partial_match
        if best:
            self._object_id_cache[cache_key] = best
            return {"success": True, **best}

        return {
            "success": False,
            "error": f"未能根据 objectName 匹配到交易物品ID: {target}",
            "tried": result.get("tried", []),
        }

    def get_latest_price(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_object_id(self._normalize_object_name(self._normalize_params(params)))
        resolved: Dict[str, Any] = {}
        if "id" not in normalized:
            object_name = str(normalized.get("objectName", "")).strip()
            if not object_name:
                return {"success": False, "error": "缺少物品标识，请传入 id/objectId 或 objectName。"}
            resolved = self.resolve_object_id(object_name)
            if not resolved.get("success"):
                return {
                    "success": False,
                    "error": resolved.get("error", "根据 objectName 解析物品ID失败"),
                    "resolve_detail": resolved,
                }
            normalized["id"] = resolved["id"]
            normalized["objectId"] = resolved["id"]
            normalized["objectName"] = resolved.get("objectName", object_name)

        request_params = {"id": normalized["id"]}
        result = self._request_operation(self.latest_price_operation, request_params)
        if result.get("success"):
            result["resolved"] = {
                "objectId": normalized.get("id"),
                "objectName": normalized.get("objectName") or resolved.get("objectName"),
            }
        return result

    def get_history_price(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_object_id(self._normalize_object_name(self._normalize_params(params)))
        request_params = dict(normalized)

        if "objectId" not in request_params and "id" in request_params:
            request_params["objectId"] = request_params["id"]

        if "objectId" not in request_params and request_params.get("objectName"):
            resolved = self.resolve_object_id(request_params["objectName"])
            if resolved.get("success"):
                request_params["objectId"] = resolved["id"]
                request_params["objectName"] = resolved.get("objectName", request_params["objectName"])

        request_params.pop("id", None)
        result = self._request_operation(self.history_price_operation, request_params)
        if result.get("success"):
            result["resolved"] = {
                "objectId": request_params.get("objectId"),
                "objectName": request_params.get("objectName"),
            }
        return result

    def get_place_profit_rank(self, params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = self._normalize_params(params)
        if "place" in request_params:
            request_params["place"] = str(request_params["place"]).strip().lower()
        return self._request_operation(self.place_profit_rank_operation, request_params)

    def get_place_profit_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = self._normalize_params(params)
        if "place" in request_params:
            request_params["place"] = str(request_params["place"]).strip().lower()
        return self._request_operation(self.place_profit_history_operation, request_params)
