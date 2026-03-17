"""Delta Force 价格查询服务。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import requests

from observability.langsmith import langsmith_trace


def _split_paths(paths: str | Iterable[str]) -> List[str]:
    if isinstance(paths, str):
        raw = [item.strip() for item in paths.split(",")]
    else:
        raw = [str(item).strip() for item in paths]
    normalized: List[str] = []
    seen = set()
    for item in raw:
        if not item:
            continue
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    return normalized


@dataclass
class DFPriceService:
    DEFAULT_OBJECT_LOOKUP_LIMIT = 3000
    DEFAULT_TIMEOUT_SECONDS = 15
    OBJECT_LOOKUP_MIN_LIMIT = 1000
    OBJECT_ID_MIN_LENGTH = 8

    base_url: str
    token: str = ""
    latest_price_paths: str | Iterable[str] = "/df/object/price/latest"
    history_price_paths: str | Iterable[str] = "/df/object/price/history/v2"
    object_lookup_paths: str | Iterable[str] = "/df/object/price/latest/v3"
    place_profit_rank_paths: str | Iterable[str] = "/df/place/profitRank/v1"
    place_profit_history_paths: str | Iterable[str] = "/df/place/profitHistory"
    object_lookup_limit: int = DEFAULT_OBJECT_LOOKUP_LIMIT
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    trace_config: Any = field(default=None, repr=False)
    _object_id_cache: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_config(cls, config) -> "DFPriceService":
        return cls(
            base_url=getattr(config, "df_api_base_url", "https://df-api.shallow.ink"),
            token=getattr(config, "df_api_token", "") or "",
            latest_price_paths=getattr(
                config,
                "df_api_latest_price_paths",
                "/df/object/price/latest",
            ),
            history_price_paths=getattr(
                config,
                "df_api_history_price_paths",
                "/df/object/price/history/v2",
            ),
            object_lookup_paths=getattr(
                config,
                "df_api_object_lookup_paths",
                "/df/object/price/latest/v3",
            ),
            place_profit_rank_paths=getattr(
                config,
                "df_api_place_profit_rank_paths",
                "/df/place/profitRank/v1",
            ),
            place_profit_history_paths=getattr(
                config,
                "df_api_place_profit_history_paths",
                "/df/place/profitHistory",
            ),
            object_lookup_limit=int(
                getattr(config, "df_api_object_lookup_limit", cls.DEFAULT_OBJECT_LOOKUP_LIMIT)
            ),
            timeout_seconds=int(getattr(config, "df_api_timeout_seconds", cls.DEFAULT_TIMEOUT_SECONDS)),
            trace_config=config,
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_get(self, path: str, params: Dict[str, Any]) -> requests.Response:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        with langsmith_trace(
            self.trace_config,
            name="df_api_get",
            run_type="tool",
            inputs={"url": url, "path": path, "params": dict(params or {})},
            tags=["df-api", path.strip("/")],
            metadata={"base_url": self.base_url, "timeout_seconds": self.timeout_seconds},
        ):
            return requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=self.timeout_seconds,
            )

    def _request_first_success(self, paths: str | Iterable[str], params: Dict[str, Any]) -> Dict[str, Any]:
        with langsmith_trace(
            self.trace_config,
            name="df_api_request_first_success",
            run_type="tool",
            inputs={"paths": list(_split_paths(paths)), "params": dict(params or {})},
            tags=["df-api", "fallback"],
            metadata={"base_url": self.base_url},
        ):
            tried: List[Dict[str, Any]] = []
            last_error: Optional[str] = None
            for path in _split_paths(paths):
                try:
                    response = self._request_get(path, params)
                    content_type = response.headers.get("content-type", "")
                    payload: Any
                    if "application/json" in content_type.lower():
                        payload = response.json()
                    else:
                        payload = response.text

                    if 200 <= response.status_code < 300:
                        return {
                            "success": True,
                            "endpoint": path,
                            "status_code": response.status_code,
                            "params": params,
                            "data": payload,
                        }

                    tried.append({"path": path, "status_code": response.status_code, "response": payload})
                    last_error = f"HTTP {response.status_code}"
                except Exception as exc:
                    tried.append({"path": path, "error": str(exc)})
                    last_error = str(exc)

            return {
                "success": False,
                "params": params,
                "error": last_error or "全部候选路径调用失败",
                "tried": tried,
            }

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

    def _filter_latest_payload(self, payload: Any, params: Dict[str, Any]) -> Any:
        items = self._pick_latest_items(payload)
        if not items:
            return payload

        target_id = str(params.get("objectId") or params.get("id") or "").strip()
        target_name = self._norm_name(str(params.get("objectName") or params.get("name") or ""))
        if not target_id and not target_name:
            return payload

        def item_id(item: Dict[str, Any]) -> str:
            return str(item.get("objectID") or item.get("objectId") or item.get("id") or "").strip()

        def item_name(item: Dict[str, Any]) -> str:
            return self._norm_name(str(item.get("objectName") or item.get("name") or ""))

        matched: List[Dict[str, Any]] = []
        for item in items:
            iid = item_id(item)
            iname = item_name(item)
            if target_id and iid == target_id:
                matched.append(item)
                break
            if target_name and iname and (target_name in iname or iname in target_name):
                matched.append(item)
                break

        if not matched:
            return payload

        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            copied = dict(payload)
            copied_data = dict(payload["data"])
            copied_data["items"] = matched
            copied["data"] = copied_data
            copied["matchedBy"] = {"objectId": target_id, "objectName": params.get("objectName")}
            return copied
        return {"data": {"items": matched}, "matchedBy": {"objectId": target_id, "objectName": params.get("objectName")}}

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

    def resolve_object_id(self, object_name: str) -> Dict[str, Any]:
        target = str(object_name or "").strip()
        if not target:
            return {"success": False, "error": "objectName 为空"}
        cache_key = self._norm_name(target)
        if cache_key in self._object_id_cache:
            return {"success": True, **self._object_id_cache[cache_key], "cached": True}

        norm_target = self._norm_name(target)
        tried: List[Dict[str, Any]] = []

        for path in _split_paths(self.object_lookup_paths):
            params = {
                "objectName": target,
                "limit": str(
                    max(
                        self.OBJECT_LOOKUP_MIN_LIMIT,
                        int(self.object_lookup_limit or self.OBJECT_LOOKUP_MIN_LIMIT),
                    )
                ),
            }
            try:
                response = self._request_get(path, params)
                if not (200 <= response.status_code < 300):
                    tried.append({"path": path, "params": params, "status_code": response.status_code})
                    continue
                payload: Any
                content_type = response.headers.get("content-type", "")
                payload = response.json() if "application/json" in content_type.lower() else response.text

                # 某些物品在前 1000 条之外，若返回数量小于 totalCount，则拉取完整范围再匹配。
                items = self._pick_latest_items(payload)
                total_count = self._latest_payload_total(payload)
                if total_count and isinstance(items, list) and len(items) < total_count:
                    full_params = {"objectName": target, "limit": str(total_count)}
                    full_response = self._request_get(path, full_params)
                    if 200 <= full_response.status_code < 300:
                        full_content_type = full_response.headers.get("content-type", "")
                        payload = (
                            full_response.json()
                            if "application/json" in full_content_type.lower()
                            else full_response.text
                        )
                        params = full_params

                exact_match = None
                partial_match = None
                for node in self._iter_dict_nodes(payload):
                    oid = self._pick_id(node, min_length=self.OBJECT_ID_MIN_LENGTH)
                    if not oid:
                        continue
                    name = self._pick_name(node)
                    norm_name = self._norm_name(name)
                    candidate = {"id": oid, "objectName": name or target, "endpoint": path}
                    if norm_name and norm_name == norm_target:
                        exact_match = candidate
                        break
                    if norm_name and (norm_target in norm_name or norm_name in norm_target):
                        partial_match = partial_match or candidate

                best = exact_match or partial_match
                if best:
                    self._object_id_cache[cache_key] = best
                    return {"success": True, **best}

                tried.append({"path": path, "params": params, "status_code": response.status_code})
            except Exception as exc:
                tried.append({"path": path, "params": params, "error": str(exc)})

        return {"success": False, "error": f"未能根据 objectName 匹配到交易物品ID: {target}", "tried": tried}

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
        result = self._request_first_success(self.latest_price_paths, request_params)
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
        result = self._request_first_success(self.history_price_paths, request_params)
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
        return self._request_first_success(self.place_profit_rank_paths, request_params)

    def get_place_profit_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        request_params = self._normalize_params(params)
        if "place" in request_params:
            request_params["place"] = str(request_params["place"]).strip().lower()
        return self._request_first_success(self.place_profit_history_paths, request_params)
