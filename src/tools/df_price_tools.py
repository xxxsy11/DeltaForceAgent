"""Delta Force 价格查询工具。"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from services.df_price_service import DFPriceService


class DFPriceTools:
    """封装最新价与历史价查询工具。"""
    BEIJING_TZ = timezone(timedelta(hours=8))
    PLACE_PROFIT_GROUPS = [
        {
            "place": "tech",
            "label": "技术中心（枪械/配件）",
            "keywords": ("技术中心", "tech", "枪械", "枪", "步枪", "冲锋枪", "狙击枪", "配件", "枪械配件"),
        },
        {
            "place": "workbench",
            "label": "工作台（子弹）",
            "keywords": ("工作台", "workbench", "子弹", "弹药"),
        },
        {
            "place": "pharmacy",
            "label": "制药台（药品/针剂/维修工具包）",
            "keywords": ("制药台", "pharmacy", "药品", "药剂", "针剂", "针", "维修工具包"),
        },
        {
            "place": "armory",
            "label": "防具台（头盔/护甲/胸挂/背包）",
            "keywords": ("防具台", "armory", "头盔", "护甲", "胸挂", "背包", "防具"),
        },
    ]

    def __init__(self, price_service: DFPriceService, rag_service: Optional[Any] = None):
        self.price_service = price_service
        self.rag_service = rag_service

    @staticmethod
    def _parse_query_to_params(query: str) -> Dict[str, Any]:
        text = (query or "").strip()
        if not text:
            return {}

        # 1) JSON 输入
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        params: Dict[str, Any] = {}

        # 2) key=value / key:value 输入
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^\s,，]+)", text):
            params[key] = value

        # 3) 自动提取物品ID（长数字串，避免把年份识别成ID）
        if "id" not in params:
            ids = re.findall(r"\b\d{6,}\b", text)
            if ids:
                params["id"] = ids[0]

        # 4) 自动提取日期（历史查询可用）
        dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if dates and "date" not in params and "startTime" not in params and "endTime" not in params:
            if len(dates) == 1:
                params["date"] = dates[0]
            else:
                params["startTime"] = dates[0]
                params["endTime"] = dates[1]

        # 5) 自动提取 objectName（当没有 id 时）
        if "id" not in params and "objectName" not in params and "name" not in params:
            inferred = DFPriceTools._infer_object_name(text)
            if inferred:
                params["objectName"] = inferred

        return params

    @staticmethod
    def _infer_object_name(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        # 清理 key=value 片段
        cleaned = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*([^\s,，]+)", " ", raw)
        cleaned = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", cleaned)
        cleaned = re.sub(r"\b\d{4,}\b", " ", cleaned)

        stop_words = [
            "查一下", "查下", "查询", "帮我查", "帮我看", "请问", "告诉我", "看看",
            "最新", "实时", "历史", "走势", "曲线", "价格", "价位", "多少钱", "行情", "交易行",
            "现在", "当前", "目前", "什么",
            "的", "一下", "帮忙", "想知道", "到", "之间",
        ]
        for token in stop_words:
            cleaned = cleaned.replace(token, " ")

        cleaned = re.sub(r"[，。,.!?！？:：;；()（）\\[\\]{}\"']", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 太短或疑似纯指令词时放弃
        if len(cleaned) < 2:
            return ""
        return cleaned

    @staticmethod
    def _iter_nodes(payload: Any):
        if isinstance(payload, dict):
            yield payload
            for value in payload.values():
                yield from DFPriceTools._iter_nodes(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from DFPriceTools._iter_nodes(item)

    @staticmethod
    def _pick_first(payload: Any, keys: List[str]) -> Any:
        for node in DFPriceTools._iter_nodes(payload):
            for key in keys:
                if key in node and node[key] not in (None, ""):
                    return node[key]
        return None

    @staticmethod
    def _pick_records(payload: Any) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for node in DFPriceTools._iter_nodes(payload):
            has_price_key = any(k in node for k in ("avgPrice", "price", "latestPrice", "lastPrice", "unitPrice"))
            has_time_key = any(k in node for k in ("currentTime", "time", "date", "updateTime", "updatedAt", "timestamp", "ts"))
            if has_price_key or has_time_key:
                records.append(node)
        return records

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _pick_record_time(record: Dict[str, Any]) -> Any:
        for key in (
            "currentTime",
            "updateTime",
            "updatedAt",
            "time",
            "date",
            "timestamp",
            "ts",
            "recordTime",
            "createdAt",
        ):
            value = record.get(key)
            if value not in (None, ""):
                return value
        return None

    @classmethod
    def _format_beijing_time(cls, value: Any) -> str:
        if value is None:
            return ""

        timestamp: float | None = None
        if isinstance(value, (int, float)):
            number = float(value)
            # 13位通常是毫秒时间戳，转换成秒。
            timestamp = number / 1000 if abs(number) >= 1e12 else number
        else:
            text = str(value).strip()
            if not text:
                return ""

            if re.fullmatch(r"\d{10,13}", text):
                number = float(text)
                # 13位时间戳按毫秒处理
                timestamp = number / 1000 if len(text) >= 13 else number
            else:
                iso_text = text.replace("Z", "+00:00")
                try:
                    dt = datetime.fromisoformat(iso_text)
                except Exception:
                    return text

                # 无时区字符串默认视为北京时间，不做偏移。
                if dt.tzinfo is None:
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                return dt.astimezone(cls.BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")

        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(cls.BEIJING_TZ)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value)

    def _format_latest_result(self, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            tried = result.get("tried") or result.get("resolve_detail", {}).get("tried") or []
            detail_lines = []
            for item in tried[:5]:
                path = item.get("path", "?")
                status = item.get("status_code", item.get("error", "unknown"))
                detail_lines.append(f"- {path}: {status}")
            detail = "\n".join(detail_lines)
            if detail:
                return f"查询失败：{result.get('error', '未知错误')}\n候选接口尝试：\n{detail}"
            return f"查询失败：{result.get('error', '未知错误')}"

        payload = result.get("data")
        object_name = self._pick_first(payload, ["objectName", "name"])
        if not object_name:
            object_name = result.get("resolved", {}).get("objectName") or result.get("params", {}).get("objectName")
        object_id = self._pick_first(payload, ["id", "objectId", "objectID"]) or result.get("params", {}).get("id")
        price = self._pick_first(payload, ["avgPrice", "latestPrice", "lastPrice", "price", "unitPrice"])
        update_time_raw = self._pick_first(payload, ["currentTime", "updateTime", "updatedAt", "time", "date", "timestamp", "ts"])
        update_time = self._format_beijing_time(update_time_raw)

        if price is None:
            return f"已调用最新价接口（{result.get('endpoint')}），但未解析到价格字段。\n{json.dumps(result, ensure_ascii=False)}"

        name_text = str(object_name or f"ID {object_id}")
        msg = f"{name_text} 的最新价格为 {price}。"
        if update_time:
            msg += f" 更新时间：{update_time}。"
        return msg

    def _extract_latest_price_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "最新价查询失败"), "raw": result}

        payload = result.get("data")
        object_name = self._pick_first(payload, ["objectName", "name"])
        if not object_name:
            object_name = result.get("resolved", {}).get("objectName") or result.get("params", {}).get("objectName")

        price_raw = self._pick_first(payload, ["avgPrice", "latestPrice", "lastPrice", "price", "unitPrice"])
        price = self._to_float(price_raw)
        update_time = self._format_beijing_time(
            self._pick_first(payload, ["currentTime", "updateTime", "updatedAt", "time", "date", "timestamp", "ts"])
        )

        if price is None:
            return {"success": False, "error": "未解析到最新价格字段", "raw": result}

        return {
            "success": True,
            "object_name": str(object_name or ""),
            "price": price,
            "update_time": update_time,
        }

    @staticmethod
    def _extract_history_points(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, dict):
            return []

        history = data.get("history")
        if isinstance(history, list):
            return [x for x in history if isinstance(x, dict)]

        items = data.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                history = item.get("history")
                if isinstance(history, list):
                    return [x for x in history if isinstance(x, dict)]
        return []

    def _format_history_result(self, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            tried = result.get("tried") or result.get("resolve_detail", {}).get("tried") or []
            detail_lines = []
            for item in tried[:5]:
                path = item.get("path", "?")
                status = item.get("status_code", item.get("error", "unknown"))
                detail_lines.append(f"- {path}: {status}")
            detail = "\n".join(detail_lines)
            if detail:
                return f"查询失败：{result.get('error', '未知错误')}\n候选接口尝试：\n{detail}"
            return f"查询失败：{result.get('error', '未知错误')}"

        payload = result.get("data")
        object_name = self._pick_first(payload, ["objectName", "name"])
        if not object_name:
            object_name = result.get("resolved", {}).get("objectName") or result.get("params", {}).get("objectName")
        object_id = self._pick_first(payload, ["id", "objectId", "objectID"]) or result.get("params", {}).get("id")
        records = self._extract_history_points(payload) or self._pick_records(payload)

        prices: List[float] = []
        priced_records: List[Dict[str, Any]] = []
        for item in records:
            value = (
                item.get("avgPrice")
                or item.get("price")
                or item.get("latestPrice")
                or item.get("lastPrice")
                or item.get("unitPrice")
            )
            number = self._to_float(value)
            if number is not None:
                prices.append(number)
                priced_records.append(item)

        if not records:
            return f"已调用历史价接口（{result.get('endpoint')}），但返回中没有可用历史记录。\n{json.dumps(result, ensure_ascii=False)}"

        name_text = str(object_name or f"ID {object_id}")
        count = len(records)
        if not prices:
            return f"{name_text} 历史价格查询成功，共 {count} 条记录。"

        latest = prices[0]
        oldest = prices[-1]
        min_price = min(prices)
        max_price = max(prices)
        latest_time = self._format_beijing_time(self._pick_record_time(priced_records[0])) if priced_records else ""
        oldest_time = self._format_beijing_time(self._pick_record_time(priced_records[-1])) if priced_records else ""

        latest_part = f"最新样本价：{latest}"
        oldest_part = f"最早样本价：{oldest}"
        if latest_time:
            latest_part += f"（{latest_time}）"
        if oldest_time:
            oldest_part += f"（{oldest_time}）"
        return (
            f"{name_text} 历史价格查询成功，共 {count} 条记录。"
            f"{latest_part}，{oldest_part}，区间最低：{min_price}，区间最高：{max_price}。"
        )

    @staticmethod
    def _format_number(value: float) -> str:
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"

    def _extract_cost_price(self, query: str, params: Dict[str, Any]) -> float | None:
        for key in ("buyPrice", "costPrice", "entryPrice", "holdPrice", "cost", "buy_price", "cost_price"):
            if key in params:
                value = self._to_float(params.get(key))
                if value is not None and value > 0:
                    return value

        text = str(query or "")
        patterns = [
            r"(?:买入价|成本价|成本|持仓价|入手价|买在|买入)\s*[:=：]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:买入|入手)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = self._to_float(match.group(1))
            if value is not None and value > 0:
                return value
        return None

    @staticmethod
    def _build_advice(position: float) -> Dict[str, str]:
        if position >= 0.8:
            return {
                "level": "高位",
                "buy": "当前处于历史偏高区，谨慎追高，建议等待回调再买。",
                "sell": "若你已持有，当前更适合分批止盈或减仓。",
            }
        if position <= 0.2:
            return {
                "level": "低位",
                "buy": "当前处于历史偏低区，可考虑分批布局。",
                "sell": "若你已持有，不建议在低位恐慌卖出。",
            }
        return {
            "level": "中位",
            "buy": "当前处于历史中位区，建议小仓位分批，不要重仓追买。",
            "sell": "当前不属于明显高位，是否卖出取决于你的持仓成本和周期。",
        }

    def _format_price_advice_result(self, query: str, latest_result: Dict[str, Any], history_result: Dict[str, Any]) -> str:
        latest_info = self._extract_latest_price_info(latest_result)
        if not latest_info.get("success"):
            return f"价格策略分析失败：{latest_info.get('error', '最新价格获取失败')}"

        payload = history_result.get("data")
        records = self._extract_history_points(payload) or self._pick_records(payload)
        prices: List[float] = []
        for item in records:
            value = (
                item.get("avgPrice")
                or item.get("price")
                or item.get("latestPrice")
                or item.get("lastPrice")
                or item.get("unitPrice")
            )
            number = self._to_float(value)
            if number is not None:
                prices.append(number)

        if not prices:
            return "价格策略分析失败：历史价格为空，无法计算区间与买卖建议。"

        current_price = float(latest_info["price"])
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        band = max_price - min_price
        position = 0.5 if band <= 0 else max(0.0, min(1.0, (current_price - min_price) / band))
        advice = self._build_advice(position)

        object_name = latest_info.get("object_name") or "该物品"
        update_time = latest_info.get("update_time")

        rise_to_high = max(0.0, max_price - current_price)
        drawdown_to_low = max(0.0, current_price - min_price)

        lines = [
            f"{object_name} 价格策略分析：",
            f"- 当前价格：{self._format_number(current_price)}" + (f"（更新时间：{update_time}）" if update_time else ""),
            f"- 历史区间：{self._format_number(min_price)} ~ {self._format_number(max_price)}（样本 {len(prices)} 条，均价 {self._format_number(avg_price)}）",
            f"- 当前区间位置：{position * 100:.1f}%（{advice['level']}）",
            f"- 买入建议：{advice['buy']}",
            f"- 卖出建议：{advice['sell']}",
            f"- 区间测算：若回到区间最高，每件理论上行 {self._format_number(rise_to_high)}；若回落到区间最低，每件理论回撤 {self._format_number(drawdown_to_low)}。",
        ]

        params = self._parse_query_to_params(query)
        cost_price = self._extract_cost_price(query=query, params=params)
        if cost_price is not None and cost_price > 0:
            pnl = current_price - cost_price
            pnl_ratio = (pnl / cost_price) * 100
            status = "浮盈" if pnl >= 0 else "浮亏"
            lines.append(
                f"- 持仓测算：按你的成本 {self._format_number(cost_price)}，当前每件{status} {self._format_number(abs(pnl))}（{pnl_ratio:+.2f}%）。"
            )

        return "\n".join(lines)

    @staticmethod
    def _pick_price_value(item: Dict[str, Any]) -> Any:
        return (
            item.get("avgPrice")
            or item.get("price")
            or item.get("latestPrice")
            or item.get("lastPrice")
            or item.get("unitPrice")
        )

    def _extract_market_history_stats(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "历史价格查询失败")}

        payload = result.get("data")
        records = self._extract_history_points(payload) or self._pick_records(payload)
        prices: List[float] = []
        for item in records:
            number = self._to_float(self._pick_price_value(item))
            if number is not None:
                prices.append(number)

        if not prices:
            return {"success": False, "error": "历史记录为空"}

        latest_time = self._format_beijing_time(self._pick_record_time(records[0])) if records else ""
        oldest_time = self._format_beijing_time(self._pick_record_time(records[-1])) if records else ""

        return {
            "success": True,
            "count": len(prices),
            "latest": prices[0],
            "oldest": prices[-1],
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "latest_time": latest_time,
            "oldest_time": oldest_time,
        }

    @staticmethod
    def _unique_keep_order(values: List[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            item = str(value or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _is_name_consistent(requested: str, resolved: str) -> bool:
        lhs = "".join(str(requested or "").lower().split())
        rhs = "".join(str(resolved or "").lower().split())
        if not lhs or not rhs:
            return False
        if lhs == rhs:
            return True
        if len(lhs) >= 2 and (lhs in rhs or rhs in lhs):
            return True
        return False

    @classmethod
    def _extract_item_names(cls, query: str, params: Dict[str, Any]) -> List[str]:
        if isinstance(params.get("items"), list):
            return cls._unique_keep_order([str(x).strip() for x in params.get("items", []) if str(x).strip()])

        if isinstance(params.get("objectNames"), list):
            return cls._unique_keep_order([str(x).strip() for x in params.get("objectNames", []) if str(x).strip()])

        merged = []
        if params.get("objectName"):
            merged.append(str(params.get("objectName")))
        if params.get("name"):
            merged.append(str(params.get("name")))
        if merged:
            tokens = re.split(r"[，,、/|\s]+|和|与|以及|并且|还有|对比|比较", " ".join(merged))
            return cls._unique_keep_order([t.strip() for t in tokens if t.strip()])

        text = str(query or "").strip()
        if not text:
            return []

        clean_text = text
        noisy_tokens = [
            "请帮我", "帮我", "查一下", "查下", "查询", "对比", "比较", "哪个", "哪一个",
            "更值得买", "值得买", "怎么买", "介绍一下", "介绍", "告诉我", "并告诉我", "并且告诉我",
            "是否建议买", "是否建议卖", "建议买", "建议卖", "利润", "价格", "历史", "现在", "当前", "目前",
            "特勤处", "制造", "最高", "前三", "top1", "top3",
        ]
        for token in noisy_tokens:
            clean_text = clean_text.replace(token, " ")

        parts = re.split(r"[，,、/|\n]+|和|与|以及|并且|还有", clean_text)
        candidates = []
        for part in parts:
            item = re.sub(r"\s+", " ", part).strip(" ：:;；。.")
            if len(item) >= 2:
                candidates.append(item)

        if not candidates:
            inferred = cls._infer_object_name(text)
            if inferred:
                candidates.append(inferred)
        return cls._unique_keep_order(candidates)

    @classmethod
    def _extract_primary_item_name(cls, query: str, params: Dict[str, Any]) -> str:
        def _normalize_name(raw: str) -> str:
            name = str(raw or "").strip()
            prefix_tokens = [
                "分析一下",
                "分析",
                "介绍一下",
                "介绍",
                "查询一下",
                "查询",
                "查一下",
                "查下",
                "告诉我",
                "帮我看",
            ]
            for token in prefix_tokens:
                if name.startswith(token) and len(name) > len(token):
                    name = name[len(token):].strip()
            suffix_tokens = [
                "利润稳定性",
                "稳定性",
                "利润波动",
                "波动",
                "回撤",
                "风险",
                "利润",
                "价格",
                "历史价格",
            ]
            for token in suffix_tokens:
                if name.endswith(token) and len(name) > len(token):
                    name = name[: -len(token)].strip()
            name = name.strip(" ：:，,。.")
            return name

        if params.get("objectName"):
            return _normalize_name(str(params.get("objectName")).strip())
        if params.get("name"):
            return _normalize_name(str(params.get("name")).strip())

        text = str(query or "").strip()
        patterns = [
            r"(?:介绍一下|介绍|查询一下|查询|查一下|分析一下|分析|说说)\s*([^\s，,。；;并和与以及]+)",
            r"(?:告诉我|帮我看)\s*([^\s，,。；;并和与以及]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            name = _normalize_name(str(match.group(1)).strip())
            if name:
                return name

        candidates = cls._extract_item_names(query=query, params=params)
        for name in candidates:
            normalized = _normalize_name(name)
            if normalized and normalized not in {"介绍", "查询", "价格", "利润", "建议", "现在", "当前"}:
                return normalized
        return ""

    def _extract_profit_history_records(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "利润历史查询失败")}

        payload = result.get("data")
        if not isinstance(payload, dict):
            return {"success": False, "error": "利润历史返回结构异常"}

        data = payload.get("data")
        if not isinstance(data, dict):
            return {"success": False, "error": "利润历史返回结构异常"}

        if isinstance(data.get("history"), list):
            history = [x for x in data.get("history", []) if isinstance(x, dict)]
            return {
                "success": True,
                "object_name": str(data.get("objectName") or ""),
                "place_name": str(data.get("placeName") or ""),
                "history": history,
            }

        items = data.get("items")
        if isinstance(items, list) and items:
            first = next((x for x in items if isinstance(x, dict)), None)
            if first and isinstance(first.get("history"), list):
                history = [x for x in first.get("history", []) if isinstance(x, dict)]
                return {
                    "success": True,
                    "object_name": str(first.get("objectName") or ""),
                    "place_name": str(first.get("placeName") or ""),
                    "history": history,
                }

        return {"success": False, "error": "未找到利润历史数据"}

    def _format_multi_item_compare_result(self, query: str, params: Dict[str, Any]) -> str:
        names = self._extract_item_names(query=query, params=params)
        if len(names) < 2:
            return "请至少提供两个物品名称用于对比，例如：非洲之心、海洋之泪对比。"

        results = []
        failed = []
        for name in names[:6]:
            latest_result = self.price_service.get_latest_price({"objectName": name})
            latest_info = self._extract_latest_price_info(latest_result)
            if not latest_info.get("success"):
                failed.append(f"{name}: {latest_info.get('error', '最新价查询失败')}")
                continue
            resolved_name = str(latest_info.get("object_name") or "")
            if resolved_name and not self._is_name_consistent(name, resolved_name):
                failed.append(f"{name}: 匹配结果异常（返回为 {resolved_name}）")
                continue

            history_result = self.price_service.get_history_price({"objectName": name})
            history_stats = self._extract_market_history_stats(history_result)
            if not history_stats.get("success"):
                failed.append(f"{name}: {history_stats.get('error', '历史价查询失败')}")
                continue

            current = float(latest_info["price"])
            min_price = float(history_stats["min"])
            max_price = float(history_stats["max"])
            band = max_price - min_price
            position = 0.5 if band <= 0 else max(0.0, min(1.0, (current - min_price) / band))
            upside = max(0.0, max_price - current)
            downside = max(0.0, current - min_price)
            score = upside - 0.8 * downside
            results.append(
                {
                    "name": latest_info.get("object_name") or name,
                    "current": current,
                    "min": min_price,
                    "max": max_price,
                    "position": position,
                    "upside": upside,
                    "downside": downside,
                    "score": score,
                    "update_time": latest_info.get("update_time", ""),
                }
            )

        if not results:
            detail = "；".join(failed) if failed else "未拿到可用数据"
            return f"对比失败：{detail}"

        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        lines = [f"多物品价格对比（共{len(ranked)}个）"]
        for index, item in enumerate(ranked, 1):
            lines.append(
                f"{index}. {item['name']}｜现价 {self._format_number(item['current'])}｜区间 {self._format_number(item['min'])}~{self._format_number(item['max'])}"
                f"｜区间位置 {item['position'] * 100:.1f}%｜上行空间 {self._format_number(item['upside'])}｜下行风险 {self._format_number(item['downside'])}"
            )
        best = ranked[0]
        lines.append(
            f"结论：当前性价比相对更优的是 {best['name']}（区间位置 {best['position'] * 100:.1f}%）。"
        )
        if failed:
            lines.append("未成功对比：" + "；".join(failed[:4]))
        return "\n".join(lines)

    @staticmethod
    def _classify_profit_stability(positive_ratio: float, cv: float, max_drawdown: float) -> str:
        if positive_ratio >= 0.8 and cv <= 0.5 and max_drawdown <= 0.35:
            return "高"
        if positive_ratio >= 0.6 and cv <= 0.9 and max_drawdown <= 0.6:
            return "中"
        return "低"

    def _format_profit_stability_result(self, query: str, params: Dict[str, Any]) -> str:
        request_params = dict(params)
        if "objectId" not in request_params and "id" in request_params:
            request_params["objectId"] = request_params["id"]
        if "objectId" not in request_params:
            inferred = self._extract_primary_item_name(query=query, params=request_params)
            if inferred:
                request_params["objectName"] = inferred

        result = self.price_service.get_place_profit_history(request_params)
        parsed = self._extract_profit_history_records(result)
        if not parsed.get("success"):
            return f"利润稳定性分析失败：{parsed.get('error', '利润历史查询失败')}"

        history = parsed.get("history", [])
        if not history:
            return "利润稳定性分析失败：利润历史为空。"

        profits = []
        timestamps = []
        for item in history:
            number = self._to_float(item.get("profit"))
            if number is None:
                continue
            profits.append(number)
            timestamps.append(item.get("timestamp"))

        if len(profits) < 2:
            return "利润稳定性分析失败：有效利润样本不足（至少需要2条）。"

        avg_profit = sum(profits) / len(profits)
        variance = sum((x - avg_profit) ** 2 for x in profits) / len(profits)
        std_profit = math.sqrt(variance)
        cv = (std_profit / abs(avg_profit)) if avg_profit != 0 else float("inf")
        positive_ratio = sum(1 for x in profits if x > 0) / len(profits)

        # 采用时间从旧到新计算最大回撤
        chrono_profits = list(reversed(profits))
        peak = chrono_profits[0]
        max_drawdown = 0.0
        for value in chrono_profits:
            if value > peak:
                peak = value
            if peak != 0:
                drawdown = max(0.0, (peak - value) / abs(peak))
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        latest_profit = profits[0]
        oldest_profit = profits[-1]
        trend = ((latest_profit - oldest_profit) / abs(oldest_profit)) if oldest_profit != 0 else 0.0
        stability = self._classify_profit_stability(
            positive_ratio=positive_ratio,
            cv=cv if math.isfinite(cv) else 999.0,
            max_drawdown=max_drawdown,
        )

        latest_time = self._format_beijing_time(timestamps[0]) if timestamps else ""
        oldest_time = self._format_beijing_time(timestamps[-1]) if timestamps else ""
        name = parsed.get("object_name") or request_params.get("objectName") or request_params.get("objectId") or "该物品"
        place_name = parsed.get("place_name", "")

        if stability == "高":
            advice = "利润波动较小且正利润占比高，可作为稳定制造选项。"
        elif stability == "中":
            advice = "利润有一定波动，建议控制产量并关注成本端变化。"
        else:
            advice = "利润波动较大或负利润占比偏高，建议谨慎生产。"

        lines = [
            f"{name} 利润稳定性分析" + (f"（{place_name}）" if place_name else ""),
            f"- 样本数：{len(profits)}",
            f"- 最新利润：{self._format_number(latest_profit)}" + (f"（{latest_time}）" if latest_time else ""),
            f"- 最早利润：{self._format_number(oldest_profit)}" + (f"（{oldest_time}）" if oldest_time else ""),
            f"- 平均利润：{self._format_number(avg_profit)}，利润标准差：{self._format_number(std_profit)}，波动系数CV：{cv:.2f}",
            f"- 正利润占比：{positive_ratio * 100:.1f}%，最大回撤：{max_drawdown * 100:.1f}%，趋势变化：{trend * 100:+.1f}%",
            f"- 稳定性评级：{stability}",
            f"- 建议：{advice}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        lowered = text.lower()
        return any((k in text) or (k in lowered) for k in keywords)

    def _format_answer_composer_result(self, query: str, params: Dict[str, Any]) -> str:
        if self.rag_service is None:
            return "综合回答工具不可用：RAG 服务未注入。"

        primary_name = self._extract_primary_item_name(query=query, params=params)
        price_query = primary_name or query

        knowledge_text = ""
        try:
            knowledge = self.rag_service.query(question=query, explain_routing=False)
            knowledge_text = str(knowledge.get("answer", "")).strip()
        except Exception as exc:
            knowledge_text = f"知识检索失败：{exc}"

        latest_text = ""
        advice_text = ""
        if primary_name:
            latest_result = self.price_service.get_latest_price({"objectName": price_query})
            latest_info = self._extract_latest_price_info(latest_result)
            history_result = self.price_service.get_history_price({"objectName": price_query})
            is_match = latest_info.get("success") and self._is_name_consistent(primary_name, str(latest_info.get("object_name") or ""))
            if is_match:
                latest_text = self._format_latest_result(latest_result)
            else:
                latest_text = f"{primary_name} 价格查询失败：未命中准确物品。"

            if history_result.get("success") and is_match:
                advice_text = self._format_price_advice_result(
                    query=price_query,
                    latest_result=latest_result,
                    history_result=history_result,
                )
            else:
                advice_text = self._format_history_result(history_result)

        need_profit_rank = self._contains_any(
            query,
            ["制造", "特勤处", "利润", "枪械配件", "子弹", "药品", "针剂", "防具台", "技术中心", "工作台", "制药台"],
        )
        profit_rank_text = self._format_place_profit_rank_result(query=query, params=params) if need_profit_rank else ""

        sections = ["综合回答："]
        if knowledge_text:
            sections.append("【资料介绍】\n" + knowledge_text)
        if latest_text:
            sections.append("【实时价格】\n" + latest_text)
        if advice_text:
            sections.append("【买卖建议】\n" + advice_text)
        if profit_rank_text:
            sections.append("【制造利润】\n" + profit_rank_text)

        return "\n\n".join(sections).strip()

    @staticmethod
    def _normalize_top_n(value: Any) -> int:
        try:
            number = int(str(value).strip())
        except Exception:
            return 3
        return 1 if number <= 1 else 3

    @classmethod
    def _detect_profit_top_n(cls, query: str, params: Dict[str, Any]) -> int:
        for key in ("top", "limit", "n", "topn"):
            if key in params:
                return cls._normalize_top_n(params.get(key))

        text = str(query or "")
        lowered = text.lower()
        m = re.search(r"top\s*([1-9])", lowered)
        if m:
            return cls._normalize_top_n(m.group(1))
        m = re.search(r"前\s*([1-9])", text)
        if m:
            return cls._normalize_top_n(m.group(1))

        if any(token in lowered for token in ("top3", "前三", "前3")):
            return 3
        if any(token in lowered for token in ("top1",)) or any(
            token in text for token in ("第一", "最高", "最赚钱", "利润最高", "净利润最高")
        ):
            return 1
        return 3

    @classmethod
    def _detect_profit_rank_type(cls, query: str, params: Dict[str, Any]) -> str:
        allowed = {"hour", "total", "hourprofit", "totalprofit"}
        configured = str(params.get("type", "")).strip().lower()
        if configured in allowed:
            return configured

        text = str(query or "")
        if "小时" in text or "每小时" in text:
            return "hourprofit"
        return "totalprofit"

    @classmethod
    def _detect_profit_places(cls, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        place_to_group = {item["place"]: item for item in cls.PLACE_PROFIT_GROUPS}
        selected: List[Dict[str, Any]] = []

        def _add(place: str):
            group = place_to_group.get(place)
            if group and group not in selected:
                selected.append(group)

        raw_place = str(params.get("place", "")).strip().lower()
        if raw_place:
            for token in re.split(r"[,，\s]+", raw_place):
                _add(token)
            if selected:
                return selected

        raw_group = str(params.get("group", "")).strip().lower()
        if raw_group:
            alias = {
                "gun": "tech",
                "guns": "tech",
                "firearm": "tech",
                "firearms": "tech",
                "attachment": "tech",
                "attachments": "tech",
                "ammo": "workbench",
                "bullet": "workbench",
                "bullets": "workbench",
                "medicine": "pharmacy",
                "medical": "pharmacy",
                "drug": "pharmacy",
                "armor": "armory",
                "armour": "armory",
                "bag": "armory",
            }
            for token in re.split(r"[,，\s]+", raw_group):
                mapped = alias.get(token, token)
                _add(mapped)
            if selected:
                return selected

        text = str(query or "")
        lowered = text.lower()
        for item in cls.PLACE_PROFIT_GROUPS:
            if any((keyword in text) or (keyword in lowered) for keyword in item["keywords"]):
                _add(item["place"])

        if selected:
            return selected
        return list(cls.PLACE_PROFIT_GROUPS)

    def _extract_rank_items(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not result.get("success"):
            return []
        payload = result.get("data")
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, dict):
            return []
        items = data.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    def _format_profit_item_line(self, index: int, item: Dict[str, Any]) -> str:
        name = str(item.get("objectName") or item.get("name") or "未知物品")
        profit = self._to_float(item.get("profit"))
        hour_profit = self._to_float(item.get("hourProfit"))
        ts = self._format_beijing_time(item.get("timestamp"))

        parts = [f"{index}. {name}"]
        if profit is not None:
            parts.append(f"净利润 {self._format_number(profit)}")
        if hour_profit is not None:
            parts.append(f"小时利润 {self._format_number(hour_profit)}")
        if ts:
            parts.append(f"采样时间 {ts}")
        return "｜".join(parts)

    def _format_place_profit_rank_result(self, query: str, params: Dict[str, Any]) -> str:
        top_n = self._detect_profit_top_n(query=query, params=params)
        rank_type = self._detect_profit_rank_type(query=query, params=params)
        groups = self._detect_profit_places(query=query, params=params)
        scope = "全部分组" if len(groups) == 4 else "指定分组"

        lines = [f"特勤处制造净利润榜（{scope}，Top{top_n}，排序维度: {rank_type}）"]
        for group in groups:
            place = group["place"]
            result = self.price_service.get_place_profit_rank(
                {"place": place, "type": rank_type, "limit": str(top_n)}
            )
            if not result.get("success"):
                lines.append(f"\n{group['label']}\n查询失败：{result.get('error', '未知错误')}")
                continue

            items = self._extract_rank_items(result)[:top_n]
            lines.append(f"\n{group['label']}")
            if not items:
                lines.append("暂无可用利润数据。")
                continue

            for idx, item in enumerate(items, 1):
                lines.append(self._format_profit_item_line(idx, item))
        return "\n".join(lines).strip()

    def build_latest_price_tool(self):
        @tool("df_market_latest_price")
        def df_market_latest_price(query: str) -> str:
            """
            查询物品最新价格。
            输入建议：
            - JSON: {"id":"14060000003"} 或 {"objectName":"非洲之心"}
            - 或文本: "id=14060000003" / "非洲之心最新价格"
            """
            params = self._parse_query_to_params(query)
            result = self.price_service.get_latest_price(params)
            return self._format_latest_result(result)

        return df_market_latest_price

    def build_history_price_tool(self):
        @tool("df_market_history_price")
        def df_market_history_price(query: str) -> str:
            """
            查询物品历史价格。
            输入建议：
            - JSON: {"id":"14060000003","startTime":"2026-02-20","endTime":"2026-02-24"}
            - JSON: {"objectName":"非洲之心","startTime":"2026-02-20","endTime":"2026-02-24"}
            - 或文本: "非洲之心 2026-02-20 到 2026-02-24 历史价格"
            具体参数名以 API 文档为准，工具会原样透传你提供的字段。
            """
            params = self._parse_query_to_params(query)
            result = self.price_service.get_history_price(params)
            return self._format_history_result(result)

        return df_market_history_price

    def build_price_advice_tool(self):
        @tool("df_market_price_advice")
        def df_market_price_advice(query: str) -> str:
            """
            基于“当前价格 + 历史区间”给出买卖建议与盈亏测算。
            输入建议：
            - "非洲之心现在能不能卖"
            - "非洲之心现在贵了还是便宜了"
            - "非洲之心现在建议买吗 成本价=12500000"
            """
            params = self._parse_query_to_params(query)
            latest_result = self.price_service.get_latest_price(params)
            history_result = self.price_service.get_history_price(params)
            if not history_result.get("success"):
                return self._format_history_result(history_result)
            return self._format_price_advice_result(query, latest_result, history_result)

        return df_market_price_advice

    def build_place_profit_rank_tool(self):
        @tool("df_place_profit_rank")
        def df_place_profit_rank(query: str) -> str:
            """
            查询特勤处制造利润榜（四个分组支持 Top1/Top3）：
            - 技术中心（枪械/配件）
            - 工作台（子弹）
            - 制药台（药品/针剂/维修工具包）
            - 防具台（头盔/护甲/胸挂/背包）

            用法示例：
            - "特勤处现在制造什么利润最高"
            - "制造什么子弹利润最高"
            - "技术中心枪械配件利润前三"
            """
            params = self._parse_query_to_params(query)
            return self._format_place_profit_rank_result(query=query, params=params)

        return df_place_profit_rank

    def build_multi_item_compare_tool(self):
        @tool("df_multi_item_compare")
        def df_multi_item_compare(query: str) -> str:
            """
            对比多个物品的价格区间与性价比，给出相对优先级。
            用法示例：
            - "非洲之心、海洋之泪、腾龙突击步枪对比一下"
            - "比较 非洲之心 和 海洋之泪 哪个更值得买"
            """
            params = self._parse_query_to_params(query)
            return self._format_multi_item_compare_result(query=query, params=params)

        return df_multi_item_compare

    def build_profit_stability_tool(self):
        @tool("df_profit_stability")
        def df_profit_stability(query: str) -> str:
            """
            分析制造利润稳定性（波动、正利润占比、最大回撤、趋势）。
            用法示例：
            - "分析 碳纤维散射箭矢 的利润稳定性"
            - "objectId=37270500002 的利润稳不稳"
            """
            params = self._parse_query_to_params(query)
            return self._format_profit_stability_result(query=query, params=params)

        return df_profit_stability

    def build_answer_composer_tool(self):
        @tool("df_answer_composer")
        def df_answer_composer(query: str) -> str:
            """
            综合回答工具：自动组合资料介绍、实时价格、买卖建议、制造利润信息。
            用法示例：
            - "介绍一下非洲之心并告诉我现在价格和是否建议买"
            - "介绍腾龙突击步枪，并给制造利润建议"
            """
            params = self._parse_query_to_params(query)
            return self._format_answer_composer_result(query=query, params=params)

        return df_answer_composer


def build_df_latest_price_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_latest_price_tool()


def build_df_history_price_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_history_price_tool()


def build_df_price_advice_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_price_advice_tool()


def build_df_place_profit_rank_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_place_profit_rank_tool()


def build_df_multi_item_compare_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_multi_item_compare_tool()


def build_df_profit_stability_tool(price_service: DFPriceService):
    return DFPriceTools(price_service).build_profit_stability_tool()


def build_df_answer_composer_tool(price_service: DFPriceService, rag_service: Any):
    return DFPriceTools(price_service, rag_service=rag_service).build_answer_composer_tool()
