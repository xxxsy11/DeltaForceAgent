"""DF 价格工具市场分析能力：最新价、历史价、买卖建议。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


class DFPriceMarketMixin:
    MAX_ENDPOINT_ATTEMPTS_TO_SHOW = 5
    POSITION_HIGH_THRESHOLD = 0.8
    POSITION_LOW_THRESHOLD = 0.2
    POSITION_MIDPOINT = 0.5

    def _format_latest_result(self, result: Dict[str, Any]) -> str:
        if not result.get("success"):
            tried = result.get("tried") or result.get("resolve_detail", {}).get("tried") or []
            detail_lines = []
            for item in tried[: self.MAX_ENDPOINT_ATTEMPTS_TO_SHOW]:
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
            for item in tried[: self.MAX_ENDPOINT_ATTEMPTS_TO_SHOW]:
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
        if position >= DFPriceMarketMixin.POSITION_HIGH_THRESHOLD:
            return {
                "level": "高位",
                "buy": "当前处于历史偏高区，谨慎追高，建议等待回调再买。",
                "sell": "若你已持有，当前更适合分批止盈或减仓。",
            }
        if position <= DFPriceMarketMixin.POSITION_LOW_THRESHOLD:
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
        position = self.POSITION_MIDPOINT if band <= 0 else max(0.0, min(1.0, (current_price - min_price) / band))
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
