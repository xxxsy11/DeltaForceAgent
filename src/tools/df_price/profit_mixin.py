"""DF 价格工具制造利润能力：稳定性与分组利润榜。"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List


class DFPriceProfitMixin:
    STABILITY_HIGH_POSITIVE_RATIO = 0.8
    STABILITY_HIGH_CV = 0.5
    STABILITY_HIGH_MAX_DRAWDOWN = 0.35
    STABILITY_MEDIUM_POSITIVE_RATIO = 0.6
    STABILITY_MEDIUM_CV = 0.9
    STABILITY_MEDIUM_MAX_DRAWDOWN = 0.6
    INFINITE_CV_FALLBACK = 999.0
    PERCENT_MULTIPLIER = 100

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

    @staticmethod
    def _classify_profit_stability(positive_ratio: float, cv: float, max_drawdown: float) -> str:
        if (
            positive_ratio >= DFPriceProfitMixin.STABILITY_HIGH_POSITIVE_RATIO
            and cv <= DFPriceProfitMixin.STABILITY_HIGH_CV
            and max_drawdown <= DFPriceProfitMixin.STABILITY_HIGH_MAX_DRAWDOWN
        ):
            return "高"
        if (
            positive_ratio >= DFPriceProfitMixin.STABILITY_MEDIUM_POSITIVE_RATIO
            and cv <= DFPriceProfitMixin.STABILITY_MEDIUM_CV
            and max_drawdown <= DFPriceProfitMixin.STABILITY_MEDIUM_MAX_DRAWDOWN
        ):
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
            cv=cv if math.isfinite(cv) else self.INFINITE_CV_FALLBACK,
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
            (
                f"- 正利润占比：{positive_ratio * self.PERCENT_MULTIPLIER:.1f}%"
                f"，最大回撤：{max_drawdown * self.PERCENT_MULTIPLIER:.1f}%"
                f"，趋势变化：{trend * self.PERCENT_MULTIPLIER:+.1f}%"
            ),
            f"- 稳定性评级：{stability}",
            f"- 建议：{advice}",
        ]
        return "\n".join(lines)

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
