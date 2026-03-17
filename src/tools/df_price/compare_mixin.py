"""DF 价格工具对比能力：多物品提取与对比分析。"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class DFPriceCompareMixin:
    MAX_COMPARE_ITEMS = 6
    POSITION_MIDPOINT = 0.5
    DOWNSIDE_WEIGHT = 0.8
    MAX_FAILED_DETAIL_LINES = 4

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
            names = [cls._sanitize_object_name(x) for x in params.get("items", [])]
            return cls._unique_keep_order([x for x in names if x])

        if isinstance(params.get("objectNames"), list):
            names = [cls._sanitize_object_name(x) for x in params.get("objectNames", [])]
            return cls._unique_keep_order([x for x in names if x])

        merged = []
        if params.get("objectName"):
            merged.append(str(params.get("objectName")))
        if params.get("name"):
            merged.append(str(params.get("name")))
        if merged:
            tokens = re.split(r"[，,、/|\s]+|和|与|以及|并且|还有|对比|比较", " ".join(merged))
            normalized = [cls._sanitize_object_name(t.strip()) for t in tokens if t.strip()]
            return cls._unique_keep_order([x for x in normalized if x])

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
        normalized = [cls._sanitize_object_name(x) for x in candidates]
        return cls._unique_keep_order([x for x in normalized if x])

    @classmethod
    def _extract_primary_item_name(cls, query: str, params: Dict[str, Any]) -> str:
        def _normalize_name(raw: str) -> str:
            name = str(raw or "").strip()
            if not name:
                return ""

            # 先按连接词截断，避免把“并告诉我现在价格”一并当作实体
            name = re.split(r"并告诉我|并告诉|并且告诉我|并且告诉|并且|以及|同时|和|与|并|；|;|，|,", name)[0].strip()

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
                "再介绍一下",
                "再介绍",
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
                "现在价格",
                "当前价格",
                "是否建议买",
                "建议买吗",
            ]
            for token in suffix_tokens:
                if name.endswith(token) and len(name) > len(token):
                    name = name[: -len(token)].strip()

            name = name.strip(" ：:，,。.")
            # 排除明显非实体短语
            if any(x in name for x in ("总结", "聊了什么", "资料介绍", "买卖建议")):
                return ""
            return name

        if params.get("objectName"):
            return cls._sanitize_object_name(_normalize_name(str(params.get("objectName")).strip()))
        if params.get("name"):
            return cls._sanitize_object_name(_normalize_name(str(params.get("name")).strip()))

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
                return cls._sanitize_object_name(name)

        candidates = cls._extract_item_names(query=query, params=params)
        for name in candidates:
            normalized = _normalize_name(name)
            if normalized and normalized not in {"介绍", "查询", "价格", "利润", "建议", "现在", "当前"}:
                return cls._sanitize_object_name(normalized)
        return ""

    def _format_multi_item_compare_result(self, query: str, params: Dict[str, Any]) -> str:
        names = self._extract_item_names(query=query, params=params)
        if len(names) < 2:
            return "请至少提供两个物品名称用于对比，例如：非洲之心、海洋之泪对比。"

        results = []
        failed = []
        for name in names[: self.MAX_COMPARE_ITEMS]:
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
            position = self.POSITION_MIDPOINT if band <= 0 else max(0.0, min(1.0, (current - min_price) / band))
            upside = max(0.0, max_price - current)
            downside = max(0.0, current - min_price)
            score = upside - self.DOWNSIDE_WEIGHT * downside
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
            lines.append("未成功对比：" + "；".join(failed[: self.MAX_FAILED_DETAIL_LINES]))
        return "\n".join(lines)
