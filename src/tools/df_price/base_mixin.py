"""DF 价格工具基础能力：参数清洗、查询解析、通用格式化。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


class DFPriceBaseMixin:
    BEIJING_TZ = timezone(timedelta(hours=8))
    MAX_OBJECT_NAME_LEN = 80
    MAX_LIMIT_VALUE = 20
    MAX_NAME_LIST_ITEMS = 6
    TIMESTAMP_MILLIS_ABS_THRESHOLD = 1e12
    TIMESTAMP_MILLIS_DIVISOR = 1000
    TIMESTAMP_MILLIS_DIGITS = 13
    OBJECT_NAME_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9\s\-\._\+\(\)（）×x/]{1,80}$")
    OBJECT_ID_PATTERN = re.compile(r"^\d{6,20}$")
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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

    @classmethod
    def _sanitize_object_name(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        if len(text) > cls.MAX_OBJECT_NAME_LEN:
            return ""
        if not cls.OBJECT_NAME_PATTERN.fullmatch(text):
            return ""
        return text

    @classmethod
    def _sanitize_object_id(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if not cls.OBJECT_ID_PATTERN.fullmatch(text):
            return ""
        return text

    @classmethod
    def _sanitize_date(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if cls.DATE_PATTERN.fullmatch(text):
            return text
        # 允许 10/13 位时间戳
        if re.fullmatch(r"\d{10}|\d{13}", text):
            return text
        return ""

    @classmethod
    def _sanitize_common_params(
        cls,
        params: Dict[str, Any],
        *,
        strict_entity: bool = True,
    ) -> tuple[Dict[str, Any], str]:
        safe: Dict[str, Any] = {}

        object_name = params.get("objectName", params.get("name"))
        if object_name not in (None, ""):
            normalized_name = cls._sanitize_object_name(object_name)
            if not normalized_name:
                if strict_entity:
                    return {}, "objectName 非法（仅允许中英文、数字、空格和常见连接符，长度 1-80）"
            else:
                safe["objectName"] = normalized_name

        object_id = params.get("id", params.get("objectId"))
        if object_id not in (None, ""):
            normalized_id = cls._sanitize_object_id(object_id)
            if not normalized_id:
                if strict_entity:
                    return {}, "id/objectId 非法（仅允许 6-20 位数字）"
            else:
                safe["id"] = normalized_id
                safe["objectId"] = normalized_id

        for key in ("date", "startTime", "endTime"):
            if key in params and params.get(key) not in (None, ""):
                normalized_date = cls._sanitize_date(params.get(key))
                if not normalized_date:
                    return {}, f"{key} 非法（仅允许 YYYY-MM-DD 或 10/13 位时间戳）"
                safe[key] = normalized_date

        for key in ("buyPrice", "costPrice", "entryPrice", "holdPrice", "cost", "buy_price", "cost_price"):
            if key in params and params.get(key) not in (None, ""):
                value = cls._to_float(params.get(key))
                if value is None or value <= 0:
                    return {}, f"{key} 非法（应为正数）"
                safe[key] = value

        if "place" in params and params.get("place") not in (None, ""):
            place = str(params.get("place")).strip().lower()
            allowed = {"tech", "workbench", "pharmacy", "armory"}
            if place not in allowed:
                return {}, "place 非法（仅允许 tech/workbench/pharmacy/armory）"
            safe["place"] = place
        if "group" in params and params.get("group") not in (None, ""):
            safe["group"] = str(params.get("group")).strip().lower()

        if "type" in params and params.get("type") not in (None, ""):
            rank_type = str(params.get("type")).strip().lower()
            allowed_type = {"hour", "total", "hourprofit", "totalprofit"}
            if rank_type not in allowed_type:
                return {}, "type 非法（仅允许 hour/total/hourprofit/totalprofit）"
            safe["type"] = rank_type

        for key in ("limit", "top", "topn", "n"):
            if key in params and params.get(key) not in (None, ""):
                try:
                    number = int(str(params.get(key)).strip())
                except Exception:
                    return {}, f"{key} 非法（应为整数）"
                safe[key] = max(1, min(number, cls.MAX_LIMIT_VALUE))

        for key in ("items", "objectNames"):
            if isinstance(params.get(key), list):
                names = [cls._sanitize_object_name(x) for x in params.get(key, [])]
                names = [x for x in names if x]
                if not names:
                    return {}, f"{key} 非法（列表中未找到有效物品名）"
                safe[key] = names[: cls.MAX_NAME_LIST_ITEMS]

        return safe, ""

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
            inferred = DFPriceBaseMixin._infer_object_name(text)
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
            "现在", "当前", "目前",
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
                yield from DFPriceBaseMixin._iter_nodes(value)
        elif isinstance(payload, list):
            for item in payload:
                yield from DFPriceBaseMixin._iter_nodes(item)

    @staticmethod
    def _pick_first(payload: Any, keys: List[str]) -> Any:
        for node in DFPriceBaseMixin._iter_nodes(payload):
            for key in keys:
                if key in node and node[key] not in (None, ""):
                    return node[key]
        return None

    @staticmethod
    def _pick_records(payload: Any) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for node in DFPriceBaseMixin._iter_nodes(payload):
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
            timestamp = (
                number / cls.TIMESTAMP_MILLIS_DIVISOR
                if abs(number) >= cls.TIMESTAMP_MILLIS_ABS_THRESHOLD
                else number
            )
        else:
            text = str(value).strip()
            if not text:
                return ""

            if re.fullmatch(r"\d{10,13}", text):
                number = float(text)
                # 13位时间戳按毫秒处理
                timestamp = (
                    number / cls.TIMESTAMP_MILLIS_DIVISOR
                    if len(text) >= cls.TIMESTAMP_MILLIS_DIGITS
                    else number
                )
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

    @staticmethod
    def _format_number(value: float) -> str:
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"
