"""Auto-split from graph_rag_retrieval.py."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rag_modules.graph_retrieval.types import GraphQuery, QueryType
from rag_modules.llm_utils import invoke_llm_text


logger = logging.getLogger(__name__)

GRAPH_DEPTH_DEFAULT = 2
GRAPH_DEPTH_MIN = 1
QUERY_LIST_MAX_SIZE = 8
MAX_NODES_DEFAULT = 50
MAX_NODES_MIN = 10
MAX_NODES_MAX = 100
SIMPLE_LOOKUP_MAX_DEPTH = 1
SIMPLE_LOOKUP_MAX_NODES = 20
QUERY_LLM_TEMPERATURE = 0.0
QUERY_LLM_MAX_TOKENS = 1000
COMPLEXITY_SIMPLE_THRESHOLD = 0.3
COMPLEXITY_MEDIUM_THRESHOLD = 0.7
PLAN_SIMPLE_MAX_NODES = 20
PLAN_MEDIUM_MAX_NODES = 50
PLAN_COMPLEX_MAX_NODES = 100


class GraphQueryMixin:
    def _normalize_max_depth(self, max_depth: int) -> int:
        configured_depth = getattr(self.config, "max_graph_depth", GRAPH_DEPTH_DEFAULT)
        return max(GRAPH_DEPTH_MIN, min(int(max_depth or GRAPH_DEPTH_MIN), int(configured_depth)))

    def _expand_relation_types_for_query(self, relation_types: Optional[List[str]]) -> List[str]:
        if not relation_types:
            return []
        expanded: List[str] = []
        seen = set()
        for relation_type in relation_types:
            if relation_type is None:
                continue
            rel = str(relation_type).strip()
            if not rel or rel in seen:
                continue
            seen.add(rel)
            expanded.append(rel)
        return expanded

    def _resolve_relation_filters(self, relation_types: Optional[List[str]]) -> Tuple[List[str], List[str]]:
        """
        解析关系过滤策略：
        - allowed_relation_types: 用于Cypher路径白名单过滤
        - preferred_relation_types: 用于路径打分加权
        """
        preferred = self._expand_relation_types_for_query(relation_types)
        # 工程策略：查询过滤始终使用稳定白名单，避免 LLM relation_types 误判造成“0结果”。
        # preferred 仅用于排序加权，不参与硬过滤。
        allowed = list(self.default_relation_whitelist)
        preferred = [rel for rel in preferred if rel in set(allowed)]
        return allowed, preferred

    @staticmethod

    def _is_simple_entity_lookup(query: str) -> bool:
        text = (query or "").strip()
        if not text:
            return False
        relation_cues = (
            "关系", "路径", "连接", "比较", "区别", "影响", "为什么", "如何", "怎么",
            "哪些", "以及", "并且", "和", "与", "搭配", "推荐", "历史", "走势", "价格",
        )
        return not any(cue in text for cue in relation_cues)

    def _extract_json_payload(self, content: str) -> Dict[str, Any]:
        raw = (content or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _normalize_string_list(self, value: Any, max_size: int = QUERY_LIST_MAX_SIZE) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        seen = set()
        for item in value:
            text = str(item).strip() if item is not None else ""
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(text)
            if len(normalized) >= max_size:
                break
        return normalized

    def _sanitize_graph_query_payload(self, payload: Dict[str, Any], fallback_query: str) -> GraphQuery:
        query_type_raw = str(payload.get("query_type", QueryType.SUBGRAPH.value)).strip()
        try:
            query_type = QueryType(query_type_raw)
        except ValueError:
            query_type = QueryType.SUBGRAPH

        source_entities = self._normalize_string_list(payload.get("source_entities"), max_size=QUERY_LIST_MAX_SIZE)
        if not source_entities:
            source_entities = [fallback_query.strip()] if fallback_query.strip() else []

        target_entities = self._normalize_string_list(payload.get("target_entities"), max_size=QUERY_LIST_MAX_SIZE)

        preferred_relations = self._expand_relation_types_for_query(payload.get("relation_types"))
        allowed_relation_pool = set(self.default_relation_whitelist) | set(self.relation_cache.keys())
        relation_types = [rel for rel in preferred_relations if rel in allowed_relation_pool]

        constraints = payload.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}

        max_depth = self._normalize_max_depth(payload.get("max_depth", GRAPH_DEPTH_DEFAULT))
        max_nodes = int(payload.get("max_nodes", MAX_NODES_DEFAULT) or MAX_NODES_DEFAULT)
        max_nodes = max(MAX_NODES_MIN, min(max_nodes, MAX_NODES_MAX))

        # 简单“实体介绍/定义”查询默认收敛到 1-hop，减少高频枢纽节点污染（如六级、类型节点扩散）。
        if query_type == QueryType.SUBGRAPH and len(source_entities) <= 1 and self._is_simple_entity_lookup(fallback_query):
            max_depth = SIMPLE_LOOKUP_MAX_DEPTH
            max_nodes = min(max_nodes, SIMPLE_LOOKUP_MAX_NODES)

        return GraphQuery(
            query_type=query_type,
            source_entities=source_entities,
            target_entities=target_entities,
            relation_types=relation_types,
            max_depth=max_depth,
            max_nodes=max_nodes,
            constraints=constraints,
        )

    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        理解查询的图结构意图
        这是图RAG的核心：从自然语言到图查询的转换
        """
        prompt = f"""
        你是三角洲行动图数据库查询规划器。只输出 JSON，不要输出任何解释文字。

        问题：{query}

        图中主要关系类型：
        ["HAS_AREA","HAS_KEY_CARD","HAS_DIFFICULTY","HAS_LEVEL","HAS_SKILL",
         "OF_CLA_TYPE","OF_EQ_TYPE","OF_COL_TYPE","OF_FIRE_TYPE","OF_ATT_TYPE","OF_AMMO_TYPE",
         "USES_AMMO","CAN_ATTACH"]

        query_type 只能是：
        - "entity_relation"
        - "multi_hop"
        - "subgraph"
        - "path_finding"
        - "clustering"

        输出 JSON 结构必须是：
        {{
          "query_type": "subgraph",
          "source_entities": ["实体名1"],
          "target_entities": [],
          "relation_types": ["HAS_AREA"],
          "max_depth": 2,
          "constraints": {{}}
        }}

        规则：
        1. source_entities 只放在图里最可能存在的具体实体名。
        2. relation_types 只允许使用上面的关系类型；不确定就返回空数组。
        3. max_depth 只允许 1~3 的整数。
        4. constraints 仅保存属性筛选（如等级、口径），无则返回空对象。
        """
        
        try:
            llm_text = invoke_llm_text(
                llm_client=self.llm_client,
                prompt=prompt,
                temperature=QUERY_LLM_TEMPERATURE,
                max_tokens=QUERY_LLM_MAX_TOKENS,
            )
            payload = self._extract_json_payload(llm_text)
            if not payload:
                raise ValueError("LLM 未返回可解析 JSON")

            return self._sanitize_graph_query_payload(payload, fallback_query=query)
            
        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query.strip()] if query.strip() else [],
                max_depth=GRAPH_DEPTH_DEFAULT
            )

    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """
        自适应查询规划：根据查询复杂度动态调整策略
        """
        # 分析查询复杂度
        complexity_score = self._analyze_query_complexity(query)
        
        query_plans = []
        
        if complexity_score < COMPLEXITY_SIMPLE_THRESHOLD:
            # 简单查询：直接邻居查询
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=PLAN_SIMPLE_MAX_NODES
            )
            query_plans.append(plan)
            
        elif complexity_score < COMPLEXITY_MEDIUM_THRESHOLD:
            # 中等复杂度：多跳查询
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=GRAPH_DEPTH_DEFAULT,
                max_nodes=PLAN_MEDIUM_MAX_NODES
            )
            query_plans.append(plan)
            
        else:
            # 复杂查询：子图提取 + 推理
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=PLAN_COMPLEX_MAX_NODES
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=PLAN_MEDIUM_MAX_NODES
            )
            query_plans.extend([plan1, plan2])
            
        return query_plans
