"""
图索引模块（Delta Force）
实现实体和关系的键值对结构 (K,V)
"""

import json
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

from .llm_utils import invoke_llm_text

logger = logging.getLogger(__name__)


@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]
    value_content: str
    entity_type: str
    metadata: Dict[str, Any]


@dataclass
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]
    value_content: str
    relation_type: str
    source_entity: str
    target_entity: str
    metadata: Dict[str, Any]


class GraphIndexingModule:
    """图索引模块"""

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client

        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}

        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)

    def _primary_label(self, labels: List[str]) -> str:
        for label in labels:
            if label != "Node":
                return label
        return labels[0] if labels else "Node"

    def create_entity_key_values(self, nodes: List[Any]) -> Dict[str, EntityKeyValue]:
        """为实体创建键值对结构（通用节点）"""
        logger.info("开始创建实体键值对...")

        for node in nodes:
            entity_id = node.node_id
            entity_name = node.name or f"实体_{entity_id}"
            entity_type = self._primary_label(getattr(node, "labels", []) or [])

            props = getattr(node, "properties", {}) or {}

            content_parts = [f"名称: {entity_name}", f"类型: {entity_type}"]
            for k, v in props.items():
                if v is None or v == "":
                    continue
                if isinstance(v, list):
                    v_str = "、".join([str(x) for x in v])
                else:
                    v_str = str(v)
                content_parts.append(f"{k}: {v_str}")

            index_keys = [entity_name]
            aliases = props.get("aliases")
            if isinstance(aliases, list):
                index_keys.extend([str(a) for a in aliases])
            for key in ("typeName", "difficulty", "caliber", "colorName"):
                if props.get(key):
                    index_keys.append(str(props.get(key)))
            if props.get("level") is not None:
                index_keys.append(str(props.get("level")))

            entity_kv = EntityKeyValue(
                entity_name=entity_name,
                index_keys=list(dict.fromkeys(index_keys)),
                value_content="\n".join(content_parts),
                entity_type=entity_type,
                metadata={
                    "node_id": entity_id,
                    "properties": props
                }
            )

            self.entity_kv_store[entity_id] = entity_kv
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        logger.info(f"实体键值对创建完成，共 {len(self.entity_kv_store)} 个实体")
        return self.entity_kv_store

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """为关系创建键值对结构"""
        logger.info("开始创建关系键值对...")

        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"rel_{i}_{source_id}_{target_id}"

            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            if not source_entity or not target_entity:
                continue

            content_parts = [
                f"关系类型: {relation_type}",
                f"源实体: {source_entity.entity_name} ({source_entity.entity_type})",
                f"目标实体: {target_entity.entity_name} ({target_entity.entity_type})"
            ]

            index_keys = self._generate_relation_index_keys(
                source_entity, target_entity, relation_type
            )

            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content="\n".join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={
                    "source_name": source_entity.entity_name,
                    "target_name": target_entity.entity_name,
                    "created_from_graph": True
                }
            )

            self.relation_kv_store[relation_id] = relation_kv
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)

        logger.info(f"关系键值对创建完成，共 {len(self.relation_kv_store)} 个关系")
        return self.relation_kv_store

    def _generate_relation_index_keys(self, source_entity: EntityKeyValue,
                                      target_entity: EntityKeyValue,
                                      relation_type: str) -> List[str]:
        keys = [relation_type, source_entity.entity_name, target_entity.entity_name]

        relation_hints = {
            "HAS_AREA": ["地图区域", "区域"],
            "HAS_KEY_CARD": ["房卡", "刷卡点"],
            "HAS_LEVEL": ["等级", "稀有度"],
            "OF_EQ_TYPE": ["装备类型", "装备分类"],
            "OF_COL_TYPE": ["收集品类型", "收集品分类"],
            "OF_FIRE_TYPE": ["枪械类型", "枪械分类"],
            "OF_ATT_TYPE": ["配件类型", "配件分类"],
            "OF_AMMO_TYPE": ["弹药类型", "弹药分类", "口径"],
            "OF_CLA_TYPE": ["干员兵种", "职业类型", "兵种分类"],
            "CAN_ATTACH": ["可装配件", "配件兼容"],
            "USES_AMMO": ["弹药", "口径"],
            "ENABLES_ATTACHMENT": ["解锁配件", "扩展配件"]
        }
        keys.extend(relation_hints.get(relation_type, []))

        if getattr(self.config, "enable_llm_relation_keys", False):
            keys.extend(self._llm_enhance_relation_keys(source_entity, target_entity, relation_type))

        return list(dict.fromkeys(keys))

    def _llm_enhance_relation_keys(self, source_entity: EntityKeyValue,
                                   target_entity: EntityKeyValue,
                                   relation_type: str) -> List[str]:
        prompt = f"""
        分析以下实体关系，生成相关的主题关键词：
        源实体: {source_entity.entity_name} ({source_entity.entity_type})
        目标实体: {target_entity.entity_name} ({target_entity.entity_type})
        关系类型: {relation_type}
        请生成3-5个相关关键词，返回JSON：{{"keywords": ["..."]}}
        """

        try:
            llm_text = invoke_llm_text(
                llm_client=self.llm_client,
                prompt=prompt,
                model=self.config.llm_model,
                temperature=0.1,
                max_tokens=200,
            )
            result = json.loads(llm_text.strip())
            return result.get("keywords", [])
        except Exception as e:
            logger.error(f"关系关键词生成失败: {e}")
            return []

    def deduplicate_entities_and_relations(self):
        """去重并重建索引映射"""
        # 以“名称+类型”作为实体唯一键，避免跨类型同名实体被误合并。
        name_to_entities = defaultdict(list)
        for entity_id, entity_kv in self.entity_kv_store.items():
            signature = (
                str(entity_kv.entity_name).strip().lower(),
                str(entity_kv.entity_type).strip().lower(),
            )
            name_to_entities[signature].append(entity_id)

        entities_to_remove = []
        for _, entity_ids in name_to_entities.items():
            if len(entity_ids) > 1:
                primary_id = entity_ids[0]
                primary_entity = self.entity_kv_store[primary_id]
                for entity_id in entity_ids[1:]:
                    duplicate_entity = self.entity_kv_store[entity_id]
                    primary_entity.value_content += f"\n\n补充信息: {duplicate_entity.value_content}"
                    entities_to_remove.append(entity_id)

        for entity_id in entities_to_remove:
            del self.entity_kv_store[entity_id]

        relation_signature_to_ids = defaultdict(list)
        for relation_id, relation_kv in self.relation_kv_store.items():
            signature = f"{relation_kv.source_entity}_{relation_kv.target_entity}_{relation_kv.relation_type}"
            relation_signature_to_ids[signature].append(relation_id)

        relations_to_remove = []
        for signature, relation_ids in relation_signature_to_ids.items():
            if len(relation_ids) > 1:
                relations_to_remove.extend(relation_ids[1:])

        for relation_id in relations_to_remove:
            del self.relation_kv_store[relation_id]

        self._rebuild_key_mappings()
        logger.info(f"去重完成 - 删除了 {len(entities_to_remove)} 个重复实体，{len(relations_to_remove)} 个重复关系")

    def _rebuild_key_mappings(self):
        self.key_to_entities.clear()
        self.key_to_relations.clear()

        for entity_id, entity_kv in self.entity_kv_store.items():
            for key in entity_kv.index_keys:
                self.key_to_entities[key].append(entity_id)

        for relation_id, relation_kv in self.relation_kv_store.items():
            for key in relation_kv.index_keys:
                self.key_to_relations[key].append(relation_id)

    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        entity_ids = self.key_to_entities.get(key, [])
        return [self.entity_kv_store[eid] for eid in entity_ids if eid in self.entity_kv_store]

    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        relation_ids = self.key_to_relations.get(key, [])
        return [self.relation_kv_store[rid] for rid in relation_ids if rid in self.relation_kv_store]

    def get_statistics(self) -> Dict[str, Any]:
        entity_type_counts: Dict[str, int] = {}
        for kv in self.entity_kv_store.values():
            entity_type_counts[kv.entity_type] = entity_type_counts.get(kv.entity_type, 0) + 1

        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": entity_type_counts
        }
