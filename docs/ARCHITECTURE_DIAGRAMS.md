# Delta Agent 系统架构图集

## 1. 主流程图（完整版）

```mermaid
flowchart TD
    START([用户输入问题]) --> ORCH[main_orchestrator<br/>流程编排起点]
    ORCH --> PMR[persistent_memory_recall<br/>长期记忆门控与召回]

    PMR -->|门控打分 >= 阈值| DEEP_RECALL[执行深召回<br/>向量+BM25+RRF]
    PMR -->|门控打分 < 阈值| SIMPLE_RECALL[仅补充最新摘要]

    DEEP_RECALL --> INTENT[intent_recognition<br/>意图识别节点]
    SIMPLE_RECALL --> INTENT

    INTENT --> RULES[IntentAnalyzer<br/>规则层初判]
    RULES --> LLM_INTENT[LLM结构化理解<br/>JSON输出]
    LLM_INTENT --> ENTITY[主体补全层<br/>代词/省略处理]
    ENTITY --> SKILLS[Skills匹配<br/>技能选择与规划]

    SKILLS -->|无工具| SUM_DIRECT[直接总结]
    SKILLS -->|简单+不需要规划| EXEC[execution<br/>工具执行]
    SKILLS -->|复杂+需要规划| PLAN[task_planning<br/>二级工具规划]

    PLAN --> EXEC

    EXEC --> TOOL_CALLS[工具链调用]
    TOOL_CALLS --> RAG_TOOL{RAG工具?}
    TOOL_CALLS --> MARKET_TOOL{市场工具?}

    RAG_TOOL -->|是| RAG_SRV[RAGService<br/>统一RAG查询]
    RAG_SRV --> RAG_ROUTE[智能路由器<br/>选择检索策略]

    RAG_ROUTE -->|传统检索| HYBIRD[hybrid_retrieval<br/>向量+BM25+RRF]
    RAG_ROUTE -->|图检索| GRAPH[graph_rag_retrieval<br/>多跳遍历]
    RAG_ROUTE -->|组合| COMBINED[combined<br/>混合策略]

    HYBIRD --> GEN[generation_integration<br/>生成回答]
    GRAPH --> GEN
    COMBINED --> GEN

    MARKET_TOOL -->|是| PRICE_SRV[DFPriceService<br/>统一市场数据服务]
    PRICE_SRV --> PRICE_API[外部API调用<br/>带重试机制]

    GEN --> ANALYSIS[结构化分析报告]
    PRICE_API --> ANALYSIS

    ANALYSIS --> NEED_SPECIALIST{需要专业分析?}
    NEED_SPECIALIST -->|是| SPEC[specialist_analysis<br/>专业解读]
    NEED_SPECIALIST -->|否| SUM[summary<br/>生成最终回答]

    SPEC --> SUM
    SUM_DIRECT --> SUM

    SUM --> COMP[memory_compression<br/>短期记忆压缩]

    COMP --> CHECK_COMPRESS{需要压缩?}
    CHECK_COMPRESS -->|pending_turns >= 阈值<br/>或 pending_tokens >= 阈值| DO_COMPRESS[执行压缩<br/>merge摘要+提取facts]
    CHECK_COMPRESS -->|否| SKIP_COMPRESS[保持现有状态]

    DO_COMPRESS --> WRITE[persistent_memory_write<br/>长期记忆写入]
    SKIP_COMPRESS --> WRITE

    WRITE --> WRITE_TURNS[写入chat_turns<br/>完整对话]
    WRITE --> WRITE_SUMM[写入memory_summaries<br/>阶段摘要]
    WRITE --> WRITE_FACTS[写入memory_facts<br/>结构化事实]
    WRITE --> EXPORT[导出可读镜像<br/>本地JSONL]

    WRITE_TURNS --> END([返回最终回答])
    WRITE_SUMM --> END
    WRITE_FACTS --> END
    EXPORT --> END

    style START fill:#e1f5e1
    style END fill:#e1f5e1
    style INTENT fill:#fff4e1
    style EXEC fill:#e1f0ff
    style SPEC fill:#ffe1f0
    style SUM fill:#f0ffe1
    style COMP fill:#f0e1ff
    style WRITE fill:#e1f0f0
```

---

## 2. 意图识别详细流程图

```mermaid
flowchart TD
    INPUT[用户问题] --> INTENT_NODE[intent_recognition节点]

    INTENT_NODE --> LAYER1[第一层: IntentAnalyzer<br/>规则基础分类]
    LAYER1 --> KEYWORDS[关键词匹配]
    KEYWORDS --> RULE_OUTPUT{规则置信度}
    RULE_OUTPUT -->|高置信度| RULE_RESULT[输出规则结果]
    RULE_OUTPUT -->|低置信度| LAYER2

    LAYER2[第二层: LLM结构化理解<br/>JSON格式解析]
    LAYER2 --> LLM_PARSE[解析: intent/tool/entities/confidence]
    LLM_PARSE --> LLM_RESULT[输出LLM理解结果]

    RULE_RESULT --> LAYER3[第三层: 主体补全<br/>记忆增强]
    LLM_RESULT --> LAYER3

    LAYER3 --> CHECK_PRONOUN{检测代词/省略?}
    CHECK_PRONOUN -->|是| EXTRACT_ENTITY[提取实体来源]
    EXTRACT_ENTITY --> PRIORITY1[优先级1: 长期召回命中]
    EXTRACT_ENTITY --> PRIORITY2[优先级2: 短期记忆]
    EXTRACT_ENTITY --> PRIORITY3[优先级3: 滚动摘要]
    PRIORITY1 --> MERGE[合并主体信息]
    PRIORITY2 --> MERGE
    PRIORITY3 --> MERGE

    CHECK_PRONOUN -->|否| LAYER4
    MERGE --> LAYER4

    LAYER4[第四层: Skills匹配与规划]
    LAYER4 --> SKILL_SCORE[技能评分计算]
    SKILL_SCORE --> S1[intent命中: +6分]
    SKILL_SCORE --> S2[tool命中: +4分]
    SKILL_SCORE --> S3[keywords_any: +3分]
    SKILL_SCORE --> S4[keywords_all: +2分]
    SKILL_SCORE --> S5[实体数量评估]

    S1 --> SELECT[选择最佳技能]
    S2 --> SELECT
    S3 --> SELECT
    S4 --> SELECT
    S5 --> SELECT

    SELECT --> CHECK_SKILL{找到匹配技能?}
    CHECK_SKILL -->|是| SKILL_PLAN[生成SkillPlan<br/>工具链+锁定标志]
    CHECK_SKILL -->|否| DEFAULT[使用默认工具链]

    SKILL_PLAN --> OUTPUT[输出意图识别结果]
    DEFAULT --> OUTPUT

    OUTPUT --> O1[selected_tool / tool_calls]
    OUTPUT --> O2[flow_type: simple/complex]
    OUTPUT --> O3[requires_task_planning]
    OUTPUT --> O4[requires_specialist_analysis]
    OUTPUT --> O5[selected_skill / skill_locked_plan]

    style LAYER1 fill:#ffe1e1
    style LAYER2 fill:#e1f0ff
    style LAYER3 fill:#f0ffe1
    style LAYER4 fill:#fff4e1
    style OUTPUT fill:#e1f0f0
```

---

## 3. 记忆系统架构图

```mermaid
flowchart TB
    subgraph SHORT_TERM["短期记忆 (SessionMemory)"]
        RECENT[recent_raw<br/>最近原文窗口<br/>默认: 最近3轮]
        PENDING[pending_buffer<br/>待压缩缓冲区<br/>等待合并的对话]
        ROLLING[rolling_summary<br/>滚动摘要<br/>压缩后的历史]
    end

    subgraph COMPRESSION["压缩机制 (memory_compression)"]
        TRIGGER[压缩触发检测]
        TRIGGER --> T1[pending_turns >= 10]
        TRIGGER --> T2[pending_tokens >= 6000]
        TRIGGER --> T3[memory_force_compress]

        T1 --> MERGE[合并摘要]
        T2 --> MERGE
        T3 --> MERGE

        MERGE --> CHECK_REBASE{达到rebase条件?}
        CHECK_REBASE -->|每5次merge| REBASE[执行rebase<br/>重写基础摘要]
        CHECK_REBASE -->|否| FACTS[提取结构化facts]

        REBASE --> FACTS
        FACTS --> UPDATE[更新rolling_summary<br/>清空pending_buffer]
    end

    subgraph LONG_TERM["长期记忆 (PersistentMemoryStore)"]
        POSTGRE[(PostgreSQL + pgvector)]

        POSTGRE --> TBL1[chat_sessions<br/>会话元信息]
        POSTGRE --> TBL2[chat_turns<br/>完整轮次记录]
        POSTGRE --> TBL3[memory_summaries<br/>阶段摘要快照]
        POSTGRE --> TBL4[memory_facts<br/>结构化事实]

        TBL4 --> VECTOR[向量索引<br/>pgvector]
        TBL4 --> BM25[BM25索引<br/>关键词]
        TBL4 --> QUALITY[质量评分<br/>0.2-1.0]
    end

    subgraph RECALL["召回机制 (persistent_memory_recall)"]
        GATE[门控打分]
        GATE --> G1[代词/指代词: +2]
        GATE --> G2[对比/建议/趋势: +2]
        GATE --> G3[历史词: +1]
        GATE --> G4[数量词N个: +1]

        G1 --> CHECK_GATE{分数 >= 2?}
        G2 --> CHECK_GATE
        G3 --> CHECK_GATE
        G4 --> CHECK_GATE

        CHECK_GATE -->|是| DEEP[深召回]
        CHECK_GATE -->|否| SHALLOW[浅召回]

        DEEP --> VEC[向量召回]
        DEEP --> KEY[BM25召回]
        DEEP --> RRF[RRF融合]

        VEC --> FUSION[融合排序]
        KEY --> FUSION
        RRF --> FUSION
    end

    subgraph WRITE["写入机制 (persistent_memory_write)"]
        W1[每轮: 写入chat_turns]
        W2[压缩时: 写入memory_summaries]
        W3[压缩时: 写入memory_facts]
        W4[每次: 导出本地镜像]
    end

    PENDING --> COMPRESSION
    RECENT --> CONTEXT[构建memory_context]
    ROLLING --> CONTEXT
    CONTEXT --> GATE

    FUSION --> CONTEXT
    UPDATE --> W1
    FACTS --> W2
    FACTS --> W3
    W1 --> POSTGRE
    W2 --> POSTGRE
    W3 --> POSTGRE
    W4 --> LOCAL[本地可读文件<br/>data/memory/readable/]

    style SHORT_TERM fill:#e1f0ff
    style LONG_TERM fill:#f0ffe1
    style RECALL fill:#fff4e1
    style WRITE fill:#ffe1f0
    style COMPRESSION fill:#f0e1ff
```

---

## 4. GraphRAG 子系统架构图

```mermaid
flowchart TD
    subgraph OFFLINE["离线构建阶段"]
        NEO4J[(Neo4j<br/>图数据库)]
        NEO4J --> PREP[graph_data_preparation<br/>图数据准备]

        PREP --> P1[拉取节点与关系]
        PREP --> P2[关系类型校验]
        PREP --> P3[构建实体文档]
        PREP --> P4[文档分块]

        P4 --> INDEX[milvus_index_construction<br/>向量索引构建]
        INDEX --> I1[创建collection/schema]
        INDEX --> I2[生成embedding]
        INDEX --> I3[批量写入Milvus]
        INDEX --> I4[创建HNSW索引]

        I4 --> GRAPH_IDX[graph_indexing<br/>图索引构建]
        GRAPH_IDX --> G1[实体键值索引]
        GRAPH_IDX --> G2[关系键值索引]
        GRAPH_IDX --> G3[去重处理]
    end

    subgraph ONLINE["在线服务阶段"]
        QUERY[用户查询] --> ROUTER[intelligent_query_router<br/>智能路由器]

        ROUTER --> ANALYSIS[QueryAnalysis]
        ANALYSIS --> A1[query_complexity]
        ANALYSIS --> A2[relationship_intensity]
        ANALYSIS --> A3[recommended_strategy]

        A3 --> STRATEGY{路由策略}

        STRATEGY -->|传统检索| TRAD[hybrid_retrieval<br/>混合传统检索]
        STRATEGY -->|图检索| GRAPH[graph_rag_retrieval<br/>图检索]
        STRATEGY -->|组合| BOTH[同时执行两种检索]

        TRAD --> TRAD1[双层检索<br/>实体层+主题层]
        TRAD --> TRAD2[向量检索增强]
        TRAD --> TRAD3[RRF融合排序]

        GRAPH --> GRAPH1[图查询理解]
        GRAPH --> GRAPH2[多跳遍历]
        GRAPH --> GRAPH3[子图提取]
        GRAPH --> GRAPH4[关系推理链]
        GRAPH --> GRAPH5[图相关性重排]

        TRAD3 --> RETRIEVE[检索结果]
        GRAPH5 --> RETRIEVE
        BOTH --> RETRIEVE
    end

    subgraph GENERATION["生成阶段"]
        RETRIEVE --> GEN[generation_integration<br/>生成模块]
        GEN --> STREAM{是否流式?}
        STREAM -->|是| STREAM_OUT[流式输出]
        STREAM -->|否或失败| NORMAL_OUT[非流式输出]

        STREAM_OUT --> RETRY{输出失败?}
        RETRY -->|是| NORMAL_OUT
        RETRY -->|否| ANSWER[最终回答]
        NORMAL_OUT --> ANSWER
    end

    style OFFLINE fill:#e1f0ff
    style ONLINE fill:#f0ffe1
    style GENERATION fill:#fff4e1
```

---

## 5. Skills 工具编排图

```mermaid
flowchart LR
    subgraph DEFINITIONS["技能定义 (definitions/*.json)"]
        S1[knowledge_profile]
        S2[market_latest_price]
        S3[market_history_price]
        S4[market_price_advice]
        S5[market_multi_item_compare]
        S6[place_profit_rank]
        S7[profit_stability]
        S8[answer_composer]
    end

    subgraph REGISTRY["Skills注册表"]
        SCORE[评分计算器]
        PLAN[计划生成器]
    end

    DEFINITIONS --> REGISTRY

    subgraph INPUTS["输入信号"]
        I1[intent]
        I2[selected_tool]
        I3[entities]
        I4[query_keywords]
    end

    INPUTS --> SCORE

    SCORE --> CALC[技能匹配评分]
    CALC --> C1[intent_hints: +6]
    CALC --> C2[tool_hints: +4]
    CALC --> C3[query_keywords_any: max+3]
    CALC --> C4[query_keywords_all: +2]
    CALC --> C5[实体数量评估]

    C1 --> TOTAL[总分计算]
    C2 --> TOTAL
    C3 --> TOTAL
    C4 --> TOTAL
    C5 --> TOTAL

    TOTAL --> SELECT[选择最佳技能]
    SELECT --> CONFIDENCE[计算置信度<br/>min(0.99, score/12.0)]

    CONFIDENCE --> THRESHOLD{置信度 >= 阈值?}
    THRESHOLD -->|是| MATCH[匹配成功]
    THRESHOLD -->|否| FALLBACK[降级到通用技能]

    MATCH --> PLAN
    FALLBACK --> PLAN

    PLAN --> OUTPUT[SkillPlan]
    OUTPUT --> OP1[tool_chain: 工具链]
    OUTPUT --> OP2[locked: 是否锁定]
    OUTPUT --> OP3[requires_task_planning]
    OUTPUT --> OP4[requires_specialist_analysis]

    subgraph EXECUTION["执行路径"]
        LOCKED{locked?}
        LOCKED -->|是| DIRECT[直接使用工具链]
        LOCKED -->|否| TASK_PLANNING[进入task_planning<br/>LLM二次规划]
        TASK_PLANNING --> DIRECT
    end

    OP2 --> EXECUTION

    style DEFINITIONS fill:#e1f0ff
    style REGISTRY fill:#f0ffe1
    style INPUTS fill:#fff4e1
    style EXECUTION fill:#ffe1f0
```

---

## 6. 工具与服务调用链图

```mermaid
flowchart TD
    subgraph AGENT["Agent层"]
        EXEC[execution_agent]
        INTENT[intent_recognition]
        PLAN[task_planning]
    end

    subgraph REGISTRY["工具注册表 (ToolRegistry)"]
        T1[rag_knowledge_search]
        T2[df_market_latest_price]
        T3[df_market_history_price]
        T4[df_market_price_advice]
        T5[df_place_profit_rank]
        T6[df_multi_item_compare]
        T7[df_profit_stability]
        T8[df_answer_composer]
    end

    EXEC --> REGISTRY
    INTENT --> REGISTRY
    PLAN --> REGISTRY

    subgraph SERVICES["服务层"]
        RAG[RAGService]
        PRICE[DFPriceService]
    end

    T1 --> RAG
    T2 --> PRICE
    T3 --> PRICE
    T4 --> PRICE
    T5 --> PRICE
    T6 --> PRICE
    T7 --> PRICE
    T8 --> RAG
    T8 --> PRICE

    subgraph RAG_IMPL["RAG实现"]
        RAG_SYS[AdvancedGraphRAGSystem]
        ROUTER[intelligent_query_router]
        HYBRID[hybrid_retrieval]
        GRAPH[graph_rag_retrieval]
        GEN[generation_integration]
    end

    RAG --> RAG_SYS
    RAG_SYS --> ROUTER
    ROUTER --> HYBRID
    ROUTER --> GRAPH
    HYBRID --> GEN
    GRAPH --> GEN

    subgraph PRICE_IMPL["市场服务实现"]
        CACHE[物品ID缓存]
        RESOLVE[resolve_object_id]
        API[外部API调用]
        RETRY[重试机制]
    end

    PRICE --> CACHE
    PRICE --> RESOLVE
    RESOLVE --> API
    API --> RETRY
    RETRY --> PRICE

    style AGENT fill:#ffe1e1
    style REGISTRY fill:#fff4e1
    style SERVICES fill:#e1f0ff
    style RAG_IMPL fill:#f0ffe1
    style PRICE_IMPL fill:#f0e1ff
```

---

## 7. 数据流向图

```mermaid
flowchart LR
    subgraph INPUT["输入层"]
        USER[用户输入]
        USER_ID[user_id]
        SESSION[session_id]
    end

    subgraph MEMORY["记忆层"]
        SHORT[SessionMemory<br/>短期记忆]
        LONG[PersistentMemoryStore<br/>长期记忆]
    end

    subgraph CORE["核心处理层"]
        INTENT[intent_recognition]
        PLAN[task_planning]
        EXEC[execution]
        SPEC[specialist_analysis]
        SUM[summary]
    end

    subgraph TOOLS["工具层"]
        RAG[RAG工具]
        MARKET[市场工具]
    end

    subgraph OUTPUT["输出层"]
        ANSWER[最终回答]
        MEMORY_UPDATE[记忆更新]
    end

    USER --> INTENT
    USER_ID --> SHORT
    SESSION --> SHORT
    SHORT --> INTENT
    LONG --> INTENT

    INTENT --> PLAN
    PLAN --> EXEC
    EXEC --> RAG
    EXEC --> MARKET
    RAG --> EXEC
    MARKET --> EXEC

    EXEC --> SPEC
    SPEC --> SUM
    SUM --> ANSWER

    ANSWER --> MEMORY_UPDATE
    MEMORY_UPDATE --> SHORT
    MEMORY_UPDATE --> LONG

    style INPUT fill:#e1f5e1
    style MEMORY fill:#e1f0ff
    style CORE fill:#fff4e1
    style TOOLS fill:#f0ffe1
    style OUTPUT fill:#ffe1f0
```

---

## 使用说明

以上图表可以从不同视角展示系统架构：

1. **主流程图** - 完整展示从用户输入到最终输出的全链路
2. **意图识别详细流程** - 展示四层意图识别机制
3. **记忆系统架构** - 展示短期和长期记忆的完整设计
4. **GraphRAG子系统** - 展示离线构建和在线服务的完整流程
5. **Skills工具编排** - 展示技能匹配和执行路径
6. **工具与服务调用链** - 展示从Agent到服务的调用层级
7. **数据流向图** - 展示数据在各层之间的流动

这些图表可以与 TECHNICAL_ARCHITECTURE.md 配合使用，提供更直观的系统理解。
