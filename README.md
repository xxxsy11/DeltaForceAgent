# DeltaForce_Agent
# 说明

个人学习所建项目，欢迎提建议，更新中！

目标：GraphRAG + Multi-Agent + SFT LoRA/QLoRA RLHF


# TODO

## 功能

- ✅ 三角洲知识问答助手
- ☐ 三角洲战备智能体（鼠鼠玩家、正常玩家、猛攻玩家）
- ☐ 三角洲改枪大师（你说需求，Agent根据数据给出改枪方案）
- ...


## 📦 开源三角洲图数据（DeltaForce_GraphData）

**路径：** `data/neo4j/`

- ✅ 地图相关（map.json）- 地图、区域、钥匙等
- ✅ 干员数据（operator.json）- 角色名称、技能等
- ✅ 收藏品（collection.json）- 类别、名称、品质、重量、大小等
- ✅ 装备数据（equipment.json）- 名称、重量、耐久度、容量、最大联通格数等
- ✅ 枪械数据（firearms.json）- 枪械、类型、基础数值等
- ✅ 配件数据（attachments.json）- 配件（弹匣、瞄具、护木、枪管等）名称、类型、效果等
- ✅ 弹药数据（ammo.json）- 弹药名称、类型等

## 其他

- ✅ 支持本地LLM模型调用
- ✅ 微调
- ☐ Post-Training
- ✅ LangChain 1.0 升级
- ...

## 已实现功能

- 多智能体主流程：采用主 Agent 调度 + 子 Agent 分工，覆盖意图识别、任务规划、工具执行、专业分析、总结生成、质量审查与重试路由。
- 双 RAG 问答能力：同时支持知识库 GraphRAG 与长期记忆召回，支持地图、干员、收藏品、装备、枪械、配件、弹药等知识检索与生成（1379 个实体，1837 条关系）。
- 市场分析工具链（工具实现已开源）：支持最新价格、历史价格、买卖建议、特勤处制造利润榜（Top1/Top3）、多物品对比、利润稳定性分析与综合回答编排。
- LoRA-SFT 三模块接入：已将 Intent、Tool Selection、Planning 三个本地微调模块接入主流程，支持 Qwen3-8B 基座 + Adapter 切换。
- 记忆系统升级：短期记忆支持滚动压缩，长期记忆支持 PostgreSQL + pgvector 持久化召回，支持多轮连续对话与跨会话信息延续。
- 用户会话管理：支持多用户与多会话隔离，可切换用户、新建会话、查看记忆状态、清空当前会话记忆。
- Agent Skills 策略层：按问题类型选择更合适的处理路径，支持单工具与多工具链式调用，降低误路由概率。
- 异步链路优化：核心编排节点与模型调用链路完成异步化改造，减少阶段阻塞，提升并发与可维护性。
- 工具稳健性增强：新增参数校验与异常工具复核机制（tool selection review），在工具失败场景下优先重选工具，减少无效全链路回退。
- 质量审查与自动纠错：工具结果审查 + 回答审查 + 分阶段重试 + 重试预算控制，降低异常结果直接输出风险。
- 工程验证能力：提供统一测试与评测脚本，覆盖功能可用性、工具调用、双 RAG 召回、记忆门控、用户隔离、审查重试链路与 SFT 模块效果对比。

完整技术报告请查看：`docs/TECHNICAL_ARCHITECTURE.md`  
SFT 模块评测请查看：`docs/SFT_EVAL_REPORT.md`

## 关于实时数据

当前开源仓库暂未开源实时市场数据获取的具体实现方式。  
如有接入需求，欢迎提 Issue 说明场景；也可以通过其他协作方式获取实时数据接入支持。



---
# 当前版本

<details open>
<summary><b> V0.7.1 - 异步链路与可观测优化 </b></summary>

## 升级内容

- 异步化增强：将意图识别、工具选择、任务规划、总结生成、专业分析、记忆压缩等关键链路改造为异步调用路径，降低阻塞与阶段串行开销。
- 本地模型调用统一：本地 Qwen 运行时补充异步调用接口，主流程通过统一异步入口调度 Base 模型与 LoRA Adapter。
- 制造利润路由优化：对制造台相关问法（制造台、特勤处制造、技术中心、工作台、制药台、防具台）增强规则优先级，减少误选 `df_multi_item_compare` 的情况。
- 控制台可观测优化：阶段日志中文化输出，统一展示“正在执行什么 + 执行耗时 + 关键结果”；压低底层无关 INFO 日志噪声，便于在线排障。
- 文档与配置同步：更新技术文档中的流程说明与关键路径，保持实现与文档一致。

## 关键模块

- 编排与阶段追踪：`src/agents/graph.py`
- 交互入口与耗时输出：`src/agents/runner.py`
- 意图识别：`src/agents/intent_recognition.py`
- 规则分析器：`src/agents/intent_analyzer.py`
- 工具规划与总结：`src/agents/tool_planner.py`、`src/agents/summary_agent.py`
- 记忆压缩与持久化：`src/memory/memory_compression_agent.py`、`src/memory/persistent/`

## 运行与验证

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

</details>

---

<details>
<summary><b> V0.7.0 - LoRA-SFT 三模块接入（本地推理） </b></summary>

## 升级内容

- 新增本地模型运行时：支持 Qwen3-8B 基座模型单例加载与按需 LoRA Adapter 切换。
- 三模块 LoRA-SFT 正式接入主流程：
  - 意图识别模块（Intent）
  - 工具选择模块（Tool Selection）
  - 复杂任务规划模块（Planning）
- 意图识别节点升级：
  - 引入结构化输出校验（Pydantic）
  - Prompt 外置到模板文件，便于版本化与持续迭代
- 新增异常链路工具复核节点：当工具执行异常时，触发 `tool_selection_review` 进行工具重选，减少误回退到全流程重跑。
- 主流程可观测性增强：新增阶段级耗时追踪与关键路由信息记录。
- 价格工具新增参数校验：统一校验 `objectName/id/date/place/type/top` 等核心参数，降低异常输入导致的调用失败。

## 关键模块

- 本地模型运行时：`src/agents/local_qwen_runtime.py`
- 意图识别（LoRA）：`src/agents/intent_recognition.py`
- 工具选择/规划（LoRA）：`src/agents/tool_planner.py`
- 工具复核：`src/agents/tool_selection_review.py`
- 主流程编排：`src/agents/graph.py`
- Prompt 模板：`src/prompts/intent_recognition_prompt.txt`

## 运行与验证

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

</details>

---

<details>
<summary><b> V0.6.0 - 质量审查与自动纠错机制 </b></summary>

## 升级内容

- 新增工具结果校验：工具执行后先做结构化检查，识别失败文本、主体缺失、格式异常等问题。
- 新增回答审查：最终回答输出前进行质量审核，拦截“工具已成功但回答仍失败”这类冲突结果。
- 新增分阶段重试：失败后按问题类型回退到对应阶段（重识别、重规划、重执行或重总结），避免无效全链路重跑。
- 新增重试预算控制：为不同阶段设置重试上限，超限后给出可解释降级结果，防止死循环。
- 可观测性增强：记录审查结论、重试路径和关键中间状态，便于排查与复盘。

## 关键模块

- 工具结果校验：`src/agents/tool_output_validator.py`
- 回答质量审查：`src/agents/answer_reviewer.py`
- 重试路由控制：`src/agents/retry_router.py`
- 主流程编排：`src/agents/graph.py`

## 运行与验证

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py

# 全量系统测试（功能 + 指标 + 重试链路）
python data/scripts/system_full_metrics_suite.py
```

</details>

---

<details>
<summary><b> V0.5.0 - Skills能力与系统评测 </b></summary>

## 升级内容

- 新增 Skills 模块：按问题类别管理处理策略，统一定义工具选择、参数组织和执行链路。
- 意图识别接入 Skills：先识别意图，再根据技能规则选择工具计划；复杂问题可自动进入多工具链路。
- 任务规划联动升级：当技能计划已明确时直接执行，减少重复规划；当技能未锁定时继续保留规划能力。
- 执行与可观测升级：执行结果中增加技能命中信息，便于排查“为什么用这个工具”。
- 测试体系升级：
  - 新增 Skills 集成测试脚本：`data/scripts/integration_memory_tools_skills_suite.py`
  - 新增系统A/B基准脚本：`data/scripts/system_ab_benchmark.py`
  - 新增评测基线数据：`data/benchmarks/system_eval_cases.json`
  - 新增评测文档：`docs/INTEGRATION_TEST_SKILLS_REPORT.md`、`docs/SYSTEM_AB_BENCHMARK_REPORT.md`

## 关键模块

- Skills 定义：`src/skills/definitions/`
- Skills 选择与组装：`src/skills/registry.py`
- 意图识别接入：`src/agents/intent_recognition.py`
- 执行阶段记录：`src/agents/execution_agent.py`
- 任务规划联动：`src/agents/task_planning.py`

## 运行与验证

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py

# 集成测试（记忆 + 工具）
python data/scripts/integration_memory_tools_suite.py

# Skills 集成测试
python data/scripts/integration_memory_tools_skills_suite.py

# 系统A/B基准测试（skills_on / skills_off）
python data/scripts/system_ab_benchmark.py
```

</details>

---

<details>
<summary><b> V0.4.0 - 记忆系统与用户会话管理 </b></summary>

## 升级内容

- 新增短期记忆链路：`recent_raw`、`pending_buffer`、`rolling_summary`，支持多轮上下文连续对话。
- 新增长期记忆链路：基于 `PostgreSQL + pgvector` 的 `chat_turns`、`memory_summaries`、`memory_facts`。
- 新增记忆门控召回：根据问题特征打分，按需触发长期深召回（向量检索 + BM25 + RRF 融合）。
- 新增 `user_id` 维度：长短期记忆均按用户隔离，支持多用户并行使用。
- 交互能力升级：支持 `new session`、`switch user <id>`、`memory stats`、`clear memory`。
- 会话归档增强：会话结束时自动压缩并落盘，避免 `pending_buffer` 信息丢失。
- 测试能力升级：新增记忆与工具联合集成测试脚本，覆盖跨会话重入与 user 隔离验证。

## Agent Memory 逻辑

- 短期记忆（内存态）
  - `recent_raw`：保存最近原文对话窗口
  - `pending_buffer`：超窗后待压缩缓冲
  - `rolling_summary`：历史压缩摘要
- 长期记忆（持久化）
  - `chat_turns`：完整对话与工具输出
  - `memory_summaries`：阶段摘要快照
  - `memory_facts`：可复用事实条目
- 召回策略
  - 先构建短期上下文
  - 达到门控阈值时触发长期深召回
  - 未达阈值时仅补充长期最新摘要

## 关键模块

- 记忆模块：`src/memory/`
- 记忆召回节点：`src/memory/persistent_memory_recall_node.py`
- 记忆写入节点：`src/memory/persistent_memory_write_node.py`
- 会话管理：`src/agents/runner.py`
- 编排入口：`src/agents/graph.py`

## 运行与验证

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py

# 可选：初始化长期记忆数据库
bash data/scripts/init_postgres_memory.sh

# 可选：运行记忆+工具集成测试
python data/scripts/integration_memory_tools_suite.py
```

</details>

---

<details>
<summary><b> V0.3.0 - 主Agent + 子Agent 编排</b></summary>

## 升级内容

- 多智能体流程升级为主Agent调度 + 子Agent协作（LangGraph A2A）。
- 新增节点分工：`main_orchestrator -> intent_recognition -> (task_planning) -> execution -> (specialist_analysis) -> summary`。
- 将复杂问题与专业分析流程按需触发，简单问题走轻路径，降低不必要的调用开销。
- 文档补充：新增/更新 `docs/USAGE_GUIDE.md`，覆盖功能清单、底层调用和可复现示例。

## Multi-Agent 逻辑

- 主流程调度：`src/agents/main_orchestrator.py`
- 意图识别：`src/agents/intent_recognition.py`
- 任务规划（复杂问题）：`src/agents/task_planning.py`
- 工具执行：`src/agents/execution_agent.py`
- 专业分析（按需触发）：`src/agents/specialist_analysis.py`
- 总结输出：`src/agents/summary_agent.py`
- 图编排定义：`src/agents/graph.py`

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```


</details>

---

<details>
<summary><b> V0.2.0 - 工程结构重整 + Agent工具化 </b></summary>

## 升级内容

- 项目目录调整为标准结构：`main.py`、`config.py`、`src/`、`data/`、`docker/`、`docs/`
- 将 RAG 封装为可调用能力，接入工具层与 Agent 路由流程
- 根入口改为 `main.py`，默认运行 LangGraph Multi-Agent（意图识别后路由到工具）
- RAG build/serve/rebuild 调试入口统一放到 `src/rag_modules/rag_system.py`
- 新增统一数据脚本：`data/scripts/data_pipeline.py`
- 合并旧数据脚本能力（统计、导出、导入、清库、重建）
- 清理旧兼容目录和重复脚本，减少维护成本

## Quick Start

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入实际配置
```

### 2. 启动数据库

```bash
# 启动Neo4j
cd docker/neo4j
docker-compose up -d

# 启动Milvus
cd docker/milvus
docker-compose up -d
```

### 3. 运行系统

```bash
# 默认：Multi-Agent 模式（intent -> route -> tool）
python main.py

# RAG子系统调试模式
RAG_RUN_MODE=build PYTHONPATH=src python -m rag_modules.rag_system
RAG_RUN_MODE=serve PYTHONPATH=src python -m rag_modules.rag_system
RAG_RUN_MODE=rebuild PYTHONPATH=src python -m rag_modules.rag_system
```

### 4. 数据处理

```bash
python data/scripts/data_pipeline.py --help
python data/scripts/data_pipeline.py stats
python data/scripts/data_pipeline.py export-csv
```

## 使用说明

- `RAG_RUN_MODE=build`：离线建库
- `RAG_RUN_MODE=serve`：在线问答
- `RAG_RUN_MODE=rebuild`：重建向量索引
- `RAG_RUN_MODE=agent`：Multi-Agent 模式（默认）

</details>

---

<details>
<summary><b> V0.1.1 - 知识库升级 </b></summary>

## 升级内容

- 丰富知识库数据：新增枪械、配件、弹药数据
- 优化数据处理脚本，支持多文件批量处理
- 数据总量：1391个节点，1836条关系

## Quick Start

### 1. 环境准备

```bash
# 安装依赖
cd rag_app
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入实际配置
```

### 2. 启动数据库

```bash
# 启动Neo4j
cd docker/neo4j
docker-compose up -d

# 启动Milvus
cd docker/milvus
docker-compose up -d
```

### 3. 运行系统

```bash
cd rag_app
python main.py
```

## 使用说明

启动后进入交互式问答模式：

- 直接输入问题进行提问
- `stats` - 查看系统统计
- `rebuild` - 重建知识库
- `quit` - 退出系统

## 样例展示

### 示例1：给我介绍一下腾龙突击步枪

![配件查询](data/images/011_1.png)


### 示例2：有什么瞄具能够切换倍率的

![瞄具查询](data/images/011_2.png)


## 技术栈

- Neo4j - 图数据库
- Milvus - 向量数据库
- BAAI/bge-small-zh-v1.5 - 嵌入模型
- Kimi API - 大语言模型

</details>

---

<details>
<summary><b> V0.1.0 </b></summary>

## DeltaForce GraphRAG知识问答系统

基础版本，包含地图、干员、收藏品、装备数据。

## 样例展示

### 示例1：蜂衣有什么用

![示例1](data/images/010_1.png)

### 示例2：东楼经理室在哪用

![示例2](data/images/010_2.png)

## 技术栈

- Neo4j - 图数据库
- Milvus - 向量数据库
- BAAI/bge-small-zh-v1.5 - 嵌入模型
- Kimi API - 大语言模型

</details>


---
