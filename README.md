# DeltaForce_Agent
# 说明

个人学习所建项目，欢迎提建议，更新中！

目标：GraphRAG + Multi-Agent + SFT LoRA/QLoRA RLHF


# TODO

## 功能

- ☐ 三角洲知识问答助手
- ☐ 三角洲战备智能体（鼠鼠玩家、正常玩家、猛攻玩家）
- ☐ 三角洲改枪大师（你说需求，Agent根据数据给出改枪方案）
- ...

## 数据

- ✅ 地图相关（地图区域、钥匙卡）
- ✅ 干员（角色名称、技能等）
- ☐ 收藏品
- ☐ 枪械、配件
- ...

## 其他

- ☐ 支持本地LLM模型调用
- ☐ 微调
- ☐ Post-Training
- ☐ LangChain 1.0 升级
- ...



---
# 当前版本
<details>
<summary><b> V0.1.0 </b></summary>

## DeltaForce GraphRAG知识问答系统

## 快速开始

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

### 示例问题

- 蜂衣有什么用
- 东楼经理室我该去哪里使用

## 样例展示

### 示例1：蜂衣有什么用

![示例1](data/images/example_1.png)

### 示例2：东楼经理室房卡使用

![示例2](data/images/example_2.png)

## 技术栈

- Neo4j - 图数据库
- Milvus - 向量数据库
- BAAI/bge-small-zh-v1.5 - 嵌入模型
- Kimi API - 大语言模型

</details>


