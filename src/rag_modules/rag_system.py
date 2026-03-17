"""RAG 系统实现与调试入口。"""

import time
import logging
from typing import List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules.graph_data_preparation import GraphDataPreparationModule
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
from rag_modules.generation_integration import GenerationIntegrationModule
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter


class AdvancedGraphRAGSystem:
    """
    图RAG系统
    
    核心特性：
    1. 智能路由：自动选择最适合的检索策略
    2. 双引擎检索：传统混合检索 + 图RAG检索
    3. 图结构推理：多跳遍历、子图提取、关系推理
    4. 查询复杂度分析：深度理解用户意图
    5. 自适应学习：基于反馈优化系统性能
    """
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or DEFAULT_CONFIG
        
        # 核心模块
        self.data_module = None
        self.index_module = None
        self.generation_module = None
        
        # 检索引擎
        self.traditional_retrieval = None
        self.graph_rag_retrieval = None
        self.query_router = None
        
        # 系统状态
        self.system_ready = False
        
    def initialize_system(self, enable_qa_modules: bool = True):
        """初始化高级图RAG系统"""
        logger.info("启动三角洲行动图RAG系统...")
        
        try:
            # 1. 数据准备模块
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )
            
            # 2. 向量索引模块
            print("初始化Milvus向量索引...")
            self.index_module = MilvusIndexConstructionModule(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.milvus_collection_name,
                dimension=self.config.milvus_dimension,
                model_name=self.config.embedding_model
            )
            
            if enable_qa_modules:
                self._initialize_qa_modules()
            
            print("✅ 高级图RAG系统初始化完成！")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def _initialize_qa_modules(self):
        """初始化问答相关模块（LLM + 路由 + 检索）"""
        if self.generation_module and self.traditional_retrieval and self.graph_rag_retrieval and self.query_router:
            return

        print("初始化生成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("初始化传统混合检索...")
        self.traditional_retrieval = HybridRetrievalModule(
            config=self.config,
            milvus_module=self.index_module,
            data_module=self.data_module,
            llm_client=self.generation_module.llm
        )

        print("初始化图RAG检索引擎...")
        self.graph_rag_retrieval = GraphRAGRetrieval(
            config=self.config,
            llm_client=self.generation_module.llm
        )

        print("初始化智能查询路由器...")
        self.query_router = IntelligentQueryRouter(
            traditional_retrieval=self.traditional_retrieval,
            graph_rag_retrieval=self.graph_rag_retrieval,
            llm_client=self.generation_module.llm,
            config=self.config
        )
    
    def build_knowledge_base(self, force_rebuild: bool = False, initialize_retrievers: bool = True):
        """离线构建知识库（可选初始化检索器）"""
        print("\n开始离线构建知识库...")

        try:
            # build 模式默认不覆盖已有库；rebuild 模式才强制重建
            if not force_rebuild and self.index_module.has_collection():
                print("✅ 检测到已存在的Milvus集合，build模式跳过重建。")
                print("如需强制覆盖请使用 rebuild 模式。")

                if not self.index_module.load_collection():
                    raise RuntimeError("已有集合存在但加载失败，请检查Milvus状态后再试。")

                print("加载图数据以更新离线统计...")
                self.data_module.load_graph_data()
                self.data_module.build_entity_documents()
                self.data_module.chunk_documents(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                self._show_knowledge_base_stats()
                return

            # 从Neo4j加载图数据
            print("从Neo4j加载图数据...")
            self.data_module.load_graph_data()
            
            # 构建实体文档
            print("构建实体文档...")
            self.data_module.build_entity_documents()
            
            # 进行文档分块
            print("进行文档分块...")
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            # 构建Milvus向量索引
            print("构建Milvus向量索引...")
            if force_rebuild:
                print("强制重建模式：将覆盖现有Milvus集合。")
            if not self.index_module.build_vector_index(
                chunks,
                force_recreate=force_rebuild,
                load_after_build=initialize_retrievers
            ):
                raise Exception("构建向量索引失败")
            
            # 在线模式才需要初始化检索器
            if initialize_retrievers:
                self._initialize_retrievers(chunks)
            
            # 显示统计信息
            self._show_knowledge_base_stats()
            
            print("✅ 知识库构建完成！")
            
        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise

    def load_knowledge_base_for_serving(self):
        """在线问答模式：只加载已有索引，不触发重建"""
        print("\n在线模式：加载已有知识库...")
        if not self.index_module.has_collection():
            raise RuntimeError(
                f"Milvus集合 '{self.config.milvus_collection_name}' 不存在。"
                "请先在 config.py 设置 run_mode='build' 并运行一次离线建库。"
            )

        if not self.index_module.load_collection():
            raise RuntimeError("已有知识库加载失败，请检查Milvus状态。")

        print("加载图数据以支持图检索...")
        self.data_module.load_graph_data()
        print("构建实体文档...")
        self.data_module.build_entity_documents()
        print("进行文档分块...")
        chunks = self.data_module.chunk_documents(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        self._initialize_retrievers(chunks)
        self._show_knowledge_base_stats()
        print("✅ 在线知识库加载完成！")
    
    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器"""
        print("初始化检索引擎...")
        self._initialize_qa_modules()
        
        # 如果没有chunks，从数据模块获取
        if chunks is None:
            chunks = self.data_module.chunks or []
        
        # 初始化传统检索器
        self.traditional_retrieval.initialize(chunks)
        
        # 初始化图RAG检索器
        self.graph_rag_retrieval.initialize()
        
        self.system_ready = True
        print("✅ 检索引擎初始化完成！")
    
    def _show_knowledge_base_stats(self):
        """显示知识库统计信息"""
        print(f"\n知识库统计:")
        
        # 数据统计
        stats = self.data_module.get_statistics()
        print(f"   实体数量: {stats.get('total_nodes', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")
        
        # Milvus统计
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")
        
        # 图RAG统计
        if self.query_router:
            route_stats = self.query_router.get_route_statistics()
            print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")
        else:
            print("   路由统计: 问答模块未初始化（离线建库模式）")
        
        if stats.get('label_counts'):
            labels = list(stats['label_counts'].keys())[:10]
            print(f"   🏷️ 主要类型: {', '.join(labels)}")
    
    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False):
        """
        智能问答：自动选择最佳检索策略
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
            
        print(f"\n❓ 用户问题: {question}")
        
        # 显示路由决策解释（可选）
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)
        
        start_time = time.time()
        
        try:
            # 1. 智能路由检索
            print("执行智能查询路由...")
            relevant_docs, analysis = self.query_router.route_query(question, self.config.top_k)
            
            # 2. 显示路由信息
            strategy_icons = {
                "hybrid_traditional": "🔍",
                "graph_rag": "🕸️", 
                "combined": "🔄"
            }
            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")
            
            # 3. 显示检索结果信息
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    entry_name = doc.metadata.get('recipe_name', doc.metadata.get('entity_name', '未知内容'))
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"{entry_name}({search_type}, {score:.3f})")
                
                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                # 保持返回值签名一致：始终返回 (result, analysis)
                return "抱歉，没有找到相关信息。请尝试其他问题。", analysis
            
            # 4. 生成回答
            print("🎯 智能生成回答...")
            
            if stream:
                try:
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出过程中出现错误: {stream_error}")
                    print(f"\n⚠️ 流式输出中断，切换到标准模式...")
                    # 使用非流式作为后备
                    result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            else:
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            
            # 5. 性能统计
            end_time = time.time()
            print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")
            
            return result, analysis
            
        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None
    

    

    
    def run_interactive(self):
        """运行交互式问答"""
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return
            
        print("\n欢迎使用三角洲行动图RAG助手！")
        print("可用功能：")
        print("   - 'stats' : 查看系统统计")
        print("   - 'rebuild' : 重建知识库")
        print("   - 'quit' : 退出系统")
        print("\n" + "="*50)
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'rebuild':
                    self._rebuild_knowledge_base()
                    continue
                
                # 普通问答 - 使用默认设置
                use_stream = True  # 默认使用流式输出
                explain_routing = False  # 默认不显示路由决策

                print("\n回答:")
                
                result, analysis = self.ask_question_with_routing(
                    user_input, 
                    stream=use_stream, 
                    explain_routing=explain_routing
                )
                
                if not use_stream and result:
                    print(f"{result}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n👋 感谢使用三角洲行动图RAG助手！")
        self._cleanup()
    
    def _show_system_stats(self):
        """显示系统统计信息"""
        print("\n系统运行统计")
        print("=" * 40)

        if not self.query_router:
            print("问答模块未初始化，当前模式不提供运行统计。")
            self._show_knowledge_base_stats()
            return
        
        # 路由统计
        route_stats = self.query_router.get_route_statistics()
        total_queries = route_stats.get('total_queries', 0)
        
        if total_queries > 0:
            print(f"总查询次数: {total_queries}")
            print(f"传统检索: {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})")
            print(f"图RAG检索: {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})")
            print(f"组合策略: {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})")
        else:
            print("暂无查询记录")
        
        # 知识库统计
        self._show_knowledge_base_stats()
    
    def _rebuild_knowledge_base(self):
        """重建知识库"""
        print("\n准备重建知识库...")
        
        # 确认操作
        confirm = input("⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 重建操作已取消")
            return
        
        try:
            print("删除现有的Milvus集合...")
            if self.index_module.delete_collection():
                print("✅ 现有集合已删除")
            else:
                print("删除集合时出现问题，继续重建...")
            
            # 重新构建知识库
            print("开始重建知识库...")
            self.build_knowledge_base(force_rebuild=True, initialize_retrievers=True)
            
            print("✅ 知识库重建完成！")
            
        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            print(f"❌ 重建失败: {e}")
            print("建议：请检查Milvus服务状态后重试")
    
    def _cleanup(self):
        """清理资源"""
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()

def run_rag_mode(mode: str | None = None):
    """
    RAG 调试入口，仅用于 RAG 子系统：
    - build
    - rebuild
    - serve
    """
    rag_system = AdvancedGraphRAGSystem()
    selected_mode = mode or rag_system.config.run_mode
    if selected_mode == "agent":
        selected_mode = "serve"

    print(f"RAG 调试模式: {selected_mode}")

    if selected_mode == "build":
        rag_system.initialize_system(enable_qa_modules=False)
        rag_system.build_knowledge_base(force_rebuild=False, initialize_retrievers=False)
        print("✅ 离线建库完成。")
        rag_system._cleanup()
        return

    if selected_mode == "rebuild":
        rag_system.initialize_system(enable_qa_modules=False)
        print("删除现有的Milvus集合...")
        if rag_system.index_module.delete_collection():
            print("✅ 现有集合已删除")
        else:
            print("删除集合时出现问题，继续重建...")
        rag_system.build_knowledge_base(force_rebuild=True, initialize_retrievers=False)
        print("✅ 离线重建完成。")
        rag_system._cleanup()
        return

    rag_system.initialize_system(enable_qa_modules=True)
    rag_system.load_knowledge_base_for_serving()
    rag_system.run_interactive()


if __name__ == "__main__":
    try:
        run_rag_mode()
    except Exception as e:
        logger.error(f"RAG 调试运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")
