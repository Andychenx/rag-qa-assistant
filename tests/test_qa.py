"""
RAGChain / QueryTransformer / Reranker 单元测试

测试覆盖:
  - RAGChain 初始化
  - ConversationMemory 功能
  - QueryTransformer 查询重写
  - QueryTransformer 子查询分解
  - Reranker 关键词加权
  - 边界条件处理
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from src.qa_chain import RAGChain, ConversationMemory
from src.query_transform import QueryTransformer
from src.reranker import Reranker


class TestConversationMemory(unittest.TestCase):
    """测试对话记忆"""

    def setUp(self):
        self.memory = ConversationMemory(max_rounds=10)

    def test_add_and_get(self):
        """测试添加和获取消息"""
        self.memory.add("user", "你好")
        self.memory.add("assistant", "你好！有什么可以帮你的？")
        history = self.memory.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "你好")

    def test_get_recent(self):
        """测试获取最近消息"""
        for i in range(10):
            self.memory.add("user", f"问题{i}")
            self.memory.add("assistant", f"回答{i}")

        recent = self.memory.get_recent(4)
        self.assertEqual(len(recent), 4)

    def test_clear(self):
        """测试清空"""
        self.memory.add("user", "test")
        self.memory.clear()
        self.assertEqual(len(self.memory), 0)

    def test_max_rounds_crop(self):
        """测试超出上限的裁剪"""
        memory = ConversationMemory(max_rounds=3)
        for i in range(10):
            memory.add("user", f"问题{i}")
            memory.add("assistant", f"回答{i}")

        # 应裁剪到 max_rounds*2 条以内
        self.assertLessEqual(len(memory), 6)

    def test_formatted_history(self):
        """测试格式化历史"""
        self.memory.add("user", "你好")
        self.memory.add("assistant", "你好！")
        formatted = self.memory.get_formatted_history()
        self.assertIn("用户", formatted)
        self.assertIn("助手", formatted)

    def test_to_dict_list(self):
        """测试转 OpenAI 格式"""
        self.memory.add("user", "test")
        dict_list = self.memory.to_dict_list()
        self.assertEqual(len(dict_list), 1)
        self.assertIn("role", dict_list[0])
        self.assertIn("content", dict_list[0])


class TestRAGChain(unittest.TestCase):
    """测试 RAGChain"""

    def test_init_without_retriever(self):
        """测试无检索器的初始化"""
        chain = RAGChain()
        self.assertIsNotNone(chain)
        self.assertIsNone(chain.retriever)

    def test_init_with_retriever(self):
        """测试带检索器的初始化"""
        mock_retriever = MagicMock()
        chain = RAGChain(retriever=mock_retriever)
        self.assertEqual(chain.retriever, mock_retriever)

    def test_build_methods(self):
        """测试构建方法"""
        chain = RAGChain()
        self.assertIsNotNone(chain.build_basic_chain())
        self.assertIsNotNone(chain.build_conv_chain())
        self.assertIsNotNone(chain.build_with_reranker())

    def test_clear_memory(self):
        """测试清空记忆"""
        chain = RAGChain()
        chain.memory.add("user", "test")
        chain.clear_memory()
        self.assertEqual(len(chain.memory), 0)

    def test_get_memory(self):
        """测试获取记忆"""
        chain = RAGChain()
        memory = chain.get_memory()
        self.assertIsInstance(memory, ConversationMemory)

    def test_format_context(self):
        """测试上下文格式化"""
        chain = RAGChain()
        docs = [
            {"content": "测试文档1", "metadata": {"source": "test1"}, "id": "chunk_0", "score": 0.9},
            {"content": "测试文档2", "metadata": {"source": "test2"}, "id": "chunk_1", "score": 0.8},
        ]
        context = chain._format_context(docs)
        self.assertIn("测试文档1", context)
        self.assertIn("测试文档2", context)
        self.assertIn("来源 1", context)
        self.assertIn("来源 2", context)

    def test_format_context_empty(self):
        """测试空文档格式化"""
        chain = RAGChain()
        context = chain._format_context([])
        self.assertIn("未检索到", context)


class TestReranker(unittest.TestCase):
    """测试重排序器"""

    def setUp(self):
        self.reranker = Reranker()
        self.docs = [
            {"content": "Python 是一种编程语言，用于 AI 开发", "score": 0.8},
            {"content": "Java 也是一种编程语言，用于企业开发", "score": 0.7},
            {"content": "今天天气很好，适合外出散步", "score": 0.6},
        ]

    def test_keyword_boost(self):
        """测试关键词加权重排序"""
        results = self.reranker.keyword_boost("Python AI", self.docs)
        self.assertEqual(len(results), 3)
        # 包含 AI/Python 关键词的文档应排名靠前
        self.assertIn("Python", results[0]["content"])

    def test_rerank_uniform(self):
        """测试统一接口"""
        results = self.reranker.rerank("Python", self.docs, method="keyword_boost", top_k=2)
        self.assertEqual(len(results), 2)

    def test_rerank_empty(self):
        """测试空文档"""
        results = self.reranker.rerank("test", [], method="keyword_boost")
        self.assertEqual(len(results), 0)

    def test_extract_keywords(self):
        """测试关键词提取"""
        keywords = Reranker._extract_keywords("Python 和 Java 的性能对比")
        self.assertGreater(len(keywords), 0)
        keyword_texts = [k[0] for k in keywords]
        self.assertIn("python", keyword_texts)
        self.assertIn("java", keyword_texts)


class TestQueryTransformer(unittest.TestCase):
    """测试查询转换器"""

    def setUp(self):
        self.transformer = QueryTransformer(llm=None)  # 无 LLM，测试基础功能

    @patch.object(QueryTransformer, '_create_llm', return_value=None)
    def test_rewrite_without_llm(self, mock_create_llm):
        """测试无 LLM 时的查询重写回退"""
        # 重新创建 transformer 使其不持有缓存
        transformer = QueryTransformer(llm=None)
        result = transformer.rewrite_query("它的原理是什么", history=[{"role": "user", "content": "什么是 RAG？"}])
        self.assertEqual(result, "它的原理是什么")  # 历史太少应直接返回

    def test_rewrite_no_history(self):
        """测试无历史时的查询重写"""
        result = self.transformer.rewrite_query("什么是 RAG？")
        self.assertEqual(result, "什么是 RAG？")

    @patch.object(QueryTransformer, '_create_llm', return_value=None)
    def test_sub_queries_without_llm(self, mock_create_llm):
        """测试无 LLM 时的子查询回退"""
        transformer = QueryTransformer(llm=None)
        result = transformer.generate_sub_queries("对比 Python 和 Java")
        self.assertEqual(result, ["对比 Python 和 Java"])

    def test_expand_query_without_llm(self):
        """测试无 LLM 时的查询扩展"""
        expanded = self.transformer.expand_query("RAG 原理")
        self.assertIn("RAG 原理", expanded)

    @patch.object(QueryTransformer, '_create_llm', return_value=None)
    def test_stepback_without_llm(self, mock_create_llm):
        """测试无 LLM 时的退后提示"""
        transformer = QueryTransformer(llm=None)
        result = transformer.stepback_query("RecursiveCharacterTextSplitter chunk_size 设置")
        self.assertEqual(result, "RecursiveCharacterTextSplitter chunk_size 设置")


if __name__ == "__main__":
    unittest.main(verbosity=2)
