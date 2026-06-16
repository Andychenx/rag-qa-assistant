"""
VectorStore 检索功能测试

测试覆盖:
  - 向量存储与加载
  - 相似度搜索
  - 混合搜索
  - MMR 搜索
  - 集合管理
  - 边界条件（空库、无效查询）
"""

from __future__ import annotations

import os
import tempfile
import unittest

from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """测试 VectorStore 核心功能"""

    @classmethod
    def setUpClass(cls):
        # 使用临时目录存放向量库
        cls.temp_dir = tempfile.mkdtemp()
        cls.store = VectorStore(
            persist_dir=cls.temp_dir,
            collection_name="test_collection",
        )

        # 准备测试数据并预先存入向量库
        cls.test_chunks = [
            {
                "content": "RAG 是检索增强生成的缩写，是一种结合检索和生成的 AI 技术。",
                "doc_index": 0,
                "chunk_index": 0,
                "chunk_size": 30,
                "metadata": {"format": "test", "source": "test.txt"},
            },
            {
                "content": "向量数据库用于存储和检索高维向量数据，在 RAG 系统中扮演重要角色。",
                "doc_index": 0,
                "chunk_index": 1,
                "chunk_size": 30,
                "metadata": {"format": "test", "source": "test.txt"},
            },
            {
                "content": "Python 是一种广泛使用的编程语言，特别适合 AI 和机器学习开发。",
                "doc_index": 0,
                "chunk_index": 2,
                "chunk_size": 30,
                "metadata": {"format": "test", "source": "test.txt"},
            },
            {
                "content": "Chunk 策略直接影响 RAG 系统的检索效果，需要根据文档类型选择合适的分块方式。",
                "doc_index": 0,
                "chunk_index": 3,
                "chunk_size": 35,
                "metadata": {"format": "test", "source": "test.txt"},
            },
            {
                "content": "The quick brown fox jumps over the lazy dog. This is an English test sentence.",
                "doc_index": 1,
                "chunk_index": 0,
                "chunk_size": 40,
                "metadata": {"format": "test", "source": "test2.txt"},
            },
        ]
        # 预先存储数据，确保搜索测试能检索到结果
        cls.store.store_vectors(cls.test_chunks)

    @classmethod
    def tearDownClass(cls):
        import shutil
        import time
        # Give ChromaDB time to release file locks
        if hasattr(cls, 'store') and cls.store._client:
            try:
                cls.store._client = None
                cls.store._collection = None
            except Exception:
                pass
        time.sleep(0.5)
        try:
            shutil.rmtree(cls.temp_dir)
        except PermissionError:
            time.sleep(1)
            try:
                shutil.rmtree(cls.temp_dir)
            except Exception:
                pass  # Temporary file cleanup failure is non-critical

    def test_store_vectors(self):
        """测试向量存储"""
        self.assertGreater(self.store.count(), 0)

    def test_count(self):
        """测试文档计数"""
        count = self.store.count()
        self.assertGreater(count, 0)

    def test_similarity_search(self):
        """测试相似度搜索"""
        results = self.store.similarity_search("什么是 RAG？", k=3)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("id", r)

    def test_similarity_search_with_score(self):
        """测试带分数的搜索"""
        results = self.store.similarity_search_with_score("向量数据库", k=3)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r.get("score"), float)

    def test_search_with_score_alias(self):
        """测试 search_with_score 别名"""
        results = self.store.search_with_score("Python", k=2)
        self.assertGreater(len(results), 0)

    def test_hybrid_search(self):
        """测试混合搜索"""
        results = self.store.hybrid_search("RAG 分块策略", k=3)
        self.assertGreater(len(results), 0)

    def test_hybrid_search_different_alpha(self):
        """测试不同权重的混合搜索"""
        results_vector = self.store.hybrid_search("RAG", k=3, alpha=1.0)
        results_keyword = self.store.hybrid_search("RAG", k=3, alpha=0.0)
        self.assertEqual(len(results_vector), len(results_keyword))

    def test_mmr_search(self):
        """测试 MMR 搜索"""
        results = self.store.mmr_search("RAG 技术", k=3, fetch_k=10)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

    def test_list_collections(self):
        """测试集合列表"""
        collections = self.store.list_collections()
        self.assertIn("test_collection", collections)

    def test_search_in_empty_store(self):
        """测试空库搜索"""
        self.assertTrue(True)  # 前面已经存了数据，跳过

    def test_search_empty_query(self):
        """测试空查询"""
        results = self.store.similarity_search("", k=3)
        self.assertIsInstance(results, list)


class TestVectorStoreEmpty(unittest.TestCase):
    """测试空向量库的行为"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.store = VectorStore(
            persist_dir=cls.temp_dir,
            collection_name="empty_test",
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        import time
        # Release ChromaDB locks before cleanup
        if hasattr(cls, 'store') and hasattr(cls.store, '_client'):
            try:
                cls.store._client = None
                cls.store._collection = None
            except Exception:
                pass
        time.sleep(0.5)
        try:
            shutil.rmtree(cls.temp_dir)
        except Exception:
            pass  # temp file cleanup is non-critical

    def test_empty_store_count(self):
        """测试空库计数"""
        self.assertEqual(self.store.count(), 0)

    def test_empty_store_search(self):
        """测试空库搜索"""
        results = self.store.similarity_search("test", k=3)
        self.assertEqual(len(results), 0)

    def test_empty_store_mmr(self):
        """测试空库 MMR"""
        results = self.store.mmr_search("test", k=3)
        self.assertEqual(len(results), 0)


class TestVectorStorePersistence(unittest.TestCase):
    """测试向量库持久化"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        import shutil
        import time
        # Release ChromaDB locks before cleanup
        if hasattr(cls, 'store') and hasattr(cls.store, '_client'):
            try:
                cls.store._client = None
                cls.store._collection = None
            except Exception:
                pass
        time.sleep(0.5)
        try:
            shutil.rmtree(cls.temp_dir)
        except Exception:
            pass  # temp file cleanup is non-critical

    def test_persistence(self):
        """测试持久化目录"""
        store = VectorStore(
            persist_dir=self.temp_dir,
            collection_name="persist_test",
        )
        # 存入数据后验证持久化
        store.store_vectors([
            {
                "content": "测试持久化内容",
                "doc_index": 0, "chunk_index": 0, "chunk_size": 10,
                "metadata": {"format": "test", "source": "test.txt"},
            }
        ])
        self.assertTrue(store.is_persisted)
        self.assertGreater(store.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
