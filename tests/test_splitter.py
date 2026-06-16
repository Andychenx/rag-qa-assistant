"""
TextSplitter 单元测试

测试覆盖:
  - recursive_split 基本功能
  - semantic_split 功能
  - token_split 功能
  - markdown_header_split 功能
  - 空文档处理
  - 单行文档处理
  - 超长文档处理
  - 统计信息
  - split_documents 快捷函数
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from src.splitter import TextSplitter, split_documents


def _make_doc(content: str, **meta) -> Dict[str, Any]:
    """创建测试用文档字典"""
    return {
        "content": content,
        "metadata": {"format": "test", "source": "test.txt", **meta},
    }


class TestTextSplitter(unittest.TestCase):
    """测试 TextSplitter 核心功能"""

    def setUp(self):
        self.simple_docs = [
            _make_doc("Hello World. This is a test document. " * 10),
        ]

        self.multi_docs = [
            _make_doc("First document. " * 20, source="doc1.txt"),
            _make_doc("Second document. " * 20, source="doc2.txt"),
        ]

        self.markdown_doc = [
            _make_doc(
                "# Title\n\nIntro paragraph. " + "A" * 500 + "\n\n"
                "## Section 1\n\nContent 1. " + "B" * 800 + "\n\n"
                "### Subsection\n\nDetail A.\n\n"
                "## Section 2\n\nContent 2." + "C" * 800 + "\n\n"
                "```python\nprint('hello')\n```\n",
            ),
        ]

        self.long_paragraph_doc = [
            _make_doc("测试句子。" * 500),
        ]

    def test_recursive_split_basic(self):
        """测试递归分块基本功能"""
        chunks = TextSplitter.recursive_split(self.simple_docs, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("chunk_index", chunk)
            self.assertIn("doc_index", chunk)
            self.assertIn("metadata", chunk)
            self.assertLessEqual(len(chunk["content"]), 120)  # chunk_size + small tolerance

    def test_recursive_split_no_overlap(self):
        """测试无重叠分块"""
        chunks = TextSplitter.recursive_split(self.simple_docs, chunk_size=100, chunk_overlap=0)
        self.assertGreater(len(chunks), 1)

    def test_recursive_split_single_chunk(self):
        """测试小文档只分一个块"""
        docs = [_make_doc("Small text")]
        chunks = TextSplitter.recursive_split(docs, chunk_size=1000)
        self.assertEqual(len(chunks), 1)

    def test_recursive_split_multi_docs(self):
        """测试多文档分块"""
        chunks = TextSplitter.recursive_split(self.multi_docs, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        doc_indices = set(c["doc_index"] for c in chunks)
        self.assertEqual(len(doc_indices), 2)

    def test_semantic_split(self):
        """测试语义分块"""
        chunks = TextSplitter.semantic_split(self.markdown_doc, max_chunk_size=800)
        self.assertGreater(len(chunks), 1)

    def test_semantic_split_preserves_headings(self):
        """测试语义分块保留标题"""
        chunks = TextSplitter.semantic_split(self.markdown_doc, max_chunk_size=500)
        contents = " ".join(c["content"] for c in chunks)
        self.assertIn("Title", contents)
        self.assertIn("Section 1", contents)
        self.assertIn("Section 2", contents)

    def test_token_split(self):
        """测试 Token 分块"""
        chunks = TextSplitter.token_split(self.simple_docs, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.get("size_type"), "token")

    def test_markdown_header_split(self):
        """测试 Markdown 标题分块"""
        chunks = TextSplitter.markdown_header_split(self.markdown_doc)
        self.assertGreater(len(chunks), 1)
        # 应该按标题分割
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            self.assertIn("content", chunk)

    def test_empty_docs(self):
        """测试空文档列表"""
        chunks = TextSplitter.recursive_split([])
        self.assertEqual(len(chunks), 0)

    def test_empty_content(self):
        """测试空内容文档"""
        docs = [_make_doc("")]
        chunks = TextSplitter.recursive_split(docs)
        self.assertEqual(len(chunks), 0)

    def test_stats(self):
        """测试统计信息"""
        chunks = TextSplitter.recursive_split(self.simple_docs, chunk_size=100, chunk_overlap=20)
        stats = TextSplitter.stats(chunks)
        self.assertIn("count", stats)
        self.assertIn("avg_chunk_size", stats)
        self.assertIn("min_chunk_size", stats)
        self.assertIn("max_chunk_size", stats)
        self.assertEqual(stats["count"], len(chunks))

    def test_stats_empty(self):
        """测试空分块的统计"""
        stats = TextSplitter.stats([])
        self.assertEqual(stats["count"], 0)

    def test_split_documents_shortcut(self):
        """测试 split_documents 快捷函数"""
        chunks = split_documents(self.simple_docs, strategy="recursive", chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 0)

    def test_split_documents_unknown_strategy(self):
        """测试未知策略抛出异常"""
        with self.assertRaises(ValueError):
            split_documents(self.simple_docs, strategy="unknown")

    def test_large_chunk_overlap(self):
        """测试重叠小于块大小的边界情况"""
        docs = [_make_doc("A" * 50)]
        chunks = TextSplitter.recursive_split(docs, chunk_size=30, chunk_overlap=10)
        self.assertGreater(len(chunks), 0)

    def test_force_split_large_paragraph(self):
        """测试超长段落的强制分割"""
        long_text = "测试句子。" * 500
        docs = [_make_doc(long_text)]
        chunks = TextSplitter.semantic_split(docs, max_chunk_size=500)
        self.assertGreater(len(chunks), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
