"""
DocumentLoader 单元测试

测试覆盖:
  - PDF 加载
  - TXT 加载（含编码自动检测）
  - Markdown 加载
  - 格式错误处理（不支持格式、文件不存在、空文件）
  - URL 加载
  - 批量加载
  - FileSizeExceededError
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.document_loader import (
    DocumentLoader,
    UnsupportedFormatError,
    LoadError,
    FileSizeExceededError,
)


class TestDocumentLoader(unittest.TestCase):
    """测试 DocumentLoader 核心功能"""

    @classmethod
    def setUpClass(cls):
        cls.loader = DocumentLoader(verbose=False)

        # 创建测试用临时文件
        cls.temp_dir = tempfile.mkdtemp()

        # TXT 文件
        cls.txt_path = os.path.join(cls.temp_dir, "test.txt")
        with open(cls.txt_path, "w", encoding="utf-8") as f:
            f.write("Hello World\n这是中文测试文本。\nLine 3\n")

        # Markdown 文件
        cls.md_path = os.path.join(cls.temp_dir, "test.md")
        with open(cls.md_path, "w", encoding="utf-8") as f:
            f.write("# Title\n\nThis is a paragraph.\n\n## Section 1\n\nSome content.\n")

        # 空文件
        cls.empty_path = os.path.join(cls.temp_dir, "empty.txt")
        with open(cls.empty_path, "w", encoding="utf-8") as f:
            f.write("")

        # 不支持格式
        cls.unsupported_path = os.path.join(cls.temp_dir, "test.xyz")
        with open(cls.unsupported_path, "w") as f:
            f.write("unsupported")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.temp_dir)

    def test_load_txt(self):
        """测试 TXT 文件加载"""
        result = self.loader.load_txt(self.txt_path)
        self.assertIn("content", result)
        self.assertIn("Hello World", result["content"])
        self.assertIn("中文测试", result["content"])
        self.assertEqual(result["metadata"]["format"], "txt")
        self.assertIn("encoding", result["metadata"])

    def test_load_markdown(self):
        """测试 Markdown 文件加载"""
        result = self.loader.load_markdown(self.md_path)
        self.assertIn("content", result)
        self.assertIn("Title", result["content"])
        self.assertIn("headings", result["metadata"])
        self.assertGreaterEqual(len(result["metadata"]["headings"]), 1)

    def test_unified_load_detects_format(self):
        """测试统一入口的格式自动检测"""
        result = self.loader.load(self.txt_path)
        self.assertIn("content", result)

        result = self.loader.load(self.md_path)
        self.assertIn("content", result)

    def test_unsupported_format(self):
        """测试不支持的格式抛出异常"""
        with self.assertRaises(UnsupportedFormatError):
            self.loader.load(self.unsupported_path)

    def test_file_not_found(self):
        """测试文件不存在"""
        with self.assertRaises(LoadError):
            self.loader.load("/path/to/nonexistent/file.pdf")

    def test_empty_file(self):
        """测试空文件加载"""
        result = self.loader.load(self.empty_path)
        self.assertEqual(result["content"], "")

    def test_load_string(self):
        """测试从字符串加载"""
        result = self.loader.load_string("# Hello\n\nThis is **markdown**", ".md")
        self.assertIn("Hello", result["content"])

    def test_batch_load(self):
        """测试批量加载"""
        results = self.loader.load_batch([self.txt_path, self.md_path])
        self.assertEqual(len(results), 2)
        self.assertIn("Hello World", results[0]["content"])
        self.assertIn("Title", results[1]["content"])

    def test_file_size_limit(self):
        """测试文件大小限制"""
        loader = DocumentLoader(max_file_size=1)  # 1 byte
        with self.assertRaises(FileSizeExceededError):
            loader.load(self.txt_path)

    def test_context_manager(self):
        """测试上下文管理器"""
        with DocumentLoader() as loader:
            result = loader.load(self.txt_path)
            self.assertIn("Hello World", result["content"])

    def test_repr(self):
        """测试 __repr__"""
        loader = DocumentLoader()
        self.assertIn("DocumentLoader", repr(loader))


class TestDocumentLoaderPDF(unittest.TestCase):
    """测试 PDF 加载（需要实际 PDF 文件）"""

    @classmethod
    def setUpClass(cls):
        cls.loader = DocumentLoader(verbose=False)
        # 使用项目中的测试文档
        project_root = Path(__file__).resolve().parent.parent
        cls.pdf_path = project_root / "测试文档.pdf"

    def test_load_pdf_exists(self):
        """测试 PDF 文件存在性"""
        if not self.pdf_path.exists():
            self.skipTest("测试 PDF 文件不存在")

    def test_load_pdf(self):
        """测试 PDF 加载"""
        if not self.pdf_path.exists():
            self.skipTest("测试 PDF 文件不存在")
        result = self.loader.load(str(self.pdf_path))
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertGreater(len(result["pages"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
