"""
文本分块模块 — 多种分块策略，为 RAG 系统准备输入

技术栈:
  - langchain_text_splitters: RecursiveCharacterTextSplitter / TokenTextSplitter

核心策略对比:
  ┌──────────────────┬──────────────────────────────┬──────────────────────┐
  │ 策略              │ 适用场景                      │ 分块依据              │
  ├──────────────────┼──────────────────────────────┼──────────────────────┤
  │ recursive_split  │ 通用文本（新闻、文章、日志）      │ 递归字符级分割          │
  │ semantic_split   │ 结构化文档（MD、技术文档、报告）   │ 语义边界（标题/段落）     │
  │ token_split      │ LLM 输入准备（需对齐 token 数）   │ Token 计数分割          │
  └──────────────────┴──────────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


# ---------------------------------------------------------------------------
# TextSplitter
# ---------------------------------------------------------------------------

class TextSplitter:
    """文本分块器 — 提供多种分块策略的静态方法集合

    所有方法接受 ``docs`` 参数（DocumentLoader 输出的文档字典列表），
    返回统一的 chunk 列表，每个 chunk 包含:
        - content:      str       — 块文本
        - doc_index:    int       — 所属源文档索引
        - chunk_index:  int       — 全局块序号
        - chunk_size:   int       — 字符数（token 分块时为 token 数）
        - metadata:     Dict      — 继承自源文档的元数据
    """

    # ------------------------------------------------------------------
    # 1. 递归字符分块
    # ------------------------------------------------------------------

    @staticmethod
    def recursive_split(
        docs: Sequence[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """递归字符分块 — 通用策略，适合大部分文本

        原理:
            从高优先级分隔符（段落 ``\\n\\n``）开始切割，若块仍大于
            ``chunk_size`` 则递归使用下一级分隔符（句子、词），
            直到全部块满足大小要求或降至字符级切割。

        Args:
            docs:         文档字典列表（来自 DocumentLoader）
            chunk_size:   每块最大字符数
            chunk_overlap: 块间重叠字符数
            separators:   分隔符优先级列表（默认: 段落→行→句→空）

        Returns:
            分块列表
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", ".", "!", "?", " ", ""]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            add_start_index=True,
        )

        return TextSplitter._apply_splitter(docs, splitter, "char")

    # ------------------------------------------------------------------
    # 2. 语义分块
    # ------------------------------------------------------------------

    @staticmethod
    def semantic_split(
        docs: Sequence[Dict[str, Any]],
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """语义分块 — 基于自然语义边界（标题 / 段落）分块

        适合 Markdown、技术文档、学术论文等结构化内容。
        优先按标题（``#`` / ``##``）划分，再按段落边界细调。

        Args:
            docs:           文档字典列表
            max_chunk_size: 每块最大字符数（超出则强制分割）
            min_chunk_size: 每块最小字符数（低于此值合并到前一块）

        Returns:
            分块列表
        """
        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for doc_idx, doc in enumerate(docs):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # 尝试按 Markdown 标题分割
            sections = TextSplitter._split_by_headings(content)

            buffer = ""
            for section in sections:
                section = section.strip()
                if not section:
                    continue

                # buffer + section 仍未超出上限 → 合并
                if len(buffer) + len(section) <= max_chunk_size:
                    buffer = (buffer + "\n\n" + section) if buffer else section
                    continue

                # 超出上限 → 先 flush buffer
                if buffer:
                    chunks.append(
                        TextSplitter._make_chunk(
                            buffer, doc_idx, global_idx, metadata
                        )
                    )
                    global_idx += 1

                # section 本身超出 max_chunk_size → 硬切
                if len(section) > max_chunk_size:
                    sub_chunks = TextSplitter._force_split(section, max_chunk_size)
                    for sc in sub_chunks:
                        if len(sc) >= min_chunk_size or not buffer:
                            chunks.append(
                                TextSplitter._make_chunk(
                                    sc, doc_idx, global_idx, metadata
                                )
                            )
                            global_idx += 1
                        else:
                            # 太小 → 合并到上一个 chunk
                            if chunks:
                                chunks[-1]["content"] += "\n" + sc
                                chunks[-1]["chunk_size"] += len(sc)
                    continue

                buffer = section

            # 处理最后一段 buffer
            if buffer:
                if len(buffer) >= min_chunk_size:
                    chunks.append(
                        TextSplitter._make_chunk(
                            buffer, doc_idx, global_idx, metadata
                        )
                    )
                    global_idx += 1
                elif chunks:
                    # 太小且已有前一块 → 合并
                    chunks[-1]["content"] += "\n" + buffer
                    chunks[-1]["chunk_size"] += len(buffer)

        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> List[str]:
        """按 Markdown 标题分割文本，保留标题行"""
        # 匹配行首 # 标题（包括可能的前后空行）
        parts = re.split(r"(?=^#{1,6}\s+\S)", text, flags=re.MULTILINE)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _force_split(text: str, chunk_size: int) -> List[str]:
        """超过最大块大小时的强制分割（按段落边界）"""
        chunks: List[str] = []
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if chunks and len(chunks[-1]) + len(paragraph) < chunk_size:
                chunks[-1] += "\n\n" + paragraph
            else:
                # 如果单个段落超长则按句子分割
                if len(paragraph) > chunk_size:
                    sentences = re.split(r"(?<=[。！？\.!?])\s*", paragraph)
                    for sent in sentences:
                        if chunks and len(chunks[-1]) + len(sent) < chunk_size:
                            chunks[-1] += sent
                        else:
                            chunks.append(sent)
                else:
                    chunks.append(paragraph)
        return chunks

    # ------------------------------------------------------------------
    # 3. Token 分块
    # ------------------------------------------------------------------

    @staticmethod
    def token_split(
        docs: Sequence[Dict[str, Any]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> List[Dict[str, Any]]:
        """Token 分块 — 按 LLM token 计数分块

        使用 ``cl100k_base`` 编码（GPT-4 / GPT-3.5-turbo / Claude 系列兼容），
        确保每个 chunk 不超过 LLM 上下文窗口限制。

        Args:
            docs:          文档字典列表
            chunk_size:    每块最大 token 数
            chunk_overlap: 块间重叠 token 数
            encoding_name: tiktoken 编码名称

        Returns:
            分块列表
        """
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name,
            add_start_index=True,
        )

        return TextSplitter._apply_splitter(docs, splitter, "token")

    # ------------------------------------------------------------------
    # 4. Markdown 标题分块（结构化保留）
    # ------------------------------------------------------------------

    @staticmethod
    def markdown_header_split(
        docs: Sequence[Dict[str, Any]],
        headers_to_split_on: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """Markdown 标题结构分块 — 保留完整章节树

        按 ``#`` 标题层级分割，每个 chunk 保留所属标题链。
        适合生成带有完整上下文路径的检索块。

        Args:
            docs:               文档字典列表
            headers_to_split_on: 分割标题层级，如 ``[("#", "h1"), ("##", "h2")]``

        Returns:
            分块列表（metadata 中包含 header 链）
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        chunks: List[Dict[str, Any]] = []
        global_idx = 0
        for doc_idx, doc in enumerate(docs):
            content = doc.get("content", "")
            if not content:
                continue

            split_docs = splitter.split_text(content)
            for sd in split_docs:
                chunks.append(
                    {
                        "content": sd.page_content,
                        "doc_index": doc_idx,
                        "chunk_index": global_idx,
                        "chunk_size": len(sd.page_content),
                        "metadata": {
                            **doc.get("metadata", {}),
                            **sd.metadata,
                        },
                    }
                )
                global_idx += 1

        return chunks

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_splitter(
        docs: Sequence[Dict[str, Any]],
        splitter: RecursiveCharacterTextSplitter | TokenTextSplitter,
        size_type: str,
    ) -> List[Dict[str, Any]]:
        """通用的分割器应用方法"""
        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for doc_idx, doc in enumerate(docs):
            content = doc.get("content", "")
            if not content:
                continue

            texts = splitter.split_text(content)
            for chunk_idx_in_doc, text in enumerate(texts):
                chunks.append(
                    {
                        "content": text,
                        "doc_index": doc_idx,
                        "chunk_index_in_doc": chunk_idx_in_doc,
                        "chunk_index": global_idx,
                        "chunk_size": (
                            len(text) if size_type == "char" else len(text.split())
                        ),
                        "size_type": size_type,
                        "metadata": doc.get("metadata", {}).copy(),
                    }
                )
                global_idx += 1

        return chunks

    @staticmethod
    def _make_chunk(
        text: str,
        doc_index: int,
        chunk_index: int,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """构造统一的 chunk 字典"""
        return {
            "content": text,
            "doc_index": doc_index,
            "chunk_index": chunk_index,
            "chunk_size": len(text),
            "size_type": "char",
            "metadata": metadata.copy(),
        }

    # ------------------------------------------------------------------
    # 5. 统计信息
    # ------------------------------------------------------------------

    @staticmethod
    def stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """返回分块后的统计信息"""
        if not chunks:
            return {"count": 0, "total_chars": 0}

        sizes = [c["chunk_size"] for c in chunks]
        return {
            "count": len(chunks),
            "total_chars": sum(sizes),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "size_type": chunks[0].get("size_type", "unknown"),
        }


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def split_documents(
    docs: Sequence[Dict[str, Any]],
    strategy: str = "recursive",
    **kwargs,
) -> List[Dict[str, Any]]:
    """快捷函数 — 按策略名称快速分块

    Args:
        docs:     文档字典列表
        strategy: 分块策略名（recursive / semantic / token / markdown_header）
        **kwargs: 传递给具体策略的参数

    Returns:
        分块列表
    """
    strategy_map = {
        "recursive": TextSplitter.recursive_split,
        "semantic": TextSplitter.semantic_split,
        "token": TextSplitter.token_split,
        "markdown_header": TextSplitter.markdown_header_split,
    }

    splitter_fn = strategy_map.get(strategy)
    if splitter_fn is None:
        raise ValueError(f"未知的分块策略: {strategy}，可选: {list(strategy_map.keys())}")

    return splitter_fn(docs, **kwargs)

if __name__ == "__main__":
    from src.document_loader import DocumentLoader
    loader = DocumentLoader()
    result = loader.load("测试文档.md")

    chunks = split_documents([result], "semantic")
    print(chunks)