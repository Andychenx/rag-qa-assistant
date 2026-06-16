"""
智能 RAG 问答助手 — 核心模块

基于检索增强生成 (RAG) 的智能文档问答系统。

模块组成:
  - document_loader: 多格式文档加载器
  - splitter:        文档分块策略
  - embeddings:      嵌入向量生成（多后端支持）
  - vector_store:    向量存储与检索
  - qa_chain:        RAG 问答链
  - query_transform: 查询转换优化
  - reranker:        结果重排序
  - config:          系统配置管理
"""

from __future__ import annotations

from src.config import settings
from src.document_loader import DocumentLoader, load_document
from src.splitter import TextSplitter, split_documents
from src.embeddings import (
    EmbeddingFactory,
    create_embeddings,
    cosine_similarity,
)
from src.vector_store import VectorStore, create_vector_store
from src.qa_chain import RAGChain, ConversationMemory, create_rag_chain
from src.query_transform import QueryTransformer
from src.reranker import Reranker

__version__ = "1.0.0"
__all__ = [
    # 配置
    "settings",
    # 文档加载
    "DocumentLoader",
    "load_document",
    # 分块
    "TextSplitter",
    "split_documents",
    # 嵌入
    "EmbeddingFactory",
    "create_embeddings",
    "cosine_similarity",
    # 向量存储
    "VectorStore",
    "create_vector_store",
    # 问答
    "RAGChain",
    "ConversationMemory",
    "create_rag_chain",
    # 查询优化
    "QueryTransformer",
    # 重排序
    "Reranker",
]
