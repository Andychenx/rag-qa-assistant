"""
向量存储模块 — 基于 ChromaDB 的向量检索系统

技术栈:
  - ChromaDB:        本地持久化向量数据库（默认）
  - FAISS:           Facebook 开源的高性能向量检索库（备选）
  - LangChain:       统一的 Retriever 接口封装

功能:
  - 向量存储:        创建 Embedding → 存入 ChromaDB → 持久化
  - 向量加载:        从本地加载已有向量库
  - 相似度搜索:      基础向量相似度检索
  - 混合搜索:        向量 + BM25 关键词混合检索
  - MMR 搜索:        最大边际相关性检索（多样性提升）
  - 带分数返回:      检索结果附带相似度分数
  - 集合管理:        多集合创建、删除、列表
  - 增量更新:        支持新增、删除文档
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class VectorStoreError(RuntimeError):
    """向量存储操作失败"""


class CollectionNotFoundError(VectorStoreError):
    """集合不存在"""


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """向量存储 — 统一 ChromaDB 接口

    用法:
        store = VectorStore()
        store.store_vectors(chunks)
        results = store.similarity_search("你的问题", k=5)
        results = store.hybrid_search("你的问题", k=5)
        results = store.mmr_search("你的问题", k=5)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function=None,
    ):
        """
        Args:
            persist_dir:        持久化目录路径
            collection_name:    集合名称
            embedding_function: 可选的嵌入函数，若不提供则从 embeddings 模块自动创建
        """
        self.persist_dir = persist_dir or settings.vector_store.persist_dir
        self.collection_name = collection_name or settings.vector_store.collection_name

        # 确保持久化目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # 嵌入函数
        self._embedding_function = embedding_function
        self._client = None
        self._collection = None

        logger.info(
            "VectorStore 初始化: persist_dir=%s, collection=%s",
            self.persist_dir, self.collection_name,
        )

    # ------------------------------------------------------------------
    # 客户端与集合管理
    # ------------------------------------------------------------------

    @property
    def client(self):
        """获取 ChromaDB 客户端（惰性初始化）"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def collection(self):
        """获取集合（惰性初始化，不存在则创建）"""
        if self._collection is None:
            self._collection = self._get_or_create_collection()
        return self._collection

    def _create_client(self):
        """创建 ChromaDB 客户端"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("chromadb 未安装: pip install chromadb")

        return chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
            ),
        )

    def _get_or_create_collection(self):
        """获取已有集合或创建新集合"""
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._get_chroma_embedding(),
            )
        except Exception as e:
            raise VectorStoreError(f"创建/获取集合失败: {e}") from e

    def _get_chroma_embedding(self):
        """获取 ChromaDB 兼容的嵌入函数"""
        if self._embedding_function is not None:
            return self._embedding_function

        # 从 embeddings 模块获取
        try:
            from chromadb.utils import embedding_functions
        except ImportError:
            return None

        provider = settings.embedding.provider

        # 优先使用 OpenAI 兼容嵌入 API（如果配置了）
        if provider == "openai" and settings.embedding.api_key:
            try:
                from chromadb.utils import embedding_functions
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=settings.embedding.api_key,
                    model_name=settings.embedding.model,
                )
            except Exception:
                pass
        elif provider == "ollama":
            # ChromaDB 不直接支持 Ollama，使用自定义函数
            return self._OllamaChromaEmbeddingFunction(
                model=settings.embedding.model,
                base_url=settings.embedding.ollama_base_url,
            )

        # 对 local / auto 模式，返回 None，让 ChromaDB 使用默认的 all-MiniLM-L6-v2
        return None

    # ------------------------------------------------------------------
    # 自定义 Ollama 嵌入适配器（用于 ChromaDB）
    # ------------------------------------------------------------------

    class _OllamaChromaEmbeddingFunction:
        """将 Ollama 嵌入适配到 ChromaDB 接口"""

        def __init__(self, model: str, base_url: str):
            self._model = model
            self._base_url = base_url.rstrip("/")
            try:
                import requests as _req
                self._requests = _req
            except ImportError:
                raise ImportError("requests 未安装: pip install requests")

        def __call__(self, input: List[str]) -> List[List[float]]:
            try:
                resp = self._requests.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": input},
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json().get("embeddings", [])
            except Exception as e:
                logger.error("Ollama 嵌入调用失败: %s", e)
                # 返回零向量作为 fallback
                return [[0.0] * 768 for _ in input]

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def list_collections(self) -> List[str]:
        """列出所有集合名称"""
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: Optional[str] = None) -> None:
        """删除集合"""
        name = name or self.collection_name
        try:
            self.client.delete_collection(name)
            if name == self.collection_name:
                self._collection = None
            logger.info("集合已删除: %s", name)
        except Exception as e:
            raise VectorStoreError(f"删除集合失败: {e}") from e

    def reset(self) -> None:
        """重置整个向量数据库（清空所有集合）"""
        try:
            self.client.reset()
            self._collection = None
            logger.info("向量数据库已重置")
        except Exception as e:
            raise VectorStoreError(f"重置失败: {e}") from e

    # ------------------------------------------------------------------
    # 存储向量
    # ------------------------------------------------------------------

    def store_vectors(
        self,
        chunks: Sequence[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """将分块后的文档存入向量数据库

        Args:
            chunks:    分块列表（来自 splitter 模块）
            batch_size: 每批存储的文档数

        Returns:
            存储的文档数
        """
        if not chunks:
            logger.warning("没有分块需要存储")
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            count = self._store_batch(batch)
            total += count
            logger.info(
                "向量存储进度: %d / %d (%.0f%%)",
                total, len(chunks), total / len(chunks) * 100,
            )

        logger.info("向量存储完成: 共 %d 个分块", total)
        return total

    def _store_batch(self, batch: List[Dict[str, Any]]) -> int:
        """存储一批分块到向量数据库"""
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in batch:
            content = chunk.get("content", "")
            if not content.strip():
                continue

            # 生成唯一 ID
            doc_idx = chunk.get("doc_index", 0)
            chunk_idx = chunk.get("chunk_index", 0)
            chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"

            ids.append(chunk_id)
            documents.append(content)
            metadatas.append({
                "doc_index": str(chunk.get("doc_index", 0)),
                "chunk_index": str(chunk.get("chunk_index", 0)),
                "chunk_size": str(chunk.get("chunk_size", 0)),
                "source": chunk.get("metadata", {}).get("format", "unknown"),
                **{
                    k: str(v) for k, v in chunk.get("metadata", {}).items()
                    if isinstance(v, (str, int, float, bool))
                },
            })

        if not ids:
            return 0

        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"批量存储失败: {e}") from e

    # ------------------------------------------------------------------
    # 检索方法
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """基础相似度搜索

        Args:
            query:  查询文本
            k:      返回结果数
            filter: 元数据过滤条件

        Returns:
            [{"content": str, "metadata": dict, "score": float, ...}]
        """
        k = k or settings.retrieval.top_k
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
            )
            return self._format_results(results)
        except Exception as e:
            logger.error("相似度搜索失败: %s", e)
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """带分数的相似度搜索

        Returns:
            结果按分数降序排列（分数为余弦距离，越小越相似）
        """
        return self.similarity_search(query, k=k)

    def search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """带分数返回（同 similarity_search_with_score 别名）"""
        return self.similarity_search_with_score(query, k=k)

    def hybrid_search(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """混合搜索 — 向量相似度 + BM25 关键词搜索

        Args:
            query: 查询文本
            k:     返回结果数
            alpha: 向量分数权重（0 纯关键词, 1 纯向量）

        Returns:
            按混合分数排序的结果列表
        """
        k = k or settings.retrieval.top_k
        alpha = alpha if alpha is not None else settings.retrieval.hybrid_alpha

        # 1. 向量搜索
        vector_results = self.similarity_search(query, k=k * 2)

        # 2. BM25 关键词搜索
        keyword_results = self._bm25_search(query, k=k * 2)

        # 3. 融合分数
        return self._fuse_results(
            vector_results, keyword_results,
            alpha=alpha,
            top_k=k,
        )

    def mmr_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """MMR 最大边际相关性检索

        在相关性和多样性之间平衡，避免返回过于相似的重复结果。

        Args:
            query:       查询文本
            k:           最终返回结果数
            fetch_k:     初始候选集大小
            lambda_mult: 多样性参数（0 纯多样性, 1 纯相关性）

        Returns:
            兼顾相关性与多样性的结果列表
        """
        k = k or settings.retrieval.top_k
        fetch_k = fetch_k or settings.retrieval.mmr_fetch_k
        lambda_mult = lambda_mult if lambda_mult is not None else settings.retrieval.mmr_lambda

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
            )
            candidates = self._format_results(results)

            if not candidates:
                return []

            # MMR 算法
            selected = self._mmr_select(
                query_embedding=None,  # 无法直接获取 query embedding
                candidates=candidates,
                k=k,
                lambda_mult=lambda_mult,
            )
            return selected
        except Exception as e:
            logger.error("MMR 搜索失败: %s", e)
            return self.similarity_search(query, k=k)

    # ------------------------------------------------------------------
    # BM25 关键词搜索
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """简单的 BM25 关键词搜索

        基于词频的本地实现，不依赖外部搜索引擎。
        """
        # 获取所有文档（分页获取）
        all_docs = self._get_all_documents()
        if not all_docs:
            return []

        # 分词
        import jieba
        query_tokens = set(jieba.cut_for_search(query))
        if not query_tokens:
            return []

        # 计算每个文档的 BM25 分数
        doc_scores: List[Tuple[int, float]] = []
        avg_doc_len = sum(len(d.get("content", "")) for d in all_docs) / max(len(all_docs), 1)

        for idx, doc in enumerate(all_docs):
            content = doc.get("content", "")
            doc_tokens = list(jieba.cut_for_search(content))
            doc_len = len(doc_tokens)

            score = 0.0
            for qt in query_tokens:
                if qt in doc_tokens:
                    tf = doc_tokens.count(qt) / max(doc_len, 1)
                    idf = 1.0  # 简化 IDF
                    score += tf * idf

            if score > 0:
                doc_scores.append((idx, score))

        # 排序取 top-k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in doc_scores[:k]]

        return [all_docs[idx] for idx in top_indices]

    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """获取集合中所有文档"""
        try:
            count = self.collection.count()
            if count == 0:
                return []
            results = self.collection.get(limit=count)
            return self._format_get_results(results)
        except Exception as e:
            logger.error("获取所有文档失败: %s", e)
            return []

    # ------------------------------------------------------------------
    # 结果融合（混合搜索）
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_results(
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float = 0.7,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """融合向量搜索和关键词搜索结果"""
        # 构建内容 → 文档的映射
        seen: Dict[str, Dict[str, Any]] = {}

        for i, doc in enumerate(vector_results):
            score = (1.0 - i / max(len(vector_results), 1)) * alpha
            content = doc.get("content", "")
            seen[content] = {**doc, "score": score}

        for i, doc in enumerate(keyword_results):
            score = (1.0 - i / max(len(keyword_results), 1)) * (1 - alpha)
            content = doc.get("content", "")
            if content in seen:
                seen[content]["score"] += score
            else:
                seen[content] = {**doc, "score": score}

        # 按分数排序
        fused = sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # MMR 选择
    # ------------------------------------------------------------------

    @staticmethod
    def _mmr_select(
        query_embedding: Optional[List[float]],
        candidates: List[Dict[str, Any]],
        k: int,
        lambda_mult: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """MMR 算法选择子集

        贪心选择：每步选一个与查询相关且与已选集合不冗余的文档。
        """
        if not candidates:
            return []

        # 使用内容 Jaccard 相似度作为多样性度量
        def _jaccard_sim(a: str, b: str) -> float:
            set_a = set(a)
            set_b = set(b)
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        selected: List[Dict[str, Any]] = []
        remaining = list(candidates)
        # 默认分数作为相关性
        for doc in remaining:
            if "score" not in doc:
                doc["score"] = 1.0

        while len(selected) < k and remaining:
            mmr_scores = []
            for doc in remaining:
                rel = doc.get("score", 0.5)
                # 与已选集合的最大相似度
                if selected:
                    max_div = max(
                        _jaccard_sim(doc.get("content", ""), s.get("content", ""))
                        for s in selected
                    )
                else:
                    max_div = 0
                mmr = lambda_mult * rel - (1 - lambda_mult) * max_div
                mmr_scores.append(mmr)

            best_idx = max(range(len(remaining)), key=lambda i: mmr_scores[i])
            selected.append(remaining.pop(best_idx))

        return selected

    # ------------------------------------------------------------------
    # 结果格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: Any) -> List[Dict[str, Any]]:
        """将 ChromaDB 查询结果格式化为统一字典列表"""
        if not results or not results.get("ids"):
            return []

        formatted: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            doc = {
                "content": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "id": results["ids"][0][i],
                "score": results["distances"][0][i] if results.get("distances") else 0.0,
            }
            formatted.append(doc)

        return formatted

    @staticmethod
    def _format_get_results(results: Any) -> List[Dict[str, Any]]:
        """将 ChromaDB get 结果格式化为统一字典列表"""
        if not results or not results.get("ids"):
            return []

        formatted: List[Dict[str, Any]] = []
        for i in range(len(results["ids"])):
            doc = {
                "content": results["documents"][i] if results.get("documents") else "",
                "metadata": results["metadatas"][i] if results.get("metadatas") else {},
                "id": results["ids"][i],
                "score": 0.0,
            }
            formatted.append(doc)

        return formatted

    # ------------------------------------------------------------------
    # 文档管理
    # ------------------------------------------------------------------

    def count(self) -> int:
        """获取集合中的文档总数"""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def delete_documents(self, ids: List[str]) -> None:
        """按 ID 删除文档"""
        try:
            self.collection.delete(ids=ids)
            logger.info("已删除 %d 个文档", len(ids))
        except Exception as e:
            raise VectorStoreError(f"删除文档失败: {e}") from e

    def clear(self) -> None:
        """清空当前集合的全部文档"""
        try:
            all_ids = self.collection.get()["ids"]
            if all_ids:
                self.collection.delete(ids=all_ids)
            logger.info("集合已清空: %s", self.collection_name)
        except Exception as e:
            raise VectorStoreError(f"清空集合失败: {e}") from e

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    @property
    def is_persisted(self) -> bool:
        """检查是否已有持久化的向量库"""
        return Path(self.persist_dir).exists() and any(
            Path(self.persist_dir).iterdir()
        )

    def get_persist_size(self) -> str:
        """获取持久化目录大小（人类可读）"""
        total = 0
        for f in Path(self.persist_dir).rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        if total < 1024:
            return f"{total} B"
        elif total < 1024 ** 2:
            return f"{total / 1024:.1f} KB"
        else:
            return f"{total / 1024 ** 2:.1f} MB"


# ---------------------------------------------------------------------------
# FAISS 备选实现
# ---------------------------------------------------------------------------

class FAISSVectorStore:
    """基于 FAISS 的向量存储（备选，用于大规模检索场景）

    注意: 不支持元数据过滤，适合纯向量检索场景。
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._index = None
        self._documents: List[Dict[str, Any]] = []

        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss 未安装: pip install faiss-cpu")

        self._index = self._faiss.IndexFlatIP(dimension)

    def add(self, embeddings: List[List[float]], documents: List[Dict[str, Any]]) -> None:
        """添加向量和文档"""
        import numpy as np

        vectors = np.array(embeddings).astype("float32")
        self._faiss.normalize_L2(vectors)
        self._index.add(vectors)
        self._documents.extend(documents)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """搜索最相似的 k 个文档"""
        import numpy as np

        vector = np.array([query_embedding]).astype("float32")
        self._faiss.normalize_L2(vector)

        scores, indices = self._index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._documents):
                doc = dict(self._documents[idx])
                doc["score"] = float(score)
                results.append(doc)
        return results

    def save(self, path: str) -> None:
        """保存索引到磁盘"""
        self._faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """从磁盘加载索引"""
        self._index = self._faiss.read_index(path)


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def create_vector_store(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """快捷函数 — 创建 VectorStore 实例"""
    return VectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    store = create_vector_store()
    print(f"持久化目录: {store.persist_dir}")
    print(f"集合名称: {store.collection_name}")

    if store.is_persisted:
        print(f"已持久化, 大小: {store.get_persist_size()}, 文档数: {store.count()}")
    else:
        print("向量库为空，请先上传文档")

    # 测试搜索
    results = store.similarity_search("测试", k=3)
    print(f"\n搜索结果: {len(results)} 条")
    for r in results:
        print(f"  - [{r['id']}] (score={r['score']:.4f}) {r['content'][:80]}...")
