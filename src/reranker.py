"""
重排序模块 — 对检索结果进行二次排序，提升最终结果质量

技术栈:
  - 关键词加权 (keyword_boost): 根据查询词在文档中的出现情况加权
  - Cross-Encoder (cross_encoder): 基于深度语义匹配的重排序
  - Cohere Rerank (cohere_rerank): Cohere 提供的重排序 API

设计理念:
    检索 (Retrieval) 阶段优先保证高召回，用 Embedding 快速筛选候选集。
    重排序 (Rerank) 阶段对候选集进行精确排序，用更强大的模型精排 Top-K。
    这种 "Retrieve-then-Rerank" 范式可以有效兼顾效率与精度。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """重排序器 — 多种重排序策略

    用法:
        reranker = Reranker()

        # 关键词加权
        reranked = reranker.keyword_boost(query, docs)

        # Cross-Encoder
        reranked = reranker.cross_encoder(query, docs)

        # Cohere API
        reranked = reranker.cohere_rerank(query, docs)
    """

    def __init__(self, method: Optional[str] = None):
        """
        Args:
            method: 默认重排序方法（keyword_boost / cross_encoder / cohere_rerank）
        """
        self.method = method or settings.rerank.method

    def rerank(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        method: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """统一重排序入口

        Args:
            query:  查询文本
            docs:   待重排序的文档列表
            method: 重排序方法名
            top_k:  返回结果数

        Returns:
            重排序后的文档列表
        """
        method = method or self.method
        top_k = top_k or settings.rerank.top_k

        if not docs:
            return []

        method_map = {
            "keyword_boost": self.keyword_boost,
            "cross_encoder": self.cross_encoder,
            "cohere_rerank": self.cohere_rerank,
        }

        rerank_fn = method_map.get(method)
        if rerank_fn is None:
            logger.warning("未知的重排序方法: %s，使用 keyword_boost", method)
            rerank_fn = self.keyword_boost

        reranked = rerank_fn(query, docs)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # 1. 关键词加权 (keyword_boost)
    # ------------------------------------------------------------------

    @staticmethod
    def keyword_boost(
        query: str,
        docs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """关键词加权重排序

        原理:
            1. 解析查询中的关键词（去除停用词）
            2. 统计每个文档中关键词的出现频率和位置
            3. 综合原始分数 + 关键词密度，重新排序

        Args:
            query: 查询文本
            docs:  待排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if not docs:
            return []

        # 提取关键词（简单的词频过滤）
        keywords = Reranker._extract_keywords(query)

        if not keywords:
            # 没有有效关键词，按原始分数排序
            return sorted(docs, key=lambda d: d.get("score", 0), reverse=True)

        # 计算每个文档的关键词加权分数
        scored_docs = []
        for doc in docs:
            content = doc.get("content", "")
            if not content:
                scored_docs.append((doc, 0))
                continue

            content_lower = content.lower()
            keyword_score = 0.0

            for kw, weight in keywords:
                # 关键词出现次数
                count = content_lower.count(kw.lower())
                # 位置加分（关键词出现在开头说明更相关）
                pos_bonus = 0
                if count > 0:
                    first_pos = content_lower.find(kw.lower())
                    pos_bonus = max(0, 1.0 - first_pos / max(len(content), 1))
                keyword_score += (count * 0.3 + pos_bonus * 0.7) * weight

            # 综合原始分数（0-1 归一化后）和关键词分数
            original_score = doc.get("score", 0.5)
            # 将原始分数也归一化到 0-1
            combined = original_score * 0.4 + keyword_score * 0.6
            scored_docs.append((doc, combined))

        # 按综合得分降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs]

    @staticmethod
    def _extract_keywords(query: str) -> List[tuple]:
        """提取查询中的关键词及权重

        Returns:
            [(keyword, weight), ...]
        """
        # 中文停用词列表
        stop_words = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
            "们", "那", "些", "什么", "怎么", "哪", "为什么", "如何", "是否",
            "能", "能够", "可以", "应该", "需要", "让", "把", "被", "将",
            "与", "及", "或", "但", "而", "因为", "所以", "如果", "虽然",
            "吗", "啊", "呢", "吧", "呀", "哦", "嗯", "嘛", "哈",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "this", "that", "these", "those", "it", "its", "what", "which",
            "who", "whom", "how", "when", "where",
        }

        # 分词（简单的按空格/标点分割 + jieba）
        try:
            import jieba
            words = list(jieba.cut(query))
        except ImportError:
            # jieba 不可用，按字符分割
            words = re.findall(r"[\w]+", query)

        # 过滤停用词
        keywords = [w.strip().lower() for w in words if w.strip().lower() not in stop_words and len(w.strip()) > 1]

        if not keywords:
            return []

        # 计算词频权重
        from collections import Counter
        freq = Counter(keywords)
        max_freq = max(freq.values())

        return [(word, count / max_freq) for word, count in freq.items()]

    # ------------------------------------------------------------------
    # 2. Cross-Encoder 重排序
    # ------------------------------------------------------------------

    def cross_encoder(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cross-Encoder 重排序

        原理:
            使用 Cross-Encoder 模型（如 BAAI/bge-reranker-v2-m3），
            将 query 和每个文档拼接后输入模型，直接输出相关性分数。
            相比传统向量检索，Cross-Encoder 精度更高但速度较慢。

        Args:
            query: 查询文本
            docs:  待排序的文档列表

        Returns:
            重排序后的文档列表
        """
        if not docs:
            return []

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "sentence-transformers 未安装，回退到 keyword_boost。"
                "请执行: pip install sentence-transformers"
            )
            return self.keyword_boost(query, docs)

        try:
            model = CrossEncoder(
                "BAAI/bge-reranker-v2-m3",
                max_length=512,
                device="cpu",
            )

            # 构建 query-doc 对
            pairs = [[query, doc.get("content", "")] for doc in docs]

            # 计算相关性分数
            scores = model.predict(pairs)

            # 与文档配对并排序
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for doc, score in scored_docs:
                doc = dict(doc)
                doc["rerank_score"] = float(score)
                reranked.append(doc)

            return reranked

        except Exception as e:
            logger.error("Cross-Encoder 重排序失败: %s", e)
            return self.keyword_boost(query, docs)

    # ------------------------------------------------------------------
    # 3. Cohere Rerank API
    # ------------------------------------------------------------------

    def cohere_rerank(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cohere Rerank API — 基于 Cohere 的云端重排序服务

        Args:
            query: 查询文本
            docs:  待排序的文档列表

        Returns:
            重排序后的文档列表
        """
        try:
            import cohere
        except ImportError:
            logger.warning(
                "cohere 包未安装，回退到 keyword_boost。"
                "请执行: pip install cohere"
            )
            return self.keyword_boost(query, docs)

        api_key = settings.llm.api_key or None
        if not api_key:
            logger.warning("Cohere API Key 未配置，回退到 keyword_boost")
            return self.keyword_boost(query, docs)

        try:
            client = cohere.Client(api_key)
            documents = [doc.get("content", "") for doc in docs]

            response = client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=documents,
                top_n=len(documents),
            )

            reranked = []
            for result in response.results:
                doc = dict(docs[result.index])
                doc["rerank_score"] = result.relevance_score
                reranked.append(doc)

            return reranked

        except Exception as e:
            logger.error("Cohere 重排序失败: %s", e)
            return self.keyword_boost(query, docs)


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def rerank_documents(
    query: str,
    docs: Sequence[Dict[str, Any]],
    method: str = "keyword_boost",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """快捷函数 — 重排序文档"""
    reranker = Reranker(method=method)
    return reranker.rerank(query, docs, top_k=top_k)


if __name__ == "__main__":
    # 测试关键词加权重排序
    query = "Python 和 Java 的性能对比"
    docs = [
        {"content": "Python 是一种解释型语言，开发效率高但运行速度较慢", "score": 0.8},
        {"content": "Java 是一种编译型语言，运行速度快但开发效率相对较低", "score": 0.7},
        {"content": "今天天气很好，适合出去散步", "score": 0.6},
    ]

    reranker = Reranker()
    results = reranker.keyword_boost(query, docs)

    print("=" * 60)
    print(f"关键词: {Reranker._extract_keywords(query)}")
    print("=" * 60)
    for r in results:
        print(f"  content: {r['content'][:60]}...")
        print(f"  score: {r.get('score', 0):.4f}")
        print()
