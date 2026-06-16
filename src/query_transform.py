"""
查询转换模块 — 提升 RAG 检索质量的多项查询优化技术

核心方法:
  1. 查询重写 (rewrite_query):     将多轮对话中的模糊问题改写为独立查询
  2. HyDE (hyde_query):           先假设答案再检索，提高检索命中率
  3. 子查询分解 (generate_sub_queries): 将复杂问题拆解为多个子问题分别检索
  4. 查询扩展 (expand_query):      通过同义词/相关词扩展查询
  5. 退后提示 (stepback_prompt):   先生成更通用的问题，再检索
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueryTransformer
# ---------------------------------------------------------------------------

class QueryTransformer:
    """查询转换器 — 多种查询优化策略

    用法:
        transformer = QueryTransformer()

        # 查询重写（多轮对话）
        rewritten = transformer.rewrite_query("它的作者是谁？", history=["..."])

        # HyDE
        results = transformer.hyde_query("RAG 的原理是什么", retriever)

        # 子查询
        sub_queries = transformer.generate_sub_queries("对比 Python 和 Java 的优缺点")
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LLM 实例（需支持 chat 方法），不提供则按配置自动创建
        """
        self._llm = llm

    @property
    def llm(self):
        """获取 LLM 实例（惰性初始化）"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    @staticmethod
    def _create_llm():
        """创建 OpenAI 兼容 API 的 LLM 实例（支持 DeepSeek / OpenAI 等）"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=settings.llm.api_key or None,
                base_url=settings.llm.base_url or None,
            )
            return client
        except ImportError:
            logger.warning("OpenAI 兼容 API 库未安装，查询转换功能受限")
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """调用 LLM 并返回文本结果"""
        if self.llm is None:
            return None

        try:
            resp = self.llm.chat.completions.create(
                model=settings.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 1. 查询重写
    # ------------------------------------------------------------------

    def rewrite_query(
        self,
        original: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """查询重写 — 将对话中的模糊问题改写为独立查询

        在 RAG 的多轮对话中，用户后续问题往往省略上下文
        （如 "它的原理是什么？" 中的 "它" 指代不明），
        重写后变成包含上下文的独立查询。

        Args:
            original: 原始用户问题
            history:  对话历史 [{"role": "user"/"assistant", "content": str}, ...]

        Returns:
            重写后的独立查询
        """
        if not history or len(history) < 2:
            # 没有历史或历史太少 → 直接返回原始查询
            return original

        # 提取最近几轮对话
        recent_history = history[-6:]  # 最近 3 轮（user + assistant）
        history_text = "\n".join(
            f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
            for h in recent_history
        )

        system_prompt = (
            "你是一个查询重写助手。用户正在进行多轮对话，"
            "他/她最新的问题可能依赖于之前的对话上下文。\n\n"
            "请将用户的最新问题改写为一个**独立、完整的查询**，"
            "使其不依赖对话历史也能被理解。\n\n"
            "要求:\n"
            "1. 用代词（它、他、她、它们）替换为具体指代的对象\n"
            "2. 补充省略的上下文信息\n"
            "3. 保持原始问题的核心意图不变\n"
            "4. **只输出重写后的查询，不要任何解释或前缀**"
        )

        result = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"对话历史:\n{history_text}\n\n最新问题: {original}",
        )

        if result:
            # 清理引号和多余空白
            result = result.strip("\"'「」『』")
            logger.info("查询重写: '%s' → '%s'", original, result)
            return result

        return original

    # ------------------------------------------------------------------
    # 2. HyDE (假设性文档嵌入)
    # ------------------------------------------------------------------

    def hyde_query(
        self,
        original: str,
        retriever: Any,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """HyDE (Hypothetical Document Embedding) — 先假设答案再检索

        原理:
            1. 让 LLM 根据问题先生成一个"假设答案"
            2. 用这个假设答案作为查询去检索
            3. 因为假设答案更接近真实文档的语义，检索效果更好

        Args:
            original:  原始用户问题
            retriever: 检索器（需有 similarity_search 方法）
            k:         返回结果数

        Returns:
            检索到的文档列表
        """
        system_prompt = (
            "你是一个文档撰写助手。请根据用户的问题，"
            "撰写一段**假设性的答案文本**。\n\n"
            "要求:\n"
            "1. 文本应当看起来像是一篇真实的文档段落\n"
            "2. 尽可能详细、具体，包含可能的关键术语\n"
            "3. 不需要标注「这是我假设的」，直接写正文\n"
            "4. 长度约 100-200 字\n"
            "5. **只输出假设答案本身，不要任何解释**"
        )

        hypothetical_answer = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"问题: {original}",
        )

        if not hypothetical_answer:
            logger.warning("HyDE 生成失败，回退到原始查询")
            return retriever.similarity_search(original, k=k)

        logger.info("HyDE 假设答案: %s...", hypothetical_answer[:100])

        # 用假设答案检索
        return retriever.similarity_search(hypothetical_answer, k=k)

    # ------------------------------------------------------------------
    # 3. 子查询分解
    # ------------------------------------------------------------------

    def generate_sub_queries(self, query: str) -> List[str]:
        """复杂查询拆解为多个子查询

        当用户问题涉及多个方面时，将问题拆解为多个独立子查询，
        分别检索后合并结果，可以更全面地覆盖用户需求。

        Args:
            query: 用户的复杂查询

        Returns:
            子查询列表
        """
        system_prompt = (
            "你是一个问题分解助手。请将用户的复杂问题拆解为 "
            "2-4 个独立的、更具体的子问题。\n\n"
            "要求:\n"
            "1. 每个子问题应覆盖原问题的不同方面\n"
            "2. 子问题应具有独立检索价值\n"
            "3. 子问题之间尽量减少重叠\n"
            "4. **只输出子问题列表，每行一个，不要序号和前缀**"
        )

        result = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"需要分解的问题: {query}",
        )

        if result:
            sub_queries = [
                line.strip().strip("-\"\"'[]()0123456789.、。 ")
                for line in result.split("\n")
                if line.strip()
            ]
            # 过滤掉可能的非查询行
            sub_queries = [
                q for q in sub_queries
                if len(q) > 5 and not q.startswith(("以下", "这是", "子问题"))
            ]
            if sub_queries:
                logger.info("子查询分解: %s → %s", query, sub_queries)
                return sub_queries

        return [query]

    # ------------------------------------------------------------------
    # 4. 查询扩展
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> List[str]:
        """查询扩展 — 通过同义词和相关术语增强查询

        生成原查询 + 多个变体，分别检索后合并结果，
        提高检索的召回率。

        Args:
            query: 原始查询

        Returns:
            扩展后的多个查询变体（含原查询）
        """
        system_prompt = (
            "你是一个查询扩展助手。请为用户的问题生成 3 个"
            "同义或近义的查询变体。\n\n"
            "要求:\n"
            "1. 保持核心意图不变\n"
            "2. 替换部分关键词为同义词或相关术语\n"
            "3. 可以改变句式但不要改变含义\n"
            "4. **只输出查询变体，每行一个，不要序号和前缀**"
        )

        result = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"原始查询: {query}",
        )

        expanded = [query]  # 原查询始终包含
        if result:
            variants = [
                line.strip().strip("-\"\"'[]()0123456789.、。 ")
                for line in result.split("\n")
                if line.strip() and len(line.strip()) > 3
            ]
            expanded.extend(variants)

        logger.info("查询扩展: %s → %s", query, expanded)
        return expanded

    # ------------------------------------------------------------------
    # 5. 退后提示 (Step-Back Prompting)
    # ------------------------------------------------------------------

    def stepback_query(self, query: str) -> str:
        """退后提示 — 先生成更通用的（退后）问题，再检索

        对于过于具体的查询，先"退后一步"生成一个更通用的问题，
        有助于检索到更相关的背景信息。

        例如:
          - 具体: "LangChain 的 RecursiveCharacterTextSplitter 的 chunk_size 设多少"
          - 退后: "LangChain 文本分块策略的工作原理"
        """
        system_prompt = (
            "你是一个问题抽象化助手。请将用户非常具体的问题"
            "「退后一步」，生成一个更通用、更抽象的问题版本。\n\n"
            "目标: 生成一个能帮助检索到相关背景信息的问题。\n\n"
            "要求:\n"
            "1. 保持核心主题不变\n"
            "2. 去掉具体参数/数字/细节\n"
            "3. 关注「是什么」「为什么」「原理」层面\n"
            "4. **只输出退后问题，不要任何解释**"
        )

        result = self._call_llm(
            system_prompt=system_prompt,
            user_prompt=f"用户问题: {query}",
        )

        if result:
            logger.info("退后提示: '%s' → '%s'", query, result)
            return result

        return query


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def rewrite_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """快捷函数 — 查询重写"""
    return QueryTransformer().rewrite_query(query, history=history)


def generate_sub_queries(query: str) -> List[str]:
    """快捷函数 — 子查询分解"""
    return QueryTransformer().generate_sub_queries(query)


if __name__ == "__main__":
    transformer = QueryTransformer()

    # 测试查询重写
    history = [
        {"role": "user", "content": "什么是 RAG？"},
        {"role": "assistant", "content": "RAG 是检索增强生成的缩写..."},
        {"role": "user", "content": "它有哪些应用场景？"},
    ]
    rewritten = transformer.rewrite_query("它有哪些应用场景？", history=history)
    print(f"重写结果: {rewritten}")

    # 测试子查询
    sub = transformer.generate_sub_queries("对比 Python 和 Java 的优劣")
    print(f"子查询: {sub}")

    # 测试查询扩展
    expanded = transformer.expand_query("RAG 的原理")
    print(f"扩展查询: {expanded}")

    # 测试退后提示
    stepback = transformer.stepback_query(
        "LangChain 的 RecursiveCharacterTextSplitter 的 chunk_size 设多少"
    )
    print(f"退后提示: {stepback}")
