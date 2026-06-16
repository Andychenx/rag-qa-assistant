"""
问答链模块 — 构建 RAG (检索增强生成) 问答流水线

核心组件:
  1. RAGChain:           主问答链，支持基础 QA、多轮对话、带重排序的增强链
  2. ConversationMemory:  对话记忆管理（缓冲 + 摘要）
  3. PromptTemplates:     统一管理的提示词模板

工作流程:
  用户提问 → 查询转换/重写 → 向量检索 → 重排序 → 构造 Prompt → LLM 生成 → 返回答案+来源
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.config import settings
from src.query_transform import QueryTransformer
from src.reranker import Reranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 对话记忆
# ---------------------------------------------------------------------------

@dataclass
class ConversationMemory:
    """对话记忆管理

    维护最近 N 轮对话，支持序列化和格式化为 Prompt。
    """

    max_rounds: int = 50
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """添加一条对话记录

        Args:
            role:   角色（user / assistant / system）
            content: 消息内容
        """
        self.messages.append({"role": role, "content": content})
        # 超出上限时裁剪（保留最旧的 2 条固定轮次）
        if len(self.messages) > self.max_rounds * 2:
            # 保留第一个 system 消息和最近的 max_rounds*2-1 条
            self.messages = [self.messages[0]] + self.messages[-(self.max_rounds * 2 - 1):]

    def get_history(self) -> List[Dict[str, str]]:
        """获取完整对话历史"""
        return self.messages.copy()

    def get_recent(self, n: int = 6) -> List[Dict[str, str]]:
        """获取最近 n 条消息"""
        return self.messages[-n:] if len(self.messages) > n else self.messages.copy()

    def get_formatted_history(self, max_rounds: Optional[int] = None) -> str:
        """格式化为文本形式的对话历史（用于 Prompt）"""
        n = max_rounds or self.max_rounds
        recent = self.messages[-(n * 2):]
        lines = []
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        """清空对话历史"""
        self.messages.clear()

    def to_dict_list(self) -> List[Dict[str, str]]:
        """转为 OpenAI 格式的消息列表（用于 API 调用）"""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages
        ]

    def __len__(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_QA = """你是一个智能文档问答助手。你的任务是基于用户提供的文档内容，准确回答用户的问题。

## 核心原则

1. **基于上下文回答**: 只能使用检索到的文档内容回答问题，不要编造信息
2. **引用来源**: 回答时标注信息来源（文档片段），让用户知道答案出处
3. **诚实面对未知**: 如果检索到的文档中没有足够信息回答，请明确告知，不要猜测
4. **中文回答**: 默认使用中文回答，除非用户的问题使用其他语言
5. **结构化表达**: 对于复杂问题，使用分点、列表等方式清晰组织答案

## 回答格式

如果找到了相关信息：
- 直接、准确地回答问题
- 引用相关的文档片段作为支撑
- 如有多个来源的观点差异，可以对比呈现

如果没有找到相关信息：
- 明确告知"在文档中未找到相关信息"
- 可以建议用户上传相关文档或换个问法

## 对话历史

以下是最近几轮的对话（仅供参考）：
{history}

## 检索到的文档内容

以下是检索到的相关文档片段（按相关性排序）：
{context}
"""

SYSTEM_PROMPT_CONV = """你是一个智能文档问答助手。你正在与用户进行多轮对话。

## 核心原则

1. **基于上下文回答**: 只能使用检索到的文档内容回答问题，不要编造信息
2. **引用来源**: 回答时标注信息来源
3. **诚实面对未知**: 如果没有足够信息，请明确告知
4. **保持对话连贯**: 记住对话上下文，对于指代词（它、它们、这个等）能正确理解
5. **中文回答**: 默认使用中文

## 对话历史

以下是之前的对话（请注意用户的当前问题可能与这些历史相关）：
{history}

## 检索到的文档内容

以下是检索到的相关文档片段（按相关性排序）：
{context}

## 当前问题

{question}
"""


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------

class RAGChain:
    """RAG 问答链

    用法:
        chain = RAGChain(retriever=vector_store)
        answer = chain.ask("RAG 的原理是什么？")
        answer, sources = chain.ask_with_sources("RAG 的原理是什么？")
    """

    def __init__(
        self,
        retriever=None,
        llm=None,
        query_transformer: Optional[QueryTransformer] = None,
        reranker: Optional[Reranker] = None,
        memory: Optional[ConversationMemory] = None,
    ):
        """
        Args:
            retriever:         检索器（需有 similarity_search 或 hybrid_search 方法）
            llm:               LLM 实例（需支持 chat.completions.create）
            query_transformer: 查询转换器
            reranker:          重排序器
            memory:            对话记忆
        """
        self.retriever = retriever
        self._llm = llm
        self.query_transformer = query_transformer or QueryTransformer(llm=llm)
        self.reranker = reranker or Reranker()
        self.memory = memory or ConversationMemory(
            max_rounds=settings.max_conversation_rounds
        )

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
            return OpenAI(
                api_key=settings.llm.api_key or None,
                base_url=settings.llm.base_url or None,
            )
        except ImportError:
            raise ImportError("OpenAI 兼容 API 库未安装: pip install openai")

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """执行检索

        Args:
            query:     查询文本
            k:         返回结果数
            use_hybrid: 是否使用混合搜索

        Returns:
            检索结果列表
        """
        if self.retriever is None:
            logger.warning("检索器未设置，无法检索")
            return []

        k = k or settings.retrieval.top_k

        try:
            if use_hybrid and hasattr(self.retriever, "hybrid_search"):
                results = self.retriever.hybrid_search(query, k=k)
            elif hasattr(self.retriever, "similarity_search"):
                results = self.retriever.similarity_search(query, k=k)
            else:
                logger.error("检索器缺少 search 方法")
                return []
        except Exception as e:
            logger.error("检索失败: %s", e)
            return []

        return results

    def _rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """对检索结果进行重排序（如启用）"""
        if not docs or not settings.rerank.enabled:
            return docs
        return self.reranker.rerank(query, docs)

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为上下文文本"""
        if not docs:
            return "（未检索到相关文档内容）"

        parts = []
        for i, doc in enumerate(docs):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            score = doc.get("score", 0)
            chunk_id = doc.get("id", f"chunk_{i}")
            parts.append(
                f"[来源 {i + 1}] (相关性: {score:.4f}) (ID: {chunk_id})\n"
                f"```\n{content}\n```"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """调用 LLM 生成回答"""
        try:
            resp = self.llm.chat.completions.create(
                model=settings.llm.model,
                messages=messages,
                temperature=temperature or settings.llm.temperature,
                max_tokens=max_tokens or settings.llm.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return f"抱歉，生成回答时出现错误: {e}"

    # ------------------------------------------------------------------
    # 核心问答方法
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_hybrid: bool = True,
    ) -> str:
        """单次问答

        Args:
            question:   用户问题
            history:    对话历史（用于查询重写）
            use_hybrid: 是否使用混合搜索

        Returns:
            生成的回答文本
        """
        start_time = time.time()

        # 1. 查询重写（如有历史）
        if history and len(history) >= 2:
            rewritten = self.query_transformer.rewrite_query(question, history=history)
        else:
            rewritten = question

        logger.info("原始问题: %s", question)
        logger.info("重写后: %s", rewritten if rewritten != question else "（无需重写）")

        # 2. 检索
        docs = self._retrieve(rewritten, use_hybrid=use_hybrid)

        # 3. 重排序
        docs = self._rerank(rewritten, docs)

        # 4. 构建上下文
        context = self._format_context(docs)

        # 5. 格式化历史
        if history:
            hist_lines = []
            for msg in history[-6:]:
                role_label = "用户" if msg["role"] == "user" else "助手"
                hist_lines.append(f"{role_label}: {msg['content']}")
            history_str = "\n".join(hist_lines)
        else:
            history_str = "（无历史对话）"

        # 6. 构建 Prompt
        system_prompt = SYSTEM_PROMPT_QA.format(history=history_str, context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # 7. 生成回答
        answer = self._call_llm(messages)
        elapsed = time.time() - start_time

        logger.info("问答完成 (%.2fs): 检索 %d 篇, 回答 %d 字", elapsed, len(docs), len(answer))
        return answer

    def ask_with_sources(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_hybrid: bool = True,
    ) -> tuple:
        """问答，返回 (回答, 来源列表)

        Returns:
            (answer: str, sources: List[Dict])
        """
        start_time = time.time()

        # 1. 查询重写
        if history and len(history) >= 2:
            rewritten = self.query_transformer.rewrite_query(question, history=history)
        else:
            rewritten = question

        # 2. 检索
        docs = self._retrieve(rewritten, use_hybrid=use_hybrid)

        # 3. 重排序
        docs = self._rerank(rewritten, docs)

        # 4. 构建上下文
        context = self._format_context(docs)

        # 5. 格式化历史
        if history:
            hist_lines = []
            for msg in history[-6:]:
                role_label = "用户" if msg["role"] == "user" else "助手"
                hist_lines.append(f"{role_label}: {msg['content']}")
            history_str = "\n".join(hist_lines)
        else:
            history_str = "（无历史对话）"

        # 6. 构建 Prompt
        system_prompt = SYSTEM_PROMPT_QA.format(history=history_str, context=context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # 7. 生成回答
        answer = self._call_llm(messages)
        elapsed = time.time() - start_time

        logger.info("问答完成 (%.2fs): 检索 %d 篇, 回答 %d 字", elapsed, len(docs), len(answer))

        # 格式化来源
        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "index": i + 1,
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "rerank_score": doc.get("rerank_score"),
                "id": doc.get("id", ""),
                "metadata": doc.get("metadata", {}),
            })

        return answer, sources

    def chat(
        self,
        question: str,
        use_hybrid: bool = True,
    ) -> str:
        """多轮对话

        自动管理对话历史，支持上下文连贯的多轮交互。

        Args:
            question:   用户问题
            use_hybrid: 是否使用混合搜索

        Returns:
            生成的回答
        """
        # 1. 添加用户消息到记忆
        self.memory.add("user", question)

        # 2. 获取对话历史
        history = self.memory.get_recent(10)  # 最近 10 条消息

        # 3. 查询重写
        if len(self.memory) >= 2:
            rewritten = self.query_transformer.rewrite_query(
                question,
                history=[{"role": m["role"], "content": m["content"]} for m in self.memory.get_recent(6)],
            )
        else:
            rewritten = question

        # 4. 检索
        docs = self._retrieve(rewritten, use_hybrid=use_hybrid)

        # 5. 重排序
        docs = self._rerank(rewritten, docs)

        # 6. 构建上下文
        context = self._format_context(docs)

        # 7. 格式化历史
        history_str = self.memory.get_formatted_history(max_rounds=5)

        # 8. 构建对话 Prompt
        system_prompt = SYSTEM_PROMPT_CONV.format(
            history=history_str,
            context=context,
            question=question,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 9. 生成回答
        answer = self._call_llm(messages)

        # 10. 将回答加入记忆
        self.memory.add("assistant", answer)

        return answer

    def chat_with_sources(
        self,
        question: str,
        use_hybrid: bool = True,
    ) -> tuple:
        """多轮对话，带来源

        Returns:
            (answer: str, sources: List[Dict])
        """
        self.memory.add("user", question)
        history = self.memory.get_recent(10)

        if len(self.memory) >= 2:
            rewritten = self.query_transformer.rewrite_query(
                question,
                history=[{"role": m["role"], "content": m["content"]} for m in self.memory.get_recent(6)],
            )
        else:
            rewritten = question

        docs = self._retrieve(rewritten, use_hybrid=use_hybrid)
        docs = self._rerank(rewritten, docs)
        context = self._format_context(docs)
        history_str = self.memory.get_formatted_history(max_rounds=5)

        system_prompt = SYSTEM_PROMPT_CONV.format(
            history=history_str,
            context=context,
            question=question,
        )

        messages = [{"role": "system", "content": system_prompt}]
        answer = self._call_llm(messages)
        self.memory.add("assistant", answer)

        sources = []
        for i, doc in enumerate(docs):
            sources.append({
                "index": i + 1,
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "rerank_score": doc.get("rerank_score"),
                "id": doc.get("id", ""),
                "metadata": doc.get("metadata", {}),
            })

        return answer, sources

    # ------------------------------------------------------------------
    # 链构建辅助方法
    # ------------------------------------------------------------------

    def build_basic_chain(self, llm=None, retriever=None) -> RAGChain:
        """构建基础 QA Chain

        返回当前实例或新实例。
        """
        if llm:
            self._llm = llm
        if retriever:
            self.retriever = retriever
        return self

    def build_conv_chain(self, llm=None, retriever=None, memory=None) -> RAGChain:
        """构建多轮对话 Chain"""
        if llm:
            self._llm = llm
        if retriever:
            self.retriever = retriever
        if memory:
            self.memory = memory
        return self

    def build_with_reranker(self, llm=None, retriever=None, reranker=None) -> RAGChain:
        """构建带重排序的 Chain"""
        if llm:
            self._llm = llm
        if retriever:
            self.retriever = retriever
        if reranker:
            self.reranker = reranker
        settings.rerank.enabled = True
        return self

    # ------------------------------------------------------------------
    # 对话管理
    # ------------------------------------------------------------------

    def clear_memory(self) -> None:
        """清空对话历史"""
        self.memory.clear()

    def get_memory(self) -> ConversationMemory:
        """获取对话记忆对象"""
        return self.memory


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------

def create_rag_chain(
    retriever=None,
    llm=None,
) -> RAGChain:
    """快捷函数 — 创建 RAGChain 实例"""
    query_transformer = QueryTransformer(llm=llm)
    reranker = Reranker()
    return RAGChain(
        retriever=retriever,
        llm=llm,
        query_transformer=query_transformer,
        reranker=reranker,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建测试用检索器
    from src.vector_store import VectorStore
    store = VectorStore()

    chain = create_rag_chain(retriever=store)

    # 测试单次问答
    print("=" * 60)
    print("单次问答测试")
    print("=" * 60)
    answer = chain.ask("什么是 RAG？")
    print(f"回答: {answer}")

    # 测试带来源
    print("\n" + "=" * 60)
    print("带来源问答测试")
    print("=" * 60)
    answer, sources = chain.ask_with_sources("RAG 的工作原理")
    print(f"回答: {answer}")
    print(f"来源: {len(sources)} 篇")
