"""
智能 RAG 问答助手 — Streamlit Web UI

功能:
  - 文件上传（支持 PDF / TXT / MD / DOCX，多文件拖拽）
  - 文档处理进度展示
  - 类 ChatGPT 聊天界面
  - 来源展示（展开可查看原始段落）
  - 历史对话管理
  - 系统设置（分块参数、模型选择、TopK、检索方式调整）

启动:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 将项目根目录加入 PATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

# 页面配置 — 必须在所有 st 命令之前
st.set_page_config(
    page_title="智能 RAG 问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import settings
from src.document_loader import DocumentLoader
from src.splitter import split_documents, TextSplitter
from src.vector_store import VectorStore
from src.qa_chain import RAGChain, ConversationMemory
from src.query_transform import QueryTransformer
from src.reranker import Reranker

logger = logging.getLogger(__name__)


# ============================================================================
# 会话状态初始化
# ============================================================================

def init_session_state() -> None:
    """初始化 Streamlit 会话状态"""
    if "initialized" in st.session_state:
        return

    st.session_state.initialized = True
    st.session_state.document_loader = DocumentLoader()
    st.session_state.vector_store = VectorStore()
    st.session_state.rag_chain = None
    st.session_state.messages: List[Dict[str, Any]] = []
    st.session_state.processed_files: List[str] = []
    st.session_state.chunk_stats: Optional[Dict[str, Any]] = None
    st.session_state.last_sources: List[Dict[str, Any]] = []
    st.session_state.show_sources = False

    # 当前设置（UI 可调）
    st.session_state.chunk_size = settings.chunk.chunk_size
    st.session_state.chunk_overlap = settings.chunk.chunk_overlap
    st.session_state.chunk_strategy = settings.chunk.strategy
    st.session_state.top_k = settings.retrieval.top_k
    st.session_state.search_type = settings.retrieval.search_type
    st.session_state.use_hybrid = True
    st.session_state.use_rerank = settings.rerank.enabled
    st.session_state.llm_model = settings.llm.model
    st.session_state.llm_temperature = settings.llm.temperature

    st.session_state.expand_queries = False
    st.session_state.sub_queries = False


# ============================================================================
# 文档处理
# ============================================================================

def process_uploaded_files(
    uploaded_files: List,
    chunk_size: int,
    chunk_overlap: int,
    chunk_strategy: str,
    progress_bar,
    status_text,
) -> int:
    """处理上传的文件：加载 → 分块 → 向量化存储

    Args:
        uploaded_files:  上传的文件列表
        chunk_size:      分块大小
        chunk_overlap:   分块重叠
        chunk_strategy:  分块策略
        progress_bar:    Streamlit 进度条
        status_text:     Streamlit 状态文本

    Returns:
        存储的文档块数
    """
    loader = st.session_state.document_loader
    vector_store = st.session_state.vector_store

    total_files = len(uploaded_files)
    all_chunks = []

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name

        if file_name in st.session_state.processed_files:
            status_text.info(f"⏭ 跳过已处理的文件: {file_name}")
            continue

        # 进度更新
        progress = (idx) / total_files
        progress_bar.progress(progress)
        status_text.info(f"📖 正在加载: {file_name}")

        # 1. 保存上传文件到临时路径
        temp_path = os.path.join(
            settings.vector_store.persist_dir,
            "..",
            "data",
            "temp",
            file_name,
        )
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. 加载文档
        try:
            doc = loader.load(temp_path)
        except Exception as e:
            status_text.error(f"❌ 加载失败 {file_name}: {e}")
            continue

        # 3. 分块
        status_text.info(f"✂️ 正在分块: {file_name} (策略: {chunk_strategy})")

        try:
            if chunk_strategy == "recursive":
                chunks = TextSplitter.recursive_split(
                    [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                )
            elif chunk_strategy == "semantic":
                chunks = TextSplitter.semantic_split([doc])
            elif chunk_strategy == "token":
                chunks = TextSplitter.token_split(
                    [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                )
            elif chunk_strategy == "markdown_header":
                chunks = TextSplitter.markdown_header_split([doc])
            else:
                chunks = TextSplitter.recursive_split(
                    [doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                )
        except Exception as e:
            status_text.error(f"❌ 分块失败 {file_name}: {e}")
            continue

        all_chunks.extend(chunks)
        st.session_state.processed_files.append(file_name)

    if not all_chunks:
        progress_bar.progress(1.0)
        status_text.warning("⚠️ 没有新文件需要处理")
        return 0

    # 4. 向量化存储
    status_text.info(f"💾 正在存储 {len(all_chunks)} 个文档块到向量数据库...")
    try:
        count = vector_store.store_vectors(all_chunks)
    except Exception as e:
        status_text.error(f"❌ 向量存储失败: {e}")
        return 0

    # 5. 更新统计
    st.session_state.chunk_stats = TextSplitter.stats(all_chunks)

    progress_bar.progress(1.0)
    status_text.success(f"✅ 处理完成: {len(all_chunks)} 个文档块已存储")

    return count


# ============================================================================
# 对话
# ============================================================================

def handle_user_input(user_input: str) -> None:
    """处理用户输入，生成回答

    Args:
        user_input: 用户输入的文本
    """
    if not user_input.strip():
        return

    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            # 创建或获取 RAGChain
            chain = get_or_create_rag_chain()

            if chain is None:
                st.error("请先上传文档并完成处理")
                return

            # 获取最近历史用于查询重写
            history_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:-1]
            ]

            # 查询重写 / HyDE / 子查询
            query = user_input

            # 子查询分解
            if st.session_state.sub_queries and len(user_input) > 10:
                sub_queries = chain.query_transformer.generate_sub_queries(user_input)
                if len(sub_queries) > 1:
                    all_docs = []
                    for sq in sub_queries:
                        if chain.retriever:
                            docs = chain.retriever.similarity_search(sq, k=st.session_state.top_k // 2)
                            all_docs.extend(docs)
                    # 若存在检索器且收集到文档，暂存供后续使用
                    if all_docs and chain.retriever:
                        pass  # 走正常流程

            # 主检索 + 生成
            answer, sources = chain.ask_with_sources(
                query,
                history=history_msgs,
                use_hybrid=st.session_state.use_hybrid,
            )

            # 保存来源
            st.session_state.last_sources = sources

        # 显示回答
        st.markdown(answer)

        # 显示来源（可展开）
        if sources:
            with st.expander("📎 查看来源", expanded=False):
                for src in sources:
                    score = src.get("score", 0)
                    rerank_score = src.get("rerank_score")
                    score_str = f"相关性: {score:.4f}"
                    if rerank_score is not None:
                        score_str += f" | 重排分: {rerank_score:.4f}"

                    st.markdown(f"**来源 #{src['index']}** ({score_str})")
                    st.text_area(
                        "",
                        value=src["content"],
                        height=100,
                        key=f"source_{src['index']}_{int(time.time())}",
                        label_visibility="collapsed",
                    )

    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 滚动到最新
    st.rerun()


def get_or_create_rag_chain() -> Optional[RAGChain]:
    """获取或创建 RAGChain 实例"""
    if st.session_state.rag_chain is not None:
        return st.session_state.rag_chain

    vector_store = st.session_state.vector_store
    if vector_store.count() == 0:
        return None

    # 创建 RAGChain
    chain = RAGChain(retriever=vector_store)
    st.session_state.rag_chain = chain

    return chain


# ============================================================================
# 侧边栏
# ============================================================================

def render_sidebar() -> None:
    """渲染侧边栏"""
    with st.sidebar:
        st.title("📚 智能 RAG 问答助手")
        st.markdown("---")

        # ---- 文件上传区 ----
        st.subheader("📁 上传文档")
        uploaded_files = st.file_uploader(
            "支持 PDF / TXT / MD / DOCX 格式",
            type=["pdf", "txt", "md", "markdown", "docx"],
            accept_multiple_files=True,
            help="拖拽或点击上传文档文件",
        )

        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                process_btn = st.button(
                    "🚀 处理文档",
                    type="primary",
                    use_container_width=True,
                )
            with col2:
                clear_btn = st.button("🗑 清空", use_container_width=True)

            if clear_btn:
                st.session_state.processed_files = []
                st.session_state.vector_store.clear()
                st.session_state.rag_chain = None
                st.session_state.chunk_stats = None
                st.success("已清空所有文档")
                st.rerun()

            if process_btn:
                progress_bar = st.progress(0)
                status_text = st.empty()

                count = process_uploaded_files(
                    uploaded_files=uploaded_files,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                    chunk_strategy=st.session_state.chunk_strategy,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

                if count > 0:
                    # 重新创建 RAGChain
                    st.session_state.rag_chain = None
                    st.success(f"✅ 成功处理 {count} 个文档块！")

        # ---- 已处理文件 ----
        if st.session_state.processed_files:
            st.markdown("---")
            st.subheader("📄 已处理文档")
            for f in st.session_state.processed_files:
                st.caption(f"✅ {f}")

        # ---- 向量库统计 ----
        vs = st.session_state.vector_store
        doc_count = vs.count()
        if doc_count > 0:
            st.markdown("---")
            st.subheader("📊 向量库状态")
            st.metric("文档块总数", doc_count)

            if st.session_state.chunk_stats:
                stats = st.session_state.chunk_stats
                st.caption(f"平均大小: {stats.get('avg_chunk_size', 0):.0f} 字符")
                st.caption(f"总字符数: {stats.get('total_chars', 0):,}")

            if vs.is_persisted:
                st.caption(f"存储大小: {vs.get_persist_size()}")

        # ---- 系统设置 ----
        st.markdown("---")
        st.subheader("⚙️ 系统设置")

        with st.expander("🔧 高级设置", expanded=False):
            # LLM 设置
            st.markdown("**🤖 LLM 设置**")
            st.session_state.llm_model = st.text_input(
                "模型名称",
                value=st.session_state.llm_model,
                help="使用的 LLM 模型",
            )
            st.session_state.llm_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.llm_temperature,
                step=0.1,
                help="生成温度，越高越有创造性",
            )

            # 分块设置
            st.markdown("**✂️ 分块设置**")
            st.session_state.chunk_strategy = st.selectbox(
                "分块策略",
                options=["recursive", "semantic", "token", "markdown_header"],
                index=["recursive", "semantic", "token", "markdown_header"].index(
                    st.session_state.chunk_strategy
                ),
                help="分块策略选择",
            )
            st.session_state.chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=4000,
                value=st.session_state.chunk_size,
                step=100,
                help="每个文档块的最大字符/token数",
            )
            st.session_state.chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.chunk_overlap,
                step=50,
                help="文档块之间的重叠字符数",
            )

            # 检索设置
            st.markdown("**🔍 检索设置**")
            st.session_state.top_k = st.number_input(
                "Top K",
                min_value=1,
                max_value=20,
                value=st.session_state.top_k,
                step=1,
                help="检索返回的文档块数量",
            )
            st.session_state.use_hybrid = st.checkbox(
                "混合检索（向量 + 关键词）",
                value=st.session_state.use_hybrid,
                help="同时使用向量相似度和关键词检索",
            )
            st.session_state.use_rerank = st.checkbox(
                "启用重排序",
                value=st.session_state.use_rerank,
                help="对检索结果进行二次排序",
            )
            st.session_state.sub_queries = st.checkbox(
                "复杂问题拆解",
                value=st.session_state.sub_queries,
                help="自动将复杂问题拆分为多个子查询",
            )

            # 应用设置按钮
            if st.button("🔄 应用设置并重建链", use_container_width=True):
                # 更新配置
                settings.chunk.chunk_size = st.session_state.chunk_size
                settings.chunk.chunk_overlap = st.session_state.chunk_overlap
                settings.chunk.strategy = st.session_state.chunk_strategy
                settings.retrieval.top_k = st.session_state.top_k
                settings.rerank.enabled = st.session_state.use_rerank
                settings.llm.model = st.session_state.llm_model
                settings.llm.temperature = st.session_state.llm_temperature

                # 重建 RAGChain
                st.session_state.rag_chain = None
                st.success("✅ 设置已更新！")

        # ---- 对话管理 ----
        st.markdown("---")
        st.subheader("💬 对话管理")

        if st.button("🔄 新建对话", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.rerun()

        if st.button("📋 导出对话", use_container_width=True):
            if st.session_state.messages:
                export_text = "# 对话导出\n\n"
                for msg in st.session_state.messages:
                    role = "用户" if msg["role"] == "user" else "助手"
                    export_text += f"**{role}**: {msg['content']}\n\n"
                st.download_button(
                    "⬇️ 下载对话",
                    data=export_text,
                    file_name=f"rag_chat_export_{int(time.time())}.md",
                    mime="text/markdown",
                )

        # ---- 项目信息 ----
        st.markdown("---")
        st.caption(
            "智能 RAG 问答助手 v1.0\n"
            "基于检索增强生成技术"
        )


# ============================================================================
# 主界面
# ============================================================================

def render_main() -> None:
    """渲染主聊天界面"""
    # 标题
    st.title("💬 智能 RAG 问答助手")

    # 状态提示
    vs = st.session_state.vector_store
    doc_count = vs.count()

    if doc_count == 0:
        # 空白状态
        st.info(
            "👋 欢迎使用智能 RAG 问答助手！\n\n"
            "请在左侧边栏上传文档（PDF/TXT/MD/DOCX），"
            "点击 **「处理文档」** 开始使用。\n\n"
            "上传后，您可以基于文档内容进行多轮对话。"
        )

        # 示例展示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### 📖 文档加载")
            st.caption("支持 PDF、TXT、Markdown、Word 等多种格式")
        with col2:
            st.markdown("##### 🔍 智能检索")
            st.caption("向量相似度 + 关键词混合搜索，精准定位")
        with col3:
            st.markdown("##### 💡 增强生成")
            st.caption("基于上下文生成准确回答，可追溯来源")

        st.markdown("---")
        return

    # 显示文档处理状态
    st.caption(
        f"📚 已处理 {len(st.session_state.processed_files)} 个文档，"
        f"共 {doc_count} 个文档块"
    )

    # 分隔线
    st.divider()

    # ---- 聊天消息展示 ----
    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---- 输入框 ----
    user_input = st.chat_input(
        "请输入您的问题...",
        disabled=(doc_count == 0),
    )

    if user_input:
        handle_user_input(user_input)


# ============================================================================
# 入口
# ============================================================================

def main() -> None:
    """应用入口"""
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    main()
