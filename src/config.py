"""
系统配置管理模块 — 集中管理所有可配置参数

支持:
  - 环境变量加载（.env 文件）
  - 多 Embedding 后端切换
  - LLM 模型选择
  - 分块参数 / 检索参数 / 重排序参数
  - 类型注解与验证
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 嵌入 配置
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """嵌入模型配置

    provider 可选值:
      - "local":           sentence-transformers 本地模型（默认，推荐搭配 DeepSeek LLM）
      - "openai":          OpenAI 兼容 API（可指向任意兼容的嵌入服务）
      - "ollama":          Ollama 本地服务（需启动 ollama serve）
      - "auto":            自动检测（优先 local → openai → ollama）
    """
    provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "auto")
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",  # 默认使用本地模型（DeepSeek 不提供嵌入 API）
        )
    )
    # OpenAI 兼容 API（可用于 text-embedding-3-small 等服务商）
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL")
    )
    dimensions: Optional[int] = None

    # Ollama 专用
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # Local 专用
    local_model_name: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    local_device: Optional[str] = field(
        default_factory=lambda: os.getenv("LOCAL_EMBEDDING_DEVICE")
    )


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """大语言模型配置

    provider 可选值:
      - "openai":   OpenAI 兼容 API（默认指向 DeepSeek Chat，可换任意兼容服务）
      - "ollama":   Ollama 本地模型（llama3 / qwen2 等）
    """
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        )
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )
    # Ollama 专用
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


# ---------------------------------------------------------------------------
# 分块配置
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """文档分块配置"""
    strategy: str = field(
        default_factory=lambda: os.getenv("CHUNK_STRATEGY", "recursive")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    # Token 分块专用
    token_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("TOKEN_CHUNK_SIZE", "512"))
    )
    token_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("TOKEN_CHUNK_OVERLAP", "50"))
    )
    # 语义分块专用
    semantic_max_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("SEMANTIC_MAX_CHUNK_SIZE", "2000"))
    )
    semantic_min_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("SEMANTIC_MIN_CHUNK_SIZE", "100"))
    )


# ---------------------------------------------------------------------------
# 检索配置
# ---------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5"))
    )
    search_type: str = field(
        default_factory=lambda: os.getenv("RETRIEVAL_SEARCH_TYPE", "similarity")
    )
    # MMR 专用
    mmr_fetch_k: int = field(
        default_factory=lambda: int(os.getenv("MMR_FETCH_K", "20"))
    )
    mmr_lambda: float = field(
        default_factory=lambda: float(os.getenv("MMR_LAMBDA", "0.5"))
    )
    # 混合搜索权重
    hybrid_alpha: float = field(
        default_factory=lambda: float(os.getenv("HYBRID_ALPHA", "0.7"))
    )


# ---------------------------------------------------------------------------
# 重排序配置
# ---------------------------------------------------------------------------

@dataclass
class RerankConfig:
    """重排序配置"""
    enabled: bool = field(
        default_factory=lambda: os.getenv("RERANK_ENABLED", "false").lower() == "true"
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_K", "3"))
    )
    method: str = field(
        default_factory=lambda: os.getenv("RERANK_METHOD", "keyword_boost")
    )


# ---------------------------------------------------------------------------
# 向量存储配置
# ---------------------------------------------------------------------------

@dataclass
class VectorStoreConfig:
    """向量数据库配置"""
    persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_PATH",
            str(PROJECT_ROOT / "chroma_db"),
        )
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "rag_documents")
    )


# ---------------------------------------------------------------------------
# 主配置类
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    """应用总配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)

    # 对话历史最大轮数
    max_conversation_rounds: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONVERSATION_ROUNDS", "50"))
    )

    @classmethod
    def load(cls) -> "AppConfig":
        """加载配置（尝试加载 .env 文件）"""
        _try_load_dotenv()
        return cls()

    def to_dict(self) -> dict:
        """转为字典（用于 UI 显示）"""
        return {
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "chunk": {
                "strategy": self.chunk.strategy,
                "chunk_size": self.chunk.chunk_size,
                "chunk_overlap": self.chunk.chunk_overlap,
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "search_type": self.retrieval.search_type,
            },
            "rerank": {
                "enabled": self.rerank.enabled,
                "method": self.rerank.method,
            },
        }


def _try_load_dotenv() -> None:
    """尝试加载 .env 文件（如果存在）"""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if key and not os.environ.get(key):
                        os.environ[key] = value
        except Exception:
            pass  # .env 加载失败不应阻断启动


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

settings = AppConfig.load()


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("当前配置")
    print("=" * 60)
    print(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2))
