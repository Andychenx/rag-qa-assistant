"""
嵌入向量模块 — 多后端统一接口，为向量检索提供文本向量化能力

技术栈:
  - OpenAI 兼容 API:   text-embedding-3-small 等（可指向 DeepSeek 等兼容服务）
  - Ollama:            本地部署模型（nomic-embed-text / llama3 等）
  - sentence-transformers: 本地 HuggingFace 模型（all-MiniLM-L6-v2 等，默认推荐）

特性:
  - 统一接口: embed(text) / embed_batch(texts) 适配所有后端
  - 后端自动切换: 按配置选择 Local / OpenAI 兼容 / Ollama
  - 批量处理: 自动分批、错误重试、指数退避
  - 嵌入缓存: 内存缓存相同文本，避免重复计算
  - 降级策略: 指定后端不可用时自动尝试其他后端
  - 注: DeepSeek 不提供原生嵌入 API，建议使用 local 或兼容的嵌入服务
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 自定义异常
# ---------------------------------------------------------------------------

class EmbeddingError(RuntimeError):
    """嵌入生成失败"""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """嵌入维度不匹配"""


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseEmbedding(ABC):
    """嵌入模型抽象基类"""

    # 默认批量大小
    DEFAULT_BATCH_SIZE: int = 32

    def __init__(self, cache_size: int = 10000):
        self.dimension: Optional[int] = None
        self.model_name: str = ""
        self._cache: Dict[str, List[float]] = {}
        self._cache_size = cache_size

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """将单条文本转换为向量"""

    def embed_batch(self, texts: Sequence[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """批量将文本转换为向量，自动分片

        Args:
            texts:      文本列表
            batch_size: 每批最大数量（None 使用默认值）

        Returns:
            向量列表，顺序与输入一致
        """
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        results: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vectors = self._embed_batch_impl(batch)
            results.extend(batch_vectors)

        return results

    @abstractmethod
    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        """子类实现的批量嵌入方法"""

    def _check_cache(self, text: str) -> Optional[List[float]]:
        """检查缓存"""
        return self._cache.get(text)

    def _update_cache(self, text: str, vector: List[float]) -> None:
        """更新缓存，LRU 淘汰"""
        if len(self._cache) >= self._cache_size:
            # 移除最早的一半
            keys = list(self._cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._cache[k]
        self._cache[text] = vector

    def clear_cache(self) -> None:
        """清空嵌入缓存"""
        self._cache.clear()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.model_name} dim={self.dimension}>"


# ---------------------------------------------------------------------------
# OpenAI 嵌入
# ---------------------------------------------------------------------------

class OpenAIEmbedding(BaseEmbedding):
    """OpenAI 兼容 API 嵌入封装

    支持任意 OpenAI 兼容的嵌入 API（包括 DeepSeek、通义千问等服务商）。

    支持模型:
      - text-embedding-3-small   (默认，1536 维，性价比高)
      - text-embedding-3-large   (3072 维，精度最高)
      - text-embedding-ada-002   (1536 维，兼容旧版)
      - 其他兼容服务商的嵌入模型

    使用 DeepSeek 时注意:
      DeepSeek 不提供原生嵌入 API，建议改用 local 后端
    """

    DEFAULT_BATCH_SIZE = 2048  # OpenAI 最大批处理限制

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,  # text-embedding-3 系列支持截断维度
        max_retries: int = 3,
        cache_size: int = 10000,
    ):
        super().__init__(cache_size=cache_size)
        self.model_name = model
        self.dimension = dimensions or self.MODEL_DIMENSIONS.get(model, 1536)
        self.max_retries = max_retries

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI 包未安装: pip install openai")

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def embed(self, text: str) -> List[float]:
        # 检查缓存
        cached = self._check_cache(text)
        if cached is not None:
            return cached

        vector = self._call_api([text])[0]
        self._update_cache(text, vector)
        return vector

    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        # 批量调用，检查缓存
        uncached_texts = []
        uncached_indices = []
        results: List[Optional[List[float]]] = [None] * len(texts)

        for i, t in enumerate(texts):
            cached = self._check_cache(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if not uncached_texts:
            return [r for r in results if r is not None]

        vectors = self._call_api(uncached_texts)
        for idx, vec in zip(uncached_indices, vectors):
            results[idx] = vec
            self._update_cache(texts[idx], vec)

        return [r for r in results if r is not None]

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenAI API，含重试与指数退避"""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }
        # text-embedding-3 系列支持 dimensions 参数
        if self.model_name.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimension

        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(**kwargs)
                # 按输入顺序排序
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                return [d.embedding for d in sorted_data]

            except Exception as e:
                logger.warning(
                    "OpenAI API 调用失败 (第 %d/%d 次): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    raise EmbeddingError(
                        f"OpenAI 嵌入 API 调用失败 (重试 {self.max_retries} 次): {e}"
                    ) from e

        return []  # unreachable


# ---------------------------------------------------------------------------
# Ollama 嵌入
# ---------------------------------------------------------------------------

class OllamaEmbedding(BaseEmbedding):
    """Ollama 本地嵌入模型封装

    常用模型:
      - nomic-embed-text     (768 维，默认，轻量高效)
      - mxbai-embed-large    (1024 维，精度更高)
      - llama3 等 LLM 也可作为嵌入模型使用

    使用前需在本地启动 Ollama 服务:
      ollama pull nomic-embed-text
      ollama serve
    """

    DEFAULT_BATCH_SIZE = 32

    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
        "llama3": 4096,      # 使用 LLM 最后一层作为嵌入
        "llama3.1": 4096,
        "qwen2": 4096,
    }

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dimension: Optional[int] = None,
        cache_size: int = 10000,
        request_timeout: int = 60,
    ):
        super().__init__(cache_size=cache_size)
        self.model_name = model
        self.dimension = dimension or self.MODEL_DIMENSIONS.get(model, 768)
        self.base_url = base_url.rstrip("/")
        self.timeout = request_timeout

        try:
            import requests as _req
            self._requests = _req
        except ImportError:
            raise ImportError("requests 未安装: pip install requests")

        # 检查 Ollama 服务是否可用
        self._check_service()

    def _check_service(self) -> None:
        """检查 Ollama 服务状态"""
        try:
            resp = self._requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info("Ollama 服务已连接, 可用模型: %s", models)

            # 检查目标模型是否存在
            if not any(self.model_name in m for m in models):
                logger.warning(
                    "模型 '%s' 未在 Ollama 中找到。可用: %s。请执行: ollama pull %s",
                    self.model_name, models, self.model_name,
                )
        except Exception as e:
            logger.warning("Ollama 服务连接失败: %s (请运行 ollama serve)", e)

    def embed(self, text: str) -> List[float]:
        cached = self._check_cache(text)
        if cached is not None:
            return cached

        vector = self._ollama_embed([text])[0]
        self._update_cache(text, vector)
        return vector

    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        uncached_texts = []
        uncached_indices = []
        results: List[Optional[List[float]]] = [None] * len(texts)

        for i, t in enumerate(texts):
            cached = self._check_cache(t)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if not uncached_texts:
            return [r for r in results if r is not None]

        vectors = self._ollama_embed(uncached_texts)
        for idx, vec in zip(uncached_indices, vectors):
            results[idx] = vec
            self._update_cache(texts[idx], vec)

        return [r for r in results if r is not None]

    def _ollama_embed(self, texts: List[str]) -> List[List[float]]:
        """调用 Ollama 嵌入 API"""
        try:
            resp = self._requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/embed 返回 {"embeddings": [[...], ...]}
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise EmbeddingError("Ollama 返回空嵌入")
            return embeddings

        except self._requests.exceptions.Timeout as e:
            raise EmbeddingError(
                f"Ollama 请求超时 ({self.timeout}s): {e}"
            ) from e
        except self._requests.exceptions.ConnectionError as e:
            raise EmbeddingError(
                f"无法连接 Ollama 服务 ({self.base_url}): {e}\n"
                f"请确认已运行: ollama serve"
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Ollama 嵌入失败: {e}") from e


# ---------------------------------------------------------------------------
# 本地嵌入 (sentence-transformers)
# ---------------------------------------------------------------------------

class LocalEmbedding(BaseEmbedding):
    """本地 sentence-transformers 嵌入模型

    常用模型 (HuggingFace):
      - all-MiniLM-L6-v2        (384 维，最快，默认)
      - all-mpnet-base-v2       (768 维，精度最高)
      - multilingual-e5-small   (384 维，多语言)
      - BAAI/bge-small-zh-v1.5  (512 维，中文优化)
      - BAAI/bge-large-zh-v1.5  (1024 维，中文，精度更好)
      - shibing624/text2vec-base-chinese (768 维，中文)
    """

    DEFAULT_BATCH_SIZE = 64

    # 常见模型的维度（不确定时自动检测）
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "intfloat/multilingual-e5-small": 384,
        "intfloat/multilingual-e5-base": 768,
        "intfloat/e5-large-v2": 1024,
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-large-zh-v1.5": 1024,
        "shibing624/text2vec-base-chinese": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_size: int = 10000,
        trust_remote_code: bool = False,
    ):
        super().__init__(cache_size=cache_size)
        self.model_name = model_name
        self._device = device
        self._trust_remote_code = trust_remote_code

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装: pip install sentence-transformers"
            )

        try:
            self._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "本地嵌入模型加载完成: %s (维度=%d, 设备=%s)",
                model_name, self.dimension,
                self._model.device,
            )
        except Exception as e:
            raise EmbeddingError(f"加载嵌入模型失败 '{model_name}': {e}") from e

    def embed(self, text: str) -> List[float]:
        cached = self._check_cache(text)
        if cached is not None:
            return cached

        vector = self._model.encode(text).tolist()
        self._update_cache(text, vector)
        return vector

    def _embed_batch_impl(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()


# ---------------------------------------------------------------------------
# 嵌入模型工厂
# ---------------------------------------------------------------------------

class EmbeddingFactory:
    """嵌入模型工厂 — 根据配置自动创建合适的嵌入后端

    用法:
        # 自动选择（推荐，优先本地模型）
        emb = EmbeddingFactory.create("auto")
        # 指定后端
        emb = EmbeddingFactory.create("local", model_name="all-MiniLM-L6-v2")
        emb = EmbeddingFactory.create("openai", model="text-embedding-3-small")
        emb = EmbeddingFactory.create("ollama", model="nomic-embed-text")

        # 从配置字典加载
        emb = EmbeddingFactory.from_config({
            "provider": "local",
            "model_name": "all-MiniLM-L6-v2",
        })
    """

    PROVIDERS = {
        "openai": OpenAIEmbedding,
        "ollama": OllamaEmbedding,
        "local": LocalEmbedding,
        "sentence-transformers": LocalEmbedding,
        "sentence_transformer": LocalEmbedding,
    }

    @classmethod
    def create(
        cls,
        provider: str = "auto",
        **kwargs,
    ) -> BaseEmbedding:
        """创建嵌入模型实例

        Args:
            provider: 后端类型
                - ``"auto"``: 自动检测（优先 local → openai → ollama）
                - ``"local"``: sentence-transformers 本地模型（推荐搭配 DeepSeek）
                - ``"openai"``: OpenAI 兼容嵌入 API（需指定 api_key）
                - ``"ollama"``: Ollama 本地服务
            **kwargs: 传递给具体后端的参数

        Returns:
            BaseEmbedding 实例
        """
        provider = provider.lower()

        if provider == "auto":
            return cls._auto_select(**kwargs)

        cls_cls = cls.PROVIDERS.get(provider)
        if cls_cls is None:
            raise ValueError(
                f"不支持的嵌入后端: {provider}。可选: {list(cls.PROVIDERS.keys())}"
            )

        return cls_cls(**kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseEmbedding:
        """从配置字典创建嵌入模型

        Args:
            config: 包含 provider 和对应参数的字典
                {
                    "provider": "local",
                    "model_name": "all-MiniLM-L6-v2",
                    "cache_size": 10000,
                }
                也可用 openai 兼容服务（含 DeepSeek 等）:
                {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "sk-xxx",
                    "base_url": "https://api.openai.com/v1",
                }

        Returns:
            BaseEmbedding 实例
        """
        config = config.copy()
        provider = config.pop("provider", "auto")
        return cls.create(provider, **config)

    @classmethod
    def _auto_select(cls, **kwargs) -> BaseEmbedding:
        """自动选择可用后端（local → ollama → openai）

        DeepSeek 不提供嵌入 API，因此优先使用本地模型；
        若本地模型不可用，再尝试 Ollama 和 OpenAI 兼容服务。
        """
        # 1. 优先本地模型（推荐，无需网络，兼容所有 LLM 后端）
        model_name = kwargs.pop("model_name", "all-MiniLM-L6-v2")
        try:
            emb = LocalEmbedding(model_name=model_name, **kwargs)
            logger.info("自动选择嵌入后端: Local (%s)", emb.model_name)
            return emb
        except Exception as e:
            logger.warning("本地嵌入模型不可用: %s", e)

        # 2. 尝试 Ollama
        try:
            emb = OllamaEmbedding(**kwargs)
            logger.info("自动选择嵌入后端: Ollama (%s)", emb.model_name)
            return emb
        except Exception as e:
            logger.warning("Ollama 不可用: %s", e)

        # 3. 回退到 OpenAI 兼容 API
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key and api_key.startswith("sk-"):
            try:
                emb = OpenAIEmbedding(**kwargs)
                logger.info("自动选择嵌入后端: OpenAI 兼容 (%s)", emb.model_name)
                return emb
            except Exception as e:
                logger.warning("OpenAI 兼容 API 不可用: %s", e)

        raise EmbeddingError(
            f"所有嵌入后端均不可用。请安装 sentence-transformers 或配置 Ollama/兼容 API。\n"
            f"最后错误: {e}"  # noqa: F821
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算余弦相似度

    Args:
        vec_a, vec_b: 等长的浮点数向量

    Returns:
        [-1, 1] 范围的相似度
    """
    if len(vec_a) != len(vec_b):
        raise EmbeddingDimensionMismatchError(
            f"向量维度不匹配: {len(vec_a)} vs {len(vec_b)}"
        )

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def normalize_vector(vector: List[float]) -> List[float]:
    """L2 归一化向量"""
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


# ---------------------------------------------------------------------------
# 快捷入口
# ---------------------------------------------------------------------------

def create_embeddings(provider: str = "auto", **kwargs) -> BaseEmbedding:
    """快捷函数 — 创建嵌入模型"""
    return EmbeddingFactory.create(provider, **kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("嵌入模块自测")
    print("=" * 60)

    # 测试向量相似度
    v1 = [1.0, 2.0, 3.0]
    v2 = [1.0, 2.0, 3.0]
    v3 = [-1.0, -2.0, -3.0]
    print(f"cosine_similarity(v1, v2) = {cosine_similarity(v1, v2):.4f}  (期望 1.0)")
    print(f"cosine_similarity(v1, v3) = {cosine_similarity(v1, v3):.4f}  (期望 -1.0)")

    # 尝试创建嵌入模型
    print("\n--- 尝试创建嵌入模型 ---")
    for provider in ["auto"]:
        try:
            emb = create_embeddings(provider)
            vec = emb.embed("你好，世界！")
            print(f"[{provider}] {emb}")
            print(f"  向量维度: {len(vec)}")
            print(f"  前5维: {vec[:5]}")

            batch = emb.embed_batch(["第一句", "第二句", "第三句"])
            print(f"  批量嵌入: {len(batch)} 条, 维度={len(batch[0])}")

        except Exception as e:
            print(f"[{provider}] 错误: {e}")

    print("\n自测完成")
