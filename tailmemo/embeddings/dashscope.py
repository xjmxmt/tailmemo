import os
from typing import Literal, Optional

from tailmemo.configs.embeddings.base import BaseEmbedderConfig
from tailmemo.embeddings.base import EmbeddingBase

try:
    import dashscope
except ImportError:
    raise ImportError("dashscope is not installed. "
                      "Please install it using `pip install dashscope`")


class DashScopeEmbedding(EmbeddingBase):
    def __init__(self, config: Optional[BaseEmbedderConfig] = None):
        super().__init__(config)

        self.config.model = self.config.model or "text-embedding-v4"
        self.config.embedding_dims = self.config.embedding_dims or 1536

        self.api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        """
        Get the embedding for the given text using Langchain.

        Args:
            text (str): The text to embed.
            memory_action (optional): The type of embedding to use. Must be one of "add", "search", or "update". Defaults to None.
        Returns:
            list: The embedding vector.
        """

        return (
            dashscope.TextEmbedding.call(
                model=self.config.model,
                input=text,
                dimension=self.config.embedding_dims,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
                output_type="dense"
            )["output"]["embeddings"][0]["embedding"]
        )
