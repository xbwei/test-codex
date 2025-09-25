"""Embedding utilities for the research pipeline."""
from __future__ import annotations

from typing import Iterable, List, Optional, TYPE_CHECKING, Any

from .config import EmbeddingConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI  # noqa: F401


class EmbeddingService:
    """Wrapper around the OpenAI embeddings endpoint."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        client: Optional[Any] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        self.config = config
        if client is None:
            from openai import OpenAI as OpenAIClient

            client = OpenAIClient(organization=organization, project=project)
        self.client = client

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        inputs = list(texts)
        if not inputs:
            return []
        response = self.client.embeddings.create(model=self.config.model, input=inputs)
        embeddings = [list(item.embedding) for item in response.data]
        return embeddings
