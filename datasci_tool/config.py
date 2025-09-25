"""Configuration dataclasses for the research pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class AgentConfig:
    """Configuration for the OpenAI research agent."""

    model: str = "gpt-4.1-mini"
    instructions: str = (
        "You are an autonomous research assistant that specializes in data science. "
        "Given a topic, you plan a brief research sprint, browse credible sources, "
        "extract the most important quantitative insights, and compile a factual summary."
    )
    tools: List[dict] = field(
        default_factory=lambda: [
            {"type": "web_search"},
            {"type": "code_interpreter"},
        ]
    )
    max_search_results: int = 5


@dataclass(slots=True)
class EmbeddingConfig:
    """Configuration for the embedding generator."""

    model: str = "text-embedding-3-large"


@dataclass(slots=True)
class VectorStoreConfig:
    """Configuration for the local vector store."""

    dimension: int = 3072
    similarity_top_k: int = 5


@dataclass(slots=True)
class PipelineConfig:
    """Top level configuration for the research pipeline."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    summary_model: Optional[str] = "gpt-4.1-mini"
    organization: Optional[str] = None
    project: Optional[str] = None
