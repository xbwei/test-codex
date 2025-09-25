"""High level orchestration for the research workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING, Any

from .config import PipelineConfig
from .embeddings import EmbeddingService
from .research_agent import ResearchAgent, ResearchSnippet
from .summary import SummaryGenerator
from .vector_store import LocalVectorStore, StoredDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI  # noqa: F401


@dataclass(slots=True)
class ResearchOutput:
    """Represents the artifacts produced by a pipeline run."""

    query: str
    snippets: list[ResearchSnippet]
    summary: str
    similar_documents: list[StoredDocument]


class ResearchPipeline:
    """Coordinates the research agent, embeddings, and vector database."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        client: Optional[Any] = None,
        persist_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        if client is None:
            from openai import OpenAI as OpenAIClient

            client = OpenAIClient(
                organization=config.organization,
                project=config.project,
            )
        self.client = client
        self.vector_store = LocalVectorStore(
            persist_path=persist_path or Path(".artifacts/vector_store.json"),
            vector_dimension=config.vector_store.dimension,
        )
        self.research_agent = ResearchAgent(
            config=config.agent,
            client=self.client,
        )
        self.embedding_service = EmbeddingService(
            config=config.embeddings,
            client=self.client,
        )
        self.summary_generator = (
            SummaryGenerator(config.summary_model, client=self.client)
            if config.summary_model
            else None
        )

    def run(self, query: str) -> ResearchOutput:
        """Execute the end-to-end workflow for a single user query."""
        snippets = self.research_agent.research(query)
        embeddings = self.embedding_service.embed(snippet.content for snippet in snippets)
        documents: list[StoredDocument] = []
        for snippet, embedding in zip(snippets, embeddings, strict=True):
            document_id = snippet.url or snippet.title
            documents.append(
                StoredDocument(
                    document_id=document_id,
                    text=snippet.content,
                    metadata={
                        "title": snippet.title,
                        "url": snippet.url,
                        "summary": snippet.summary,
                    },
                    embedding=embedding,
                )
            )
        self.vector_store.add(documents)
        query_embedding = self.embedding_service.embed([query])[0]
        similar_documents = self.vector_store.query(
            query_embedding, self.config.vector_store.similarity_top_k
        )
        summary = self._build_summary(query, snippets)
        return ResearchOutput(
            query=query,
            snippets=snippets,
            summary=summary,
            similar_documents=list(similar_documents),
        )

    def _build_summary(self, query: str, snippets: Iterable[ResearchSnippet]) -> str:
        bullet_points = [
            f"{snippet.title}: {snippet.summary or snippet.content[:200]}" for snippet in snippets
        ]
        if not bullet_points:
            return "No research findings were produced."
        if self.summary_generator is None:
            return "\n".join(bullet_points)
        return self.summary_generator.summarize(query, bullet_points)
