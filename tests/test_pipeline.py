from __future__ import annotations

from unittest.mock import Mock, patch

from datasci_tool.config import PipelineConfig, VectorStoreConfig
from datasci_tool.pipeline import ResearchPipeline
from datasci_tool.research_agent import ResearchSnippet


def test_pipeline_runs_end_to_end(tmp_path):
    config = PipelineConfig()
    config.vector_store = VectorStoreConfig(dimension=3, similarity_top_k=2)

    research_snippets = [
        ResearchSnippet(
            title="Article A",
            url="https://example.com/a",
            content="Detailed notes about article A",
            summary="Key findings from A",
        ),
        ResearchSnippet(
            title="Article B",
            url="https://example.com/b",
            content="Insights from article B",
            summary="Highlights from B",
        ),
    ]

    with (
        patch("datasci_tool.pipeline.ResearchAgent") as mock_research_agent,
        patch("datasci_tool.pipeline.EmbeddingService") as mock_embedding_service,
        patch("datasci_tool.pipeline.SummaryGenerator") as mock_summary_generator,
    ):
        mock_agent_instance = mock_research_agent.return_value
        mock_agent_instance.research.return_value = research_snippets

        mock_embedding = mock_embedding_service.return_value
        mock_embedding.embed.side_effect = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.8, 0.2, 0.0]],
        ]

        mock_summary = mock_summary_generator.return_value
        mock_summary.summarize.return_value = "final summary"

        pipeline = ResearchPipeline(
            config,
            client=Mock(),
            persist_path=tmp_path / "vectors.json",
        )

        output = pipeline.run("Bayesian optimization use cases")

        assert output.summary == "final summary"
        assert output.snippets == research_snippets
        assert len(output.similar_documents) == 2
        assert mock_summary.summarize.called
        assert mock_embedding.embed.called
        assert len(pipeline.vector_store) == 2
