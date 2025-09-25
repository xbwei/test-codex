"""Summary generation utilities."""
from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI  # noqa: F401


class SummaryGenerator:
    """Use the OpenAI Responses API to craft concise summaries."""

    def __init__(
        self,
        model: str,
        *,
        client: Optional[Any] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ) -> None:
        self.model = model
        if client is None:
            from openai import OpenAI as OpenAIClient

            client = OpenAIClient(organization=organization, project=project)
        self.client = client

    def summarize(self, query: str, bullet_points: Iterable[str]) -> str:
        bullets_text = "\n".join(f"- {point}" for point in bullet_points)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Write a concise executive summary based on the following "
                                "research findings. Highlight quantitative results and "
                                "methodological notes when available."
                                f"\n\nUser query: {query}\n\nFindings:\n{bullets_text}"
                            ),
                        }
                    ],
                }
            ],
        )
        return response.output_text
