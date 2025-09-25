"""Utilities for interacting with the OpenAI Agent API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, TYPE_CHECKING, Any

from .config import AgentConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai import OpenAI  # noqa: F401


@dataclass(slots=True)
class ResearchSnippet:
    """Represents a single researched web page."""

    title: str
    url: str
    content: str
    summary: str


class ResearchAgent:
    """High level wrapper around the OpenAI Agent API."""

    def __init__(
        self,
        config: AgentConfig,
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
        self._agent_id = self._ensure_agent()

    def _ensure_agent(self) -> str:
        agent = self.client.agents.create(
            name="Autonomous Data Science Researcher",
            model=self.config.model,
            instructions=self.config.instructions,
            tools=self.config.tools,
        )
        return agent.id

    def research(self, query: str) -> List[ResearchSnippet]:
        session = self.client.agents.sessions.create(agent_id=self._agent_id)
        with self.client.responses.stream(
            model=self.config.model,
            modalities=["text"],
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Research the following data science topic. Return a JSON array "
                                "of objects with title, url, summary, and the most relevant "
                                "content you discovered. Limit to "
                                f"{self.config.max_search_results} high quality sources.\n\n"
                                f"Topic: {query}"
                            ),
                        }
                    ],
                }
            ],
            session_id=session.id,
            stream=True,
        ) as stream:
            buffer: list[str] = []
            for event in stream:
                if event.type == "response.output_text.delta":
                    buffer.append(event.delta)
                elif event.type == "response.completed":
                    break
            stream.close()
        text = "".join(buffer).strip()
        snippets = self._parse_json_array(text)
        return snippets

    def _parse_json_array(self, text: str) -> List[ResearchSnippet]:
        import json

        try:
            raw: Iterable[dict] = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Agent response was not valid JSON: {text[:1000]}") from exc
        snippets: List[ResearchSnippet] = []
        for idx, item in enumerate(raw):
            try:
                snippets.append(
                    ResearchSnippet(
                        title=item["title"],
                        url=item.get("url", ""),
                        content=item.get("content", item.get("body", "")),
                        summary=item.get("summary", ""),
                    )
                )
            except KeyError as exc:  # pragma: no cover - defensive branch
                raise RuntimeError(f"Missing key in snippet #{idx}: {item}") from exc
        return snippets
