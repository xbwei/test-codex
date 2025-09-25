"""Command line interface for running the research pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasci_tool.config import PipelineConfig
from datasci_tool.pipeline import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous data science research pipeline")
    parser.add_argument("query", help="Topic that the agent should research")
    parser.add_argument(
        "--persist-path",
        default=Path(".artifacts/vector_store.json"),
        type=Path,
        help="Where to persist the local vector store",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    pipeline = ResearchPipeline(config, persist_path=args.persist_path)
    output = pipeline.run(args.query)
    result = {
        "query": output.query,
        "summary": output.summary,
        "snippets": [
            {
                "title": snippet.title,
                "url": snippet.url,
                "summary": snippet.summary,
            }
            for snippet in output.snippets
        ],
        "similar_documents": [
            {
                "document_id": doc.document_id,
                "title": doc.metadata.get("title"),
                "url": doc.metadata.get("url"),
            }
            for doc in output.similar_documents
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
