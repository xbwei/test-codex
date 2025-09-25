"""A lightweight vector store for embedding persistence and similarity search."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, List, Sequence

import json


@dataclass(slots=True)
class StoredDocument:
    """Represents a document stored in the vector database."""

    document_id: str
    text: str
    metadata: dict
    embedding: List[float]


class LocalVectorStore:
    """Simple vector store backed by JSON persistence.

    The store keeps embeddings on disk inside a JSON file so repeated executions
    of the pipeline can reuse the previously indexed knowledge base. The
    implementation is intentionally lightweight and does not require external
    dependencies.
    """

    def __init__(self, persist_path: Path, vector_dimension: int) -> None:
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_dimension = vector_dimension
        self._documents: list[StoredDocument] = []
        if self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        with self.persist_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        documents: list[StoredDocument] = []
        for item in payload.get("documents", []):
            documents.append(
                StoredDocument(
                    document_id=item["document_id"],
                    text=item["text"],
                    metadata=item.get("metadata", {}),
                    embedding=list(item["embedding"]),
                )
            )
        self._documents = documents

    def _save(self) -> None:
        payload = {
            "documents": [
                {
                    "document_id": doc.document_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding,
                }
                for doc in self._documents
            ]
        }
        with self.persist_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(self, documents: Iterable[StoredDocument]) -> None:
        docs = list(documents)
        if not docs:
            return
        for doc in docs:
            if len(doc.embedding) != self.vector_dimension:
                raise ValueError(
                    f"Embedding for {doc.document_id} has length {len(doc.embedding)}, "
                    f"expected {self.vector_dimension}"
                )
        self._documents.extend(docs)
        self._save()

    def query(self, embedding: List[float], top_k: int) -> List[StoredDocument]:
        if not self._documents:
            return []
        if len(embedding) != self.vector_dimension:
            raise ValueError(
                f"Query embedding has length {len(embedding)}, expected {self.vector_dimension}"
            )
        scored = [
            (self._cosine_similarity(doc.embedding, embedding), doc) for doc in self._documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._documents)

    @property
    def documents(self) -> Sequence[StoredDocument]:  # pragma: no cover - trivial accessor
        return tuple(self._documents)
