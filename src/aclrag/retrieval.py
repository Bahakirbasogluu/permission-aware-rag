"""Retrievers. Both apply the same security filter; one runs in memory, one in Azure AI Search."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Protocol

from aclrag.models import Chunk
from aclrag.security import GroupFilter

_STOP_WORDS = (
    "the a an of to in on for and or is are was be by with at as it this that what which how who when where do does "
    "can i my our your we you they their from not no yes any all per each than then if but about into out up there "
    "ve veya ile bir bu şu için mi mı mu mü ne nedir nasıl kaç hangi olan olarak de da ki en çok daha gibi her"
)
_STOP = frozenset(_STOP_WORDS.split())


def fold(text: str) -> str:
    return text.replace("İ", "i").replace("ı", "i").lower()


def stems(text: str) -> list[str]:
    """Words reduced to a short prefix, which is enough to match Turkish and English inflections."""
    return [w[:5] for w in re.findall(r"[^\W_]+", fold(text)) if w not in _STOP and len(w) > 1]


class Retriever(Protocol):
    def search(self, query: str, group_filter: GroupFilter, top_k: int) -> list[Chunk]: ...


class InMemoryRetriever:
    """Keyword retrieval with TF-IDF weights, for local runs, tests and the evaluation."""

    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._terms = [Counter(stems(f"{c.title} {c.text}")) for c in chunks]
        df = Counter(term for terms in self._terms for term in terms)
        self._idf = {term: math.log(1 + len(chunks) / n) for term, n in df.items()}

    def search(self, query: str, group_filter: GroupFilter, top_k: int) -> list[Chunk]:
        wanted = set(stems(query))
        scored = []
        for chunk, terms in zip(self._chunks, self._terms):
            if not group_filter.allows(chunk.group_ids):  # filtered before scoring, not after
                continue
            score = sum(self._idf[t] * (1 + math.log(terms[t])) for t in wanted if t in terms)
            if score > 0:
                scored.append((score, chunk))
        return [
            chunk for _, chunk in sorted(scored, key=lambda pair: (-pair[0], pair[1].id))[:top_k]
        ]


class AzureSearchRetriever:
    """Hybrid (keyword + vector) search with semantic ranking in Azure AI Search.

    The index stores `group_ids` on every chunk as a filterable collection, and
    the query vector is produced by the index's integrated vectoriser.
    """

    def __init__(self, vector_field: str = "text_vector", semantic_configuration: str = "default"):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self._client = SearchClient(
            os.environ["AZURE_SEARCH_ENDPOINT"],
            os.environ["AZURE_SEARCH_INDEX"],
            AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]),
        )
        self._vector_field, self._semantic = vector_field, semantic_configuration

    def search(self, query: str, group_filter: GroupFilter, top_k: int) -> list[Chunk]:
        from azure.search.documents.models import VectorizableTextQuery

        results = self._client.search(
            search_text=query,
            vector_queries=[
                VectorizableTextQuery(
                    text=query, k_nearest_neighbors=top_k * 5, fields=self._vector_field
                )
            ],
            filter=group_filter.to_odata(),
            query_type="semantic",
            semantic_configuration_name=self._semantic,
            top=top_k,
        )
        return [
            Chunk(
                r["id"],
                r["doc_id"],
                r["title"],
                r["text"],
                tuple(r.get("group_ids", ())),
                r.get("source_url", ""),
            )
            for r in results
        ]
