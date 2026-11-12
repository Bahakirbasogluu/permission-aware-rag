from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    """The signed-in user as the identity provider describes them."""

    user_id: str
    groups: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    """A passage of a document. The document's access groups are stamped on every chunk."""

    id: str
    doc_id: str
    title: str
    text: str
    group_ids: tuple[str, ...]
    source_url: str = ""


@dataclass(frozen=True)
class DocRef:
    doc_id: str
    title: str
    source_url: str = ""


@dataclass
class Answer:
    text: str
    cited: list[DocRef] = field(default_factory=list)
    consulted: list[DocRef] = field(
        default_factory=list
    )  # every document whose text reached the model
    refused: bool = False
