"""Indexing: documents are split into chunks and each chunk inherits the document's access groups."""

from __future__ import annotations

from pathlib import Path

from aclrag.models import Chunk


class IngestionError(Exception):
    pass


def parse_document(path: Path) -> tuple[dict[str, str], str]:
    """A markdown file with a front matter block of `key: value` lines between two `---` lines."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise IngestionError(f"{path.name}: no front matter")
    _, header, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, body.strip()


def split(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    """Paragraph-aware splitting; a long paragraph is cut with overlap."""
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in text.split("\n\n") if p.strip()):
        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
            continue
        if current:
            pieces.append(current)
        while len(paragraph) > size:
            pieces.append(paragraph[:size])
            paragraph = paragraph[size - overlap :]
        current = paragraph
    return [*pieces, current] if current else pieces


def load_corpus(folder: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*.md")):
        meta, body = parse_document(path)
        groups = tuple(g.strip() for g in meta.get("groups", "").split(",") if g.strip())
        if not groups:
            # A document without access groups would be readable by nobody or by everybody,
            # depending on how the filter is written. Neither is a decision to make silently.
            raise IngestionError(f"{path.name}: a document must name the groups that may read it")
        for n, piece in enumerate(split(body)):
            chunks.append(
                Chunk(
                    f"{path.stem}#{n}",
                    path.stem,
                    meta.get("title", path.stem),
                    piece,
                    groups,
                    meta.get("url", ""),
                )
            )
    return chunks
