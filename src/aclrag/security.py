"""The security filter. It is built from the user's groups and applied inside the search engine,
so a passage the user may not read is never retrieved and never reaches the model."""

from __future__ import annotations

import re
from dataclasses import dataclass

DELIMITER = ","
_VALID_GROUP = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")


class AccessDenied(Exception):
    pass


@dataclass(frozen=True)
class GroupFilter:
    groups: frozenset[str]

    def allows(self, chunk_groups: tuple[str, ...]) -> bool:
        return bool(self.groups.intersection(chunk_groups))

    def to_odata(self, field: str = "group_ids") -> str:
        """Filter expression for Azure AI Search."""
        joined = DELIMITER.join(sorted(self.groups))
        return f"{field}/any(g: search.in(g, '{joined}', '{DELIMITER}'))"


def build_group_filter(groups: tuple[str, ...] | list[str]) -> GroupFilter:
    """Fails closed: no groups, or a group id that could alter the filter expression, means no search."""
    if not groups:
        raise AccessDenied("the user belongs to no group; nothing can be searched on their behalf")
    for group in groups:
        if not _VALID_GROUP.fullmatch(group):
            raise AccessDenied(f"group id {group!r} is not acceptable in a filter")
    return GroupFilter(frozenset(groups))
