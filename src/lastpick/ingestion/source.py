"""A future provider supplies discovery pages and raw detail payloads."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DiscoveryPage:
    entries: tuple[dict[str, Any], ...]
    next_cursor: int | None


class MatchSource(Protocol):
    name: str

    def discover_matches(self, cursor: int | None) -> DiscoveryPage: ...

    def fetch_match(self, match_id: int) -> Any: ...
