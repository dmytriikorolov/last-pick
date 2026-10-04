"""Provider adapters translate source records; downstream stages know only Match."""

from typing import Any, Protocol

from lastpick.data.schemas import Match


class MatchAdapter(Protocol):
    name: str

    def identifier(self, payload: Any) -> int | None: ...

    def normalize(self, payload: Any, discovery: dict[str, Any] | None = None) -> Match: ...
