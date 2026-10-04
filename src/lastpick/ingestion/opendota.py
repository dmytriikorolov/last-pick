"""OpenDota transport and discovery policies; credentials only from environment."""

import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lastpick.config import Filters, IngestionSettings
from lastpick.ingestion.source import DiscoveryPage


class FetchFailure(RuntimeError):
    pass


class InvalidPayload(ValueError):
    pass


class OpenDotaSource:
    name = "opendota"

    def __init__(
        self,
        settings: IngestionSettings,
        filters: Filters,
        consume_request: Callable[[], None],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.filters = filters
        self.consume_request = consume_request
        self.sleep = sleep
        self.api_key = os.environ.get("OPENDOTA_API_KEY", "")

    def request(self, route: str, params: dict[str, str | int] | None = None) -> Any:
        query = dict(params or {})
        if self.api_key:
            query["api_key"] = self.api_key
        url = f"{self.settings.api_url}/{route}"
        if query:
            url += "?" + urlencode(query)
        for attempt in range(self.settings.retries):
            self.consume_request()
            self.sleep(self.settings.delay_seconds)
            try:
                with urlopen(
                    Request(url, headers={"User-Agent": "lastpick/2.0"}),
                    timeout=self.settings.timeout_seconds,
                ) as response:
                    return json.load(response)
            except HTTPError as error:
                retry = error.code == 429 or 500 <= error.code < 600
                if not retry or attempt + 1 == self.settings.retries:
                    # Never serialize an exception URL containing an API key.
                    raise FetchFailure(f"http_{error.code}") from None
                retry_after = error.headers.get("Retry-After") if error.headers else None
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else self.settings.retry_seconds * (2**attempt)
                )
                self.sleep(delay)
            except (URLError, TimeoutError, OSError):
                if attempt + 1 == self.settings.retries:
                    raise FetchFailure("transport_failure") from None
                self.sleep(self.settings.retry_seconds * (2**attempt))
            except (ValueError, UnicodeError):
                raise InvalidPayload("invalid_json_response") from None
        raise FetchFailure("retry_exhausted")

    def discover_matches(self, cursor: int | None) -> DiscoveryPage:
        if self.settings.discovery == "public":
            payload = self.request(
                "publicMatches", {"less_than_match_id": cursor} if cursor else {}
            )
        elif self.settings.discovery == "explorer":
            # Configured numbers only; no SQL supplied by a remote source or user string.
            rank = self.filters.min_rank if self.filters.min_rank is not None else 0
            modes = ",".join(str(int(v)) for v in self.filters.game_modes)
            lobbies = ",".join(str(int(v)) for v in self.filters.lobby_types)
            cursor_clause = f"AND match_id < {int(cursor)}" if cursor else ""
            sql = (
                "SELECT match_id, avg_rank_tier, num_rank_tier, game_mode, lobby_type "
                f"FROM public_matches WHERE avg_rank_tier >= {float(rank)} "
                f"AND game_mode IN ({modes}) AND lobby_type IN ({lobbies}) "
                f"{cursor_clause} ORDER BY match_id DESC LIMIT 1000"
            )
            result = self.request("explorer", {"sql": sql})
            payload = result.get("rows") if isinstance(result, dict) else None
        else:
            raise ValueError("discovery must be public or explorer")
        if not isinstance(payload, list) or any(
            not isinstance(e, dict) or type(e.get("match_id")) is not int or e["match_id"] <= 0
            for e in payload
        ):
            raise InvalidPayload("invalid_discovery_payload")
        entries = {e["match_id"]: e for e in payload}
        next_cursor = min(entries) if entries else None
        if cursor is not None and next_cursor is not None and next_cursor >= cursor:
            raise InvalidPayload("non_advancing_cursor")
        return DiscoveryPage(tuple(entries[k] for k in sorted(entries, reverse=True)), next_cursor)

    def fetch_match(self, match_id: int) -> Any:
        return self.request(f"matches/{match_id}")
