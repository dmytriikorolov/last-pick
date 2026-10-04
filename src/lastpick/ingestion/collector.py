"""Sequential durable collector. A page remains pending until every entry is resolved."""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lastpick.config import Config
from lastpick.data.adapter import MatchAdapter
from lastpick.data.io import atomic_json
from lastpick.data.opendota import OpenDotaAdapter
from lastpick.data.schemas import Rejection
from lastpick.data.validation import validate_filters, validate_integrity
from lastpick.drafts.reconstruction import reconstruct
from lastpick.ingestion.opendota import FetchFailure, InvalidPayload
from lastpick.ingestion.source import MatchSource


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class CrawlState:
    source: str
    cursor: int | None = None
    page_cursor: int | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)
    requests: int = 0
    request_day: str = ""
    config_fingerprint: str = ""


class Collector:
    def __init__(
        self,
        root: Path,
        config: Config,
        source: MatchSource | None = None,
        adapter: MatchAdapter | None = None,
    ) -> None:
        self.config = config
        if adapter is None and config.source != "opendota":
            raise ValueError("an adapter is required for this source")
        self.adapter = adapter or OpenDotaAdapter()
        if self.adapter.name != config.source:
            raise ValueError("adapter/source mismatch")
        self.raw_dir = root / config.paths.raw
        self.directory = root / config.paths.ingestion
        self.state_path = self.directory / "state.json"
        if self.state_path.exists():
            # Corrupted state fails loudly; never silently reset budgets/cursors.
            self.state = CrawlState(**json.loads(self.state_path.read_text()))
        else:
            self.state = CrawlState(config.source)
        if self.state.source != config.source:
            raise ValueError("state belongs to another source")
        if type(self.state.requests) is not int or self.state.requests < 0:
            raise ValueError("invalid persisted request count")
        if any(type(e.get("match_id")) is not int or e["match_id"] <= 0
               for e in self.state.pending):
            raise ValueError("invalid persisted pending entries")
        fingerprint = json.dumps(
            {
                "source": config.source,
                "filters": asdict(config.filters),
                "discovery": config.ingestion.discovery,
            },
            sort_keys=True,
        )
        if self.state.config_fingerprint and self.state.config_fingerprint != fingerprint:
            raise ValueError("ingestion policy changed; use a new state directory")
        self.state.config_fingerprint = fingerprint
        self.source = source
        if source is not None and source.name != config.source:
            raise ValueError("provider/source mismatch")

    def save(self) -> None:
        atomic_json(self.state_path, asdict(self.state))

    def consume_request(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self.config.ingestion.daily_budget and today != self.state.request_day:
            self.state.requests = 0
        self.state.request_day = today
        if self.state.requests >= self.config.ingestion.request_budget:
            raise BudgetExhausted("request_budget_exhausted")
        self.state.requests += 1
        self.save()

    def cache_or_fetch(self, entry: dict[str, Any]) -> tuple[Any, bool]:
        assert self.source is not None
        match_id = entry["match_id"]
        path = self.raw_dir / f"{match_id}.json"
        metadata_path = self.directory / "metadata" / f"{match_id}.json"
        previous: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                saved_metadata = json.loads(metadata_path.read_text())
                previous = saved_metadata["discovery"]
                if saved_metadata.get("source", self.source.name) != self.source.name or not isinstance(previous, dict):
                    raise ValueError("invalid source metadata")
            except (OSError, ValueError, TypeError, KeyError):
                raise InvalidPayload("invalid_existing_metadata") from None
        merged = dict(previous, **{key: value for key, value in entry.items() if value is not None})
        atomic_json(metadata_path, {"source": self.source.name, "discovery": merged})
        if path.exists():
            try:
                cached = json.loads(path.read_text())
            except (OSError, ValueError):
                raise InvalidPayload("invalid_existing_cache") from None
            if self.adapter.identifier(cached) != match_id:
                raise InvalidPayload("cache_match_id_mismatch")
            return cached, False
        payload = self.source.fetch_match(match_id)
        if self.adapter.identifier(payload) != match_id:
            atomic_json(self.directory / "invalid_payloads" / f"{match_id}.json", payload)
            raise InvalidPayload("invalid_detail_payload")
        # Store validly identified raw responses even if dataset policy later rejects them.
        atomic_json(path, payload)
        return payload, True

    def run(self, max_pages: int | None = None) -> dict[str, int]:
        assert self.source is not None
        counts: Counter[str] = Counter()
        pages = 0
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        total = sum(1 for _ in self.raw_dir.glob("*.json"))
        if not self.state.pending and self.state.page_cursor is not None:
            self.state.cursor = self.state.page_cursor
            self.state.page_cursor = None
            self.save()
        while total < self.config.ingestion.target_matches:
            if not self.state.pending:
                if max_pages is not None and pages >= max_pages:
                    break
                try:
                    page = self.source.discover_matches(self.state.cursor)
                except (FetchFailure, InvalidPayload, BudgetExhausted) as error:
                    counts[str(error)] += 1
                    atomic_json(
                        self.directory / "discovery_error.json",
                        {"reason": str(error), "cursor": self.state.cursor},
                    )
                    break
                pages += 1
                if not page.entries:
                    break
                # Provider-independent validation; don't trust third-party adapters blindly.
                if any(
                    type(e.get("match_id")) is not int or e["match_id"] <= 0 for e in page.entries
                ):
                    raise InvalidPayload("invalid_discovery_payload")
                if page.next_cursor is None or (
                    self.state.cursor is not None and page.next_cursor >= self.state.cursor
                ):
                    raise InvalidPayload("non_advancing_cursor")
                unique = {e["match_id"]: e for e in page.entries}
                self.state.pending = [unique[k] for k in sorted(unique, reverse=True)]
                self.state.page_cursor = page.next_cursor
                self.save()
            while self.state.pending:
                entry = self.state.pending[0]
                match_id = entry["match_id"]
                outcome_path = self.directory / "outcomes" / f"{match_id}.json"
                if outcome_path.exists():
                    counts["duplicate"] += 1
                else:
                    try:
                        payload, downloaded = self.cache_or_fetch(entry)
                        counts["downloaded" if downloaded else "cache_reused"] += 1
                        total += int(downloaded)
                        metadata = json.loads((self.directory / "metadata" / f"{match_id}.json").read_text())
                        match = self.adapter.normalize(payload, metadata["discovery"])
                        validate_integrity(match)
                        validate_filters(match, self.config.filters)
                        reconstruct(match)
                    except (FetchFailure, BudgetExhausted) as error:
                        # Keep failed entries pending. Restart retries before advancing cursor.
                        counts["fetch_failure"] += 1
                        atomic_json(
                            self.directory / "last_failure.json",
                            {"match_id": match_id, "kind": "fetch_failure", "reason": str(error)},
                        )
                        return dict(counts)
                    except (Rejection, InvalidPayload) as error:
                        reason = error.reason if isinstance(error, Rejection) else str(error)
                        kind = (
                            "validation_rejection"
                            if isinstance(error, Rejection)
                            else "invalid_payload"
                        )
                        atomic_json(
                            outcome_path, {"match_id": match_id, "kind": kind, "reason": reason}
                        )
                        counts[kind] += 1
                    else:
                        atomic_json(
                            outcome_path,
                            {"match_id": match_id, "kind": "accepted", "source": match.source},
                        )
                        counts["accepted"] += 1
                self.state.pending.pop(0)
                self.save()
                if total >= self.config.ingestion.target_matches:
                    # Remaining page is saved for a subsequent run with a larger target.
                    return dict(counts)
            self.state.cursor = self.state.page_cursor
            self.state.page_cursor = None
            self.save()
        return dict(counts)
