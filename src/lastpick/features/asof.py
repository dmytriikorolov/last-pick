"""One statistics engine for v2, with completed-match availability cutoffs.

Aggregate exports and model features must come from an explicit snapshot.
Current match telemetry is used to construct labels, never current draft inputs.
"""

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lastpick.config import FeatureSettings
from lastpick.data.schemas import DraftDecision, Match, Rejection, match_from_record
from lastpick.data.validation import validate_integrity
from lastpick.features.carry import add_match, empty_stats, feature_vector
from lastpick.features.selection import FeatureSpec


@dataclass
class Measurement:
    total: float = 0
    count: int = 0

    def add(self, value: float | None) -> None:
        if value is not None:
            self.total += value
            self.count += 1

    @property
    def average(self) -> float | None:
        return self.total / self.count if self.count else None

    def summary(self, prior_games: float) -> dict[str, float | int | None]:
        return {
            "average": self.average,
            "support": self.count,
            "reliability": self.count / (self.count + prior_games),
        }


@dataclass
class Extras:
    farm: dict[str, dict[str, Measurement]] = field(default_factory=dict)
    lanes: dict[str, dict[str, Measurement]] = field(default_factory=dict)
    items: dict[str, Measurement] = field(default_factory=dict)
    item_log_support: Counter[str] = field(default_factory=Counter)

    def add(self, match: Match) -> None:
        lane_groups: dict[tuple[int, int], list[Any]] = {}
        for player in match.players:
            profile = self.farm.setdefault(str(player.hero_id), {})
            values: dict[str, float | None] = {
                "gpm": player.gpm,
                "xpm": player.xpm,
                "net_worth": player.net_worth,
            }
            for minute in (5, 10, 15, 20, 30):
                for name in ("lh_t", "gold_t", "xp_t"):
                    value = player.snapshot(name, minute)
                    if match.duration is not None and minute * 60 > match.duration:
                        value = None
                    values[f"{name.removesuffix('_t')}_{minute}"] = value
            for name, value in values.items():
                profile.setdefault(name, Measurement()).add(value)
            if player.lane_role in (1, 3):
                lane_groups.setdefault((player.side, player.lane_role), []).append(player)
            if player.purchases is not None:
                context = f"{player.hero_id}:{match.patch}"
                self.item_log_support[context] += 1
                first: dict[str, float] = {}
                for purchase in player.purchases:
                    if purchase.time is not None:
                        first[purchase.item] = min(
                            purchase.time, first.get(purchase.item, purchase.time)
                        )
                for item, time in first.items():
                    self.items.setdefault(f"{context}:{item}", Measurement()).add(time)
        for (side, lane), players in lane_groups.items():
            if len(players) != 2:
                continue
            a, b = sorted(p.hero_id for p in players)
            profile = self.lanes.setdefault(f"{a}:{b}:{lane}", {})
            profile.setdefault("win", Measurement()).add(
                float(bool(match.radiant_win) == (side == 0))
            )
            for name in ("lh_t", "gold_t", "xp_t"):
                lane_values = [p.snapshot(name, 10) for p in players]
                total = (
                    sum(v for v in lane_values if v is not None)
                    if all(v is not None for v in lane_values)
                    else None
                )
                profile.setdefault(f"combined_{name}_10", Measurement()).add(total)

    def summary(self, prior_games: float) -> dict[str, Any]:
        return {
            "farm": {
                hero: {name: stat.summary(prior_games) for name, stat in fields.items()}
                for hero, fields in self.farm.items()
            },
            "lanes": {
                pair: {name: stat.summary(prior_games) for name, stat in fields.items()}
                for pair, fields in self.lanes.items()
            },
            "items": {
                key: dict(
                    stat.summary(prior_games),
                    purchase_log_support=self.item_log_support[key.rsplit(":", 1)[0]],
                    purchase_rate=stat.count / self.item_log_support[key.rsplit(":", 1)[0]],
                )
                for key, stat in self.items.items()
            },
        }


def training_record(match: Match) -> dict[str, Any]:
    return {
        "match_id": match.match_id,
        "start_time": match.start_time,
        "radiant_win": int(bool(match.radiant_win)),
        "duration": match.duration,
        "radiant": [p.hero_id for p in match.players if p.side == 0],
        "dire": [p.hero_id for p in match.players if p.side == 1],
        "farm": {
            p.hero_id: [p.gpm, p.snapshot("gold_t", 10), p.snapshot("lh_t", 10)]
            for p in match.players
        },
    }


@dataclass(frozen=True)
class Snapshot:
    as_of: int
    latest_available_time: float | None
    stats: dict[str, Any]
    extras: dict[str, Any]
    source_match_ids: tuple[int, ...]

    def model_features(self, candidate: int, allies: list[int], enemies: list[int]) -> list[float]:
        """Explicit compatibility selection: v1's 20 numeric inputs with support.

        Missing numeric means use the v1 neutral default here only; rich extras
        retain None and independent sample counts for future model selection.
        """
        return feature_vector(self.stats, candidate, allies, enemies)


class AsOfHistory:
    def __init__(
        self,
        matches: list[Match],
        allowed_match_ids: set[int] | None = None,
        settings: FeatureSettings | None = None,
    ) -> None:
        seen: set[int] = set()
        self.missing_availability = 0
        self.matches: list[tuple[float, Match]] = []
        for match in matches:
            if allowed_match_ids is not None and match.match_id not in allowed_match_ids:
                continue
            validate_integrity(match)
            if match.match_id in seen:
                raise Rejection("duplicate_historical_match")
            seen.add(match.match_id)
            if match.start_time is None or match.duration is None or match.duration <= 0:
                self.missing_availability += 1
                continue
            self.matches.append((match.start_time + match.duration, match))
        self.matches.sort(key=lambda pair: (pair[0], pair[1].match_id))
        self.settings = settings or FeatureSettings()
        self.stats = empty_stats(
            self.settings.prior_games, self.settings.early_seconds, self.settings.late_seconds
        )
        self.extras = Extras()
        self.position = 0
        self.last_as_of: int | None = None
        self.ids: list[int] = []

    def advance(self, as_of: int) -> None:
        if self.last_as_of is not None and as_of < self.last_as_of:
            raise ValueError(
                "as-of queries must be nondecreasing; create a new history for earlier queries"
            )
        while self.position < len(self.matches) and self.matches[self.position][0] < as_of:
            _, match = self.matches[self.position]
            add_match(self.stats, training_record(match))
            self.extras.add(match)
            self.ids.append(match.match_id)
            self.position += 1
        self.last_as_of = as_of

    def snapshot(self, as_of: int) -> Snapshot:
        self.advance(as_of)
        latest = self.matches[self.position - 1][0] if self.position else None
        # A returned snapshot must never change when a future query updates the engine.
        return Snapshot(
            as_of,
            latest,
            deepcopy(self.stats),
            self.extras.summary(self.settings.prior_games),
            tuple(self.ids),
        )


def load_canonical(directory: Path) -> list[Match]:
    index = json.loads((directory / "index.json").read_text())
    return [
        match_from_record(json.loads((directory / filename).read_text()))
        for filename in index["files"]
    ]


@dataclass(frozen=True)
class FeatureRow:
    match_id: int
    side: int
    as_of: int
    source_matches: int
    latest_available_time: float | None
    values: tuple[float, ...]


def build_feature_rows(
    decisions: list[DraftDecision],
    matches: list[Match],
    training_match_ids: set[int],
    frozen_as_of: int | None = None,
    spec: FeatureSpec | None = None,
    settings: FeatureSettings | None = None,
) -> list[FeatureRow]:
    """Training queries use individual decision time; validation/test freeze history.

    The training source allowlist is required even for frozen evaluation. Numeric
    selection reads only draft heroes and past statistics, never role evidence,
    current outcomes, current telemetry, or unlocated bans.
    """
    if any(d.start_time is None for d in decisions):
        raise Rejection("missing_decision_time")
    selected = spec or FeatureSpec()
    history = AsOfHistory(matches, training_match_ids, settings)
    indexed = sorted(
        enumerate(decisions),
        key=lambda pair: (pair[1].start_time or 0, pair[1].match_id, pair[1].side),
    )
    output: dict[int, FeatureRow] = {}
    for index, decision in indexed:
        assert decision.start_time is not None
        cutoff = (
            min(decision.start_time, frozen_as_of)
            if frozen_as_of is not None
            else decision.start_time
        )
        history.advance(cutoff)
        values = selected.values(
            history.stats,
            decision.chosen_hero,
            list(decision.visible_allies),
            list(decision.visible_enemies),
        )
        latest = history.matches[history.position - 1][0] if history.position else None
        output[index] = FeatureRow(
            decision.match_id, decision.side, cutoff, history.position, latest, values
        )
    return [output[i] for i in range(len(decisions))]
