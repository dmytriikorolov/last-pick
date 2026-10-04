"""JSON-friendly domain records; None means unavailable, never measured zero."""

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "2.0"


class Rejection(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class Purchase:
    item: str
    time: float | None


@dataclass(frozen=True)
class Player:
    slot: int
    side: int
    hero_id: int
    lane: int | None = None
    lane_role: int | None = None
    position_est: int | None = None
    times: tuple[float | None, ...] | None = None
    lh_t: tuple[float | None, ...] | None = None
    gold_t: tuple[float | None, ...] | None = None
    xp_t: tuple[float | None, ...] | None = None
    kills: float | None = None
    deaths: float | None = None
    assists: float | None = None
    net_worth: float | None = None
    gpm: float | None = None
    xpm: float | None = None
    purchases: tuple[Purchase, ...] | None = None
    final_items: tuple[int | None, ...] = ()
    ability_upgrades: tuple[int | None, ...] | None = None
    leaver_status: int | None = None
    abandons: int | None = None
    rank_tier: float | None = None

    def snapshot(self, field_name: str, minute: int) -> float | None:
        values = getattr(self, field_name)
        if values is None:
            return None
        # Parsed `times` can have non-minute intervals; do not assume index = minute.
        if self.times is not None:
            target = minute * 60
            if target not in self.times:
                return None
            index = self.times.index(target)
        else:
            index = minute
        return values[index] if len(values) > index else None


@dataclass(frozen=True)
class DraftEvent:
    order: int
    team: int | None
    hero_id: int
    is_pick: bool
    successful: bool | None = None
    phase: int | None = None
    phase_source: str | None = None


@dataclass(frozen=True)
class Match:
    match_id: int
    source: str
    players: tuple[Player, ...]
    events: tuple[DraftEvent, ...]
    start_time: int | None = None
    patch: int | None = None
    duration: float | None = None
    game_mode: int | None = None
    lobby_type: int | None = None
    avg_rank_tier: float | None = None
    num_rank_tier: int | None = None
    rank_bracket: str | None = None
    rank_source: str | None = None
    region: int | None = None
    cluster: int | None = None
    radiant_win: bool | None = None
    # AP picks_bans order can be pick sequence followed by an unordered ban list.
    event_order_kind: str = "source_order"
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class Draft:
    events: tuple[DraftEvent, ...]
    picks: tuple[DraftEvent, ...]
    order_kind: str


@dataclass(frozen=True)
class RolePrediction:
    player_slot: int | None
    hero_id: int | None
    predicted_role: int | None
    confidence: float
    scores: dict[int, float]
    evidence: dict[str, Any]
    method: str
    version: str
    reason: str | None = None
    position_probabilities: dict[int, float] | None = None


@dataclass(frozen=True)
class DraftDecision:
    match_id: int
    source: str
    side: int
    phase: int
    team_pick_number: int
    event_order: int
    visible_allies: tuple[int, ...]
    visible_enemies: tuple[int, ...]
    visible_bans: tuple[int, ...]
    unlocated_bans: tuple[int, ...]
    chosen_hero: int
    role: RolePrediction
    patch: int | None
    rank_bracket: str | None
    avg_rank_tier: float | None
    region: int | None
    start_time: int | None
    outcome: bool
    legal_candidates: tuple[int, ...] | None = None
    candidate_universe: tuple[int, ...] | None = None
    schema_version: str = SCHEMA_VERSION


def record(value: Match | DraftDecision | RolePrediction) -> dict[str, Any]:
    return asdict(value)


def match_from_record(value: dict[str, Any]) -> Match:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise Rejection("unsupported_schema")
    players = []
    for row in value["players"]:
        row = dict(row)
        if row.get("purchases") is not None:
            row["purchases"] = tuple(Purchase(**p) for p in row["purchases"])
        for name in ("times", "lh_t", "gold_t", "xp_t", "final_items", "ability_upgrades"):
            if row.get(name) is not None:
                row[name] = tuple(row[name])
        players.append(Player(**row))
    return Match(
        **dict(
            value, players=tuple(players), events=tuple(DraftEvent(**e) for e in value["events"])
        )
    )


def decision_from_record(value: dict[str, Any]) -> DraftDecision:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise Rejection("unsupported_schema")
    data = dict(value)
    role = dict(data["role"])
    role["scores"] = {int(slot): score for slot, score in role["scores"].items()}
    if role.get("position_probabilities") is not None:
        role["position_probabilities"] = {
            int(position): probability
            for position, probability in role["position_probabilities"].items()
        }
    data["role"] = RolePrediction(**role)
    for key in (
        "visible_allies",
        "visible_enemies",
        "visible_bans",
        "unlocated_bans",
        "legal_candidates",
        "candidate_universe",
    ):
        if data.get(key) is not None:
            data[key] = tuple(data[key])
    return DraftDecision(**data)
