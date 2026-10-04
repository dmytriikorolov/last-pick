"""The only v2 module that understands raw OpenDota match fields."""

import math
from typing import Any

from lastpick.data.schemas import DraftEvent, Match, Player, Purchase, Rejection


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def integer(value: Any) -> int | None:
    return value if type(value) is int else None


def timeline(value: Any) -> tuple[float | None, ...] | None:
    return tuple(number(v) for v in value) if isinstance(value, list) else None


def rank_bucket(rank: float | None) -> str | None:
    buckets = {
        1: "herald",
        2: "guardian",
        3: "crusader",
        4: "archon",
        5: "legend",
        6: "ancient",
        7: "divine",
        8: "immortal",
    }
    return buckets.get(int(rank // 10)) if rank is not None else None


class OpenDotaAdapter:
    name = "opendota"

    def identifier(self, payload: Any) -> int | None:
        return integer(payload.get("match_id")) if isinstance(payload, dict) else None

    def normalize(self, payload: Any, discovery: dict[str, Any] | None = None) -> Match:
        return normalize(payload, discovery)


def normalize(raw: Any, discovery: dict[str, Any] | None = None) -> Match:
    if not isinstance(raw, dict) or integer(raw.get("match_id")) is None:
        raise Rejection("invalid_source_payload", "match object and integer match_id required")
    if raw["match_id"] <= 0:
        raise Rejection("invalid_match_id")
    players_raw = raw.get("players")
    events_raw = raw.get("picks_bans")
    if not isinstance(players_raw, list) or not isinstance(events_raw, list):
        raise Rejection("missing_players_or_draft")
    players = []
    for p in players_raw:
        if not isinstance(p, dict):
            raise Rejection("invalid_player")
        slot, hero = integer(p.get("player_slot")), integer(p.get("hero_id"))
        if slot is None or hero is None:
            raise Rejection("invalid_player")
        purchases = p.get("purchase_log")
        if isinstance(purchases, list):
            if any(not isinstance(e, dict) or not isinstance(e.get("key"), str) for e in purchases):
                raise Rejection("invalid_purchase_log")
            parsed_purchases = tuple(Purchase(e["key"], number(e.get("time"))) for e in purchases)
        else:
            parsed_purchases = None
        upgrades = p.get("ability_upgrades_arr")
        players.append(
            Player(
                slot=slot,
                side=int(slot >= 128),
                hero_id=hero,
                lane=integer(p.get("lane")),
                lane_role=integer(p.get("lane_role")),
                position_est=integer(p.get("position_est")),
                times=timeline(p.get("times")),
                lh_t=timeline(p.get("lh_t")),
                gold_t=timeline(p.get("gold_t")),
                xp_t=timeline(p.get("xp_t")),
                kills=number(p.get("kills")),
                deaths=number(p.get("deaths")),
                assists=number(p.get("assists")),
                net_worth=number(p.get("net_worth")),
                gpm=number(p.get("gold_per_min")),
                xpm=number(p.get("xp_per_min")),
                purchases=parsed_purchases,
                final_items=tuple(
                    integer(p.get(f"{prefix}_{i}"))
                    for prefix, count in (("item", 6), ("backpack", 3))
                    for i in range(count)
                ),
                ability_upgrades=tuple(integer(v) for v in upgrades)
                if isinstance(upgrades, list)
                else None,
                leaver_status=integer(p.get("leaver_status")),
                abandons=integer(p.get("abandons")),
                rank_tier=number(p.get("rank_tier")),
            )
        )
    events = []
    for e in events_raw:
        if not isinstance(e, dict):
            raise Rejection("invalid_draft_event")
        order, hero, team = (
            integer(e.get("order")),
            integer(e.get("hero_id")),
            integer(e.get("team")),
        )
        if (
            order is None
            or order < 0
            or hero is None
            or hero <= 0
            or type(e.get("is_pick")) is not bool
        ):
            raise Rejection("invalid_draft_event")
        if team not in (0, 1) and (e["is_pick"] or e.get("team") is not None):
            raise Rejection("invalid_event_team")
        # No phase is invented for a ban: source ordering alone is insufficient in AP.
        events.append(DraftEvent(order, team, hero, e["is_pick"]))
    metadata = discovery or {}
    rank = number(raw.get("avg_rank_tier"))
    rank_source = "match" if rank is not None else None
    if rank is None:
        rank = number(metadata.get("avg_rank_tier"))
        rank_source = "discovery" if rank is not None else None
    coverage = integer(raw.get("num_rank_tier"))
    if coverage is None:
        coverage = integer(metadata.get("num_rank_tier"))
    if rank is None:
        ranks = [p.rank_tier for p in players if p.rank_tier is not None and p.rank_tier > 0]
        if ranks:
            rank, coverage, rank_source = sum(ranks) / len(ranks), len(ranks), "player_ranks"
    return Match(
        match_id=raw["match_id"],
        source="opendota",
        players=tuple(players),
        events=tuple(events),
        start_time=integer(raw.get("start_time")),
        patch=integer(raw.get("patch")),
        duration=number(raw.get("duration")),
        game_mode=integer(raw.get("game_mode")),
        lobby_type=integer(raw.get("lobby_type")),
        avg_rank_tier=rank,
        num_rank_tier=coverage,
        rank_bracket=rank_bucket(rank),
        rank_source=rank_source,
        region=integer(raw.get("region")),
        cluster=integer(raw.get("cluster")),
        radiant_win=raw.get("radiant_win") if type(raw.get("radiant_win")) is bool else None,
        event_order_kind="pick_sequence" if raw.get("game_mode") == 22 else "source_order",
    )
