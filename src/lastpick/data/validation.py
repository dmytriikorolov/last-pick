"""Canonical integrity and selection policies, separate from download failures."""

import math

from lastpick.config import Filters
from lastpick.data.schemas import Match, Rejection


def validate_integrity(match: Match) -> None:
    if type(match.match_id) is not int or match.match_id <= 0:
        raise Rejection("invalid_match_id")
    if type(match.radiant_win) is not bool:
        raise Rejection("missing_or_invalid_outcome")
    if type(match.start_time) is not int or match.start_time < 0:
        raise Rejection("missing_start_time")
    if match.duration is not None and (not math.isfinite(match.duration) or match.duration <= 0):
        raise Rejection("invalid_duration")
    slots = set(range(5)) | set(range(128, 133))
    if len(match.players) != 10 or {p.slot for p in match.players} != slots:
        raise Rejection("invalid_final_players")
    if len({p.hero_id for p in match.players}) != 10:
        raise Rejection("duplicate_final_hero")
    if any(
        type(p.slot) is not int
        or type(p.side) is not int
        or type(p.hero_id) is not int
        or p.side != int(p.slot >= 128)
        or p.hero_id <= 0
        for p in match.players
    ):
        raise Rejection("invalid_player_side_or_hero")


def validate_filters(match: Match, filters: Filters) -> None:
    if match.game_mode not in filters.game_modes:
        raise Rejection("unsupported_game_mode")
    if match.lobby_type not in filters.lobby_types:
        raise Rejection("unsupported_lobby_type")
    if match.duration is None or match.duration < filters.min_duration:
        raise Rejection("missing_or_short_duration")
    if filters.min_rank is not None:
        if match.avg_rank_tier is None and filters.require_rank:
            raise Rejection("missing_rank")
        if match.avg_rank_tier is not None and match.avg_rank_tier < filters.min_rank:
            raise Rejection("below_min_rank")
    if filters.min_ranked_players is not None:
        if match.num_rank_tier is None and filters.require_rank:
            raise Rejection("missing_rank_coverage")
        if match.num_rank_tier is not None and match.num_rank_tier < filters.min_ranked_players:
            raise Rejection("insufficient_rank_coverage")
    if filters.patches and match.patch not in filters.patches:
        raise Rejection("patch_not_allowed")
    if filters.min_patch is not None and (match.patch is None or match.patch < filters.min_patch):
        raise Rejection("missing_or_old_patch")
