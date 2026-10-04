"""Preserve ISP recovery rules while retaining the complete event stream."""

from collections import Counter
from dataclasses import replace

from lastpick.data.schemas import Draft, DraftEvent, Match, Rejection
from lastpick.data.validation import validate_integrity


def reconstruct(match: Match) -> Draft:
    validate_integrity(match)
    if match.game_mode != 22:
        raise Rejection("unsupported_draft_visibility", "only ranked All Pick is implemented")
    events = sorted(match.events, key=lambda e: e.order)
    if len({e.order for e in events}) != len(events):
        raise Rejection("ambiguous_event_orders")
    if any(e.order < 0 or e.hero_id <= 0 or (e.is_pick and e.team not in (0, 1)) for e in events):
        raise Rejection("invalid_draft_event")
    # Source gaps imply missing events. Interleaved bans may occupy pick-order gaps.
    if {e.order for e in events} != set(range(len(events))):
        raise Rejection("incomplete_event_orders")
    final_teams = {p.hero_id: p.side for p in match.players}
    attempts = [e for e in events if e.is_pick]
    successful = [e for e in attempts if e.hero_id in final_teams]
    if len(successful) != 10 or {e.hero_id for e in successful} != set(final_teams):
        raise Rejection("ambiguous_final_picks")
    if any(e.team != final_teams[e.hero_id] for e in successful):
        raise Rejection("final_pick_team_mismatch")
    teams = [e.team for e in successful]
    for subset, counts, reason in (
        (teams, {0: 5, 1: 5}, "team_pick_counts_not_5_each"),
        (teams[:4], {0: 2, 1: 2}, "first_phase_pick_counts_not_2_each"),
        (teams[:8], {0: 4, 1: 4}, "known_pick_counts_not_4_each"),
        (teams[8:], {0: 1, 1: 1}, "final_pick_counts_not_1_each"),
    ):
        if Counter(subset) != Counter(counts):
            raise Rejection(reason)
    if any(e.hero_id in final_teams for e in events if not e.is_pick):
        raise Rejection("final_hero_also_banned")
    recovered: list[DraftEvent] = []
    seen = 0
    for event in events:
        if not event.is_pick:
            recovered.append(event)
            continue
        if seen == 10:
            raise Rejection("extra_after_final_pick")
        phase = 1 if seen < 4 else 2 if seen < 8 else 3
        is_final = event.hero_id in final_teams
        recovered.append(
            replace(event, successful=is_final, phase=phase, phase_source="final_pick_counts")
        )
        seen += int(is_final)
    return Draft(
        tuple(recovered), tuple(e for e in recovered if e.successful), match.event_order_kind
    )


def visible_picks(draft: Draft, candidate: DraftEvent) -> tuple[tuple[int, ...], tuple[int, ...]]:
    allies = tuple(
        e.hero_id for e in draft.picks if e.team == candidate.team and e.order < candidate.order
    )
    team_pick = len(allies) + 1
    enemy_count = 0 if team_pick <= 2 else 2 if team_pick <= 4 else 4
    enemies = tuple(e.hero_id for e in draft.picks if e.team != candidate.team)[:enemy_count]
    return allies, enemies
