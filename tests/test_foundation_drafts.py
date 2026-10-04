from dataclasses import replace

import pytest
from test_draft_parsing import add_extra_attempt, make_match

from lastpick.config import Config
from lastpick.data.dataset import decisions_for_match
from lastpick.data.opendota import normalize
from lastpick.data.schemas import DraftEvent, Rejection, match_from_record, record
from lastpick.drafts.reconstruction import reconstruct, visible_picks


def canonical(order=8, both_teams=False):
    raw, teams, heroes = make_match(order, both_teams=both_teams)
    raw.update(duration=2400, avg_rank_tier=80, num_rank_tier=10, region=3)
    return normalize(raw), teams, heroes


def test_complete_events_and_round_trip_preserve_missing_values():
    match, _, heroes = canonical()
    events = match.events + (DraftEvent(10, 0, 99, False),)
    match = replace(match, events=events)
    draft = reconstruct(match)
    assert [e.hero_id for e in draft.picks] == heroes
    assert draft.events[-1].is_pick is False and draft.events[-1].phase is None
    loaded = match_from_record(record(match))
    assert loaded == match
    assert loaded.players[0].gpm is None and loaded.players[0].purchases is None


@pytest.mark.parametrize("order", range(4, 10))
def test_current_phase_enemy_choices_hidden_for_both_sides(order):
    match, teams, heroes = canonical(order)
    draft = reconstruct(match)
    allies, enemies = visible_picks(draft, draft.picks[order])
    assert allies == tuple(heroes[i] for i in range(order) if teams[i] == teams[order])
    assert (
        enemies
        == tuple(h for h, t in zip(heroes, teams) if t != teams[order])[: 2 if order < 8 else 4]
    )
    hidden = {h for h, t in zip(heroes, teams) if t != teams[order]} - set(enemies)
    assert hidden.isdisjoint(allies + enemies)


def test_failed_attempt_recovery_and_phase_exclusion():
    raw, _, heroes = make_match(8)
    raw.update(duration=2400, avg_rank_tier=80, num_rank_tier=10)
    add_extra_attempt(raw, 4, 99, 0)
    match = normalize(raw)
    draft = reconstruct(match)
    assert [e.hero_id for e in draft.picks] == heroes
    attempt = next(e for e in draft.events if e.hero_id == 99)
    assert attempt.successful is False and attempt.phase == 2
    assert len(decisions_for_match(match, Config())[0]) == 1
    add_extra_attempt(raw, 9, 100, 1)
    decisions, reasons = decisions_for_match(normalize(raw), Config())
    assert not decisions and reasons["candidate_phase_extra_attempt"] == 1


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("repeated_final", "ambiguous_final_picks"),
        ("same_order", "ambiguous_event_orders"),
        ("wrong_team", "final_pick_team_mismatch"),
        ("duplicate_hero", "duplicate_final_hero"),
        ("missing_event", "incomplete_event_orders"),
    ],
)
def test_ambiguous_streams_rejected(mutation, reason):
    match, _, _ = canonical()
    if mutation == "repeated_final":
        match = replace(match, events=match.events + (replace(match.events[0], order=10),))
    elif mutation == "same_order":
        match = replace(match, events=(replace(match.events[0], order=1), *match.events[1:]))
    elif mutation == "wrong_team":
        match = replace(match, events=(replace(match.events[0], team=1), *match.events[1:]))
    elif mutation == "duplicate_hero":
        match = replace(
            match,
            players=(
                replace(match.players[0], hero_id=match.players[1].hero_id),
                *match.players[1:],
            ),
        )
    else:
        match = replace(match, events=match.events[1:])
    with pytest.raises(Rejection, match=reason):
        reconstruct(match)


def test_bans_with_proven_phase_visible_unknown_timing_retained():
    match, _, _ = canonical()
    bans = (
        DraftEvent(10, 0, 99, False, phase=0, phase_source="fixture"),
        DraftEvent(11, 1, 100, False),
    )
    decisions, _ = decisions_for_match(replace(match, events=match.events + bans), Config())
    decision = decisions[0]
    assert decision.visible_bans == (99,) and decision.unlocated_bans == (100,)
    assert decision.legal_candidates is None


def test_banned_final_hero_rejected():
    match, _, _ = canonical()
    match = replace(
        match, events=match.events + (DraftEvent(10, 0, match.players[0].hero_id, False),)
    )
    with pytest.raises(Rejection, match="final_hero_also_banned"):
        reconstruct(match)
