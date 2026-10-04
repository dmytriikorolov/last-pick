from dataclasses import replace

from test_foundation_drafts import canonical

from lastpick.config import RoleSettings
from lastpick.roles.inference import FarmRoleBaseline, LegacyPos1Heuristic


def test_obvious_carry_with_confidence_and_evidence():
    match, _, _ = canonical()
    result = FarmRoleBaseline().predict(match, 0)
    assert result.hero_id == 25 and result.predicted_role == 1
    assert 0.65 <= result.confidence < 1
    assert result.evidence["strong_minutes"] == [10]
    assert result.method == "farm-v2" and result.evidence["confidence_calibrated"] is False


def test_ambiguous_case_abstains():
    match, _, _ = canonical()
    players = tuple(
        replace(p, lh_t=tuple([0] * 10 + [20]), gold_t=tuple([0] * 10 + [2200]))
        if p.side == 0 and p.lane_role == 1
        else p
        for p in match.players
    )
    result = FarmRoleBaseline().predict(replace(match, players=players), 0)
    assert result.hero_id is None and result.reason == "ambiguous_pos1"


def test_missing_telemetry_is_not_fabricated_by_position_est():
    match, _, _ = canonical()
    players = tuple(
        replace(p, lh_t=None, position_est=1) if p.hero_id == 25 else p for p in match.players
    )
    result = FarmRoleBaseline().predict(replace(match, players=players), 0)
    assert result.reason == "missing_telemetry" and result.confidence == 0


def test_unrelated_missing_lane_metadata_does_not_exclude_v2_carry():
    match, _, _ = canonical()
    players = tuple(
        replace(p, lane_role=None) if p.side == 0 and p.lane_role != 1 else p for p in match.players
    )
    match = replace(match, players=players)
    assert FarmRoleBaseline().predict(match, 0).hero_id == 25
    assert LegacyPos1Heuristic().predict(match, 0).reason == "missing_required_data"


def test_legacy_behavior_and_configurable_threshold():
    match, _, _ = canonical()
    result = LegacyPos1Heuristic().predict(match, 0)
    assert result.hero_id == 25 and result.confidence == 1
    strict = LegacyPos1Heuristic(RoleSettings(legacy_lh_gap=100))
    assert strict.predict(match, 0).reason == "ambiguous_pos1"


def test_multiple_timestamps_raise_confidence_and_conflicts_lower_it():
    match, _, _ = canonical()
    players = []
    for p in match.players:
        lh, gold = [0] * 16, [0] * 16
        for minute in (5, 10, 15):
            lh[minute] = 65 if p.hero_id == 25 else 15
            gold[minute] = 4000 if p.hero_id == 25 else 2000
        players.append(replace(p, lh_t=tuple(lh), gold_t=tuple(gold)))
    enriched = replace(match, players=tuple(players))
    base = FarmRoleBaseline().predict(match, 0)
    strong = FarmRoleBaseline().predict(enriched, 0)
    assert strong.confidence > base.confidence and strong.hero_id == 25
    conflict = []
    for p in enriched.players:
        if p.hero_id == 25:
            p = replace(
                p, lh_t=tuple([0] * 10 + [65] + [0] * 5), gold_t=tuple([0] * 10 + [4000] + [0] * 5)
            )
        conflict.append(p)
    weaker = FarmRoleBaseline().predict(replace(match, players=tuple(conflict)), 0)
    assert weaker.confidence < strong.confidence and weaker.predicted_role is None


def test_snapshot_uses_actual_times_when_available():
    match, _, _ = canonical()
    player = replace(match.players[0], times=(0, 600, 900), lh_t=(0, 60, None))
    assert player.snapshot("lh_t", 10) == 60
    assert player.snapshot("lh_t", 5) is None


def test_trilane_and_later_telemetry_can_identify_carry():
    match, _, _ = canonical()
    players = []
    for p in match.players:
        if p.side == 0:
            p = replace(
                p,
                lane_role=1 if p.slot in (0, 1, 4) else p.lane_role,
                lh_t=tuple([None] * 15 + [65 if p.hero_id == 25 else 15]),
                gold_t=tuple([None] * 15 + [4000 if p.hero_id == 25 else 2000]),
            )
        players.append(p)
    match = replace(match, players=tuple(players))
    result = FarmRoleBaseline().predict(match, 0)
    assert result.hero_id == 25 and result.evidence["strong_minutes"] == [15]
    assert LegacyPos1Heuristic().predict(match, 0).reason == "ambiguous_pos1"
