from dataclasses import replace

import pytest
from test_foundation_drafts import canonical

from lastpick.data.schemas import Purchase, Rejection
from lastpick.features.asof import AsOfHistory


def historical(match_id, start_time, winner=True):
    match, _, _ = canonical()
    return replace(
        match, match_id=match_id, start_time=start_time, duration=1000, radiant_win=winner
    )


def test_history_requires_completed_match_strictly_before_cutoff():
    matches = [historical(1, 1000), historical(2, 2000), historical(3, 2000)]
    history = AsOfHistory(matches)
    assert history.snapshot(2000).source_match_ids == ()
    first = history.snapshot(2001)
    assert first.source_match_ids == (1,)
    assert first.latest_available_time < first.as_of
    assert history.snapshot(3000).source_match_ids == (1,)
    assert history.snapshot(3001).source_match_ids == (1, 2, 3)
    assert first.source_match_ids == (1,) and first.stats["matches"] == 1
    with pytest.raises(ValueError, match="nondecreasing"):
        history.snapshot(2000)


def test_future_result_and_same_time_do_not_change_features():
    matches = [historical(1, 1000), historical(2, 2000), historical(3, 2000)]
    changed = [
        matches[0],
        replace(matches[1], radiant_win=False),
        replace(matches[2], radiant_win=False),
    ]
    left = AsOfHistory(matches).snapshot(3000)
    right = AsOfHistory(changed).snapshot(3000)
    assert left.model_features(25, [2, 5], [3, 7]) == right.model_features(25, [2, 5], [3, 7])
    assert left.source_match_ids == (1,)


def test_training_allowlist_prevents_validation_and_test_history():
    matches = [historical(i, i * 1000) for i in (1, 2, 3)]
    history = AsOfHistory(matches, allowed_match_ids={1})
    assert history.snapshot(10000).source_match_ids == (1,)


def test_missing_values_and_item_log_support_are_explicit():
    match = historical(1, 1000)
    player = next(p for p in match.players if p.hero_id == 25)
    player = replace(player, purchases=(Purchase("boots", -10), Purchase("boots", 30)))
    players = tuple(player if p.hero_id == 25 else p for p in match.players)
    first = replace(match, players=players)
    second_players = tuple(replace(p, purchases=()) if p.hero_id == 25 else p for p in players)
    second = replace(match, match_id=2, players=second_players)
    snapshot = AsOfHistory([first, second]).snapshot(3000)
    farm = snapshot.extras["farm"]["25"]
    assert farm["gpm"] == {"average": None, "support": 0, "reliability": 0}
    assert farm["lh_10"]["average"] == 65 and farm["lh_10"]["support"] == 2
    item = snapshot.extras["items"]["25:60:boots"]
    assert item["average"] == -10 and item["purchase_log_support"] == 2
    assert item["purchase_rate"] == 0.5 and 0 < item["reliability"] < 0.01
    lane = next(iter(snapshot.extras["lanes"].values()))
    assert lane["combined_xp_t_10"]["average"] is None


def test_duplicate_history_rejected_and_unknown_availability_reported():
    match = historical(1, 1000)
    with pytest.raises(Rejection, match="duplicate_historical_match"):
        AsOfHistory([match, match])
    history = AsOfHistory([replace(match, duration=None)])
    assert history.missing_availability == 1 and history.snapshot(10000).stats["matches"] == 0


def test_pair_directionality_and_support_use_same_canonical_engine():
    match = historical(1, 1000)
    snapshot = AsOfHistory([match]).snapshot(3000)
    radiant = [p.hero_id for p in match.players if p.side == 0]
    dire = [p.hero_id for p in match.players if p.side == 1]
    a, b = radiant[0], dire[0]
    assert snapshot.stats["matchups"][f"{a}:{b}"] == [1, 1]
    assert snapshot.stats["matchups"][f"{b}:{a}"] == [1, 0]
    assert 0 < snapshot.model_features(a, radiant[1:3], dire[:2])[-1] < 0.01
