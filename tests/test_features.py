import numpy as np
import torch

from scripts.carry_historical_features import (add_match, empty_stats, feature_vector,
                                              pair_values, training_features)
from scripts.carry_model import static_hero_features
from scripts.hero_lookup import hero_id_to_name, hero_name_to_id, names_by_id


# makes one small carry decision for feature tests
def decision(time):
    return {"match_id": str(time), "start_time": str(time), "candidate_hero_id": "1",
            "ally_1": "2", "ally_2": "3", "ally_3": "", "ally_4": "",
            "enemy_1": "6", "enemy_2": "7", "enemy_3": "", "enemy_4": ""}


# makes one earlier match with known result and farm values
def historical_match(time, win=1):
    return {"match_id": str(time), "start_time": time, "radiant": [1, 2, 3, 4, 5],
            "dire": [6, 7, 8, 9, 10], "radiant_win": win, "duration": 1500,
            "farm": {1: [500, 4000, 60]}}


# checks hero names and the size of model inputs
def test_hero_metadata_and_feature_shapes():
    for hero, name in names_by_id.items():
        assert hero_id_to_name(hero) == name
        assert hero_name_to_id(name) == hero
    static = static_hero_features(sorted(names_by_id))
    assert tuple(static.shape) == (len(names_by_id) + 1, 14)
    assert torch.isfinite(static).all()
    history = feature_vector(empty_stats(), 1, [2, 3], [6, 7])
    assert len(history) == 20 and np.isfinite(history).all()
    assert history[0] == 0.5
    assert history[-8:] == [0] * 8


# checks ally synergy, enemy matchups and weak evidence
def test_pair_direction_and_small_sample_defaults():
    stats = empty_stats()
    add_match(stats, historical_match(1))
    assert pair_values(stats, 1, [2], True) == pair_values(stats, 2, [1], True)
    effect = pair_values(stats, 1, [2], True)[0]
    assert 0 < effect < 0.01
    assert pair_values(stats, 1, [6], False)[0] > 0
    assert pair_values(stats, 6, [1], False)[0] < 0
    assert pair_values(stats, 1, [99], False) == [0] * 4


# checks that a match cannot use its own result as history
def test_history_uses_only_earlier_matches():
    rows = [decision(1), decision(2), decision(2), decision(3)]
    matches = [historical_match(i) for i in (1, 2, 3)]
    values, _, blocks = training_features(rows, matches, blocks=3)
    changed = [historical_match(1), historical_match(2, 0), historical_match(3, 0)]
    other, _, _ = training_features(rows, changed, blocks=3)
    # changing current and future results cannot change earlier inputs
    assert np.array_equal(values[:3], other[:3])
    assert np.array_equal(values[1], values[2])
    for block in blocks:
        assert block["latest_source_time"] < block["start_time"]
