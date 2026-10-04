from dataclasses import replace

import pytest
from test_foundation_drafts import canonical
from test_foundation_history import historical

from lastpick.config import Config
from lastpick.data.dataset import decisions_for_match
from lastpick.data.schemas import Rejection
from lastpick.evaluation.splits import temporal_split
from lastpick.features.asof import build_feature_rows
from lastpick.features.selection import FeatureSpec


def test_equal_timestamps_and_match_sides_stay_together():
    match, _, _ = canonical(both_teams=True)
    original, _ = decisions_for_match(match, Config())
    decisions = [
        replace(d, match_id=i, start_time=1000 + (i // 2) * 1000)
        for i in range(10)
        for d in original
    ]
    split = temporal_split(decisions)
    ids = [{d.match_id for d in group} for group in (split.train, split.validation, split.test)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert max(d.start_time for d in split.train) < split.validation_cutoff
    assert max(d.start_time for d in split.validation) < split.test_cutoff
    with pytest.raises(Rejection, match="duplicate_decision"):
        temporal_split(decisions + [decisions[0]])


def test_model_feature_rows_obey_training_allowlist_freeze_and_selection():
    match, _, _ = canonical()
    decisions, _ = decisions_for_match(match, Config())
    decision = decisions[0]
    contexts = [
        replace(decision, match_id=i, start_time=time) for i, time in ((5, 6000), (4, 4000))
    ]
    matches = [historical(i, i * 1000) for i in (1, 2, 3)]
    spec = FeatureSpec(("hero_win_rate", "hero_log_games"))
    rows = build_feature_rows(contexts, matches, {1, 2}, frozen_as_of=3000, spec=spec)
    assert [r.match_id for r in rows] == [5, 4]
    assert all(r.source_matches == 1 and r.as_of == 3000 for r in rows)
    assert all(r.latest_available_time < r.as_of and len(r.values) == 2 for r in rows)
    # Changing evaluation labels and all post-match role evidence cannot change inputs.
    changed = [
        replace(d, outcome=not d.outcome, role=replace(d.role, evidence={"future_gold": 999999}))
        for d in contexts
    ]
    assert build_feature_rows(changed, matches, {1, 2}, frozen_as_of=3000, spec=spec) == rows
