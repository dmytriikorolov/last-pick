from unittest.mock import patch

import pytest
import torch
from test_draft_parsing import make_match
from test_recommender import allies, enemies, hero_ids, model, pool

from lastpick.features.carry import FEATURE_FIELDS
from lastpick.models import training
from lastpick.recommender.carry import rank_candidates
from scripts.build_dataset import build_match_examples


def test_questionable_example_has_half_gradient_weight():
    logits = torch.zeros(2, requires_grad=True)
    loss = training.weighted_loss(logits, torch.ones(2), torch.tensor([1.0, 0.5]))
    loss.backward()
    assert logits.grad[1] == logits.grad[0] * 0.5
    with pytest.raises(ValueError):
        training.weighted_loss(logits, torch.ones(2), torch.zeros(2))


def test_actual_training_loop_receives_saved_quality_weights():
    raw, _, _ = make_match(8)
    rows, _, _ = build_match_examples(raw)
    clean = dict(rows[0], win=1, training_weight=1.0)
    questionable = dict(rows[0], win=0, training_weight=0.5)
    mapping = {hero: i for i, hero in enumerate(hero_ids, 1)}
    tensors = training.tensors([clean, questionable], mapping)
    history = torch.zeros(2, len(FEATURE_FIELDS))
    captured = []
    original = training.weighted_loss

    def record_loss(logits, labels, weights):
        captured.extend(weights.tolist())
        return original(logits, labels, weights)

    with (
        patch.object(training, "MAX_EPOCHS", 1),
        patch.object(training, "weighted_loss", side_effect=record_loss),
    ):
        training.train_model(tensors, tensors, history, history, hero_ids, None)
    assert sorted(captured) == [0.5, 1.0]


def test_bans_are_enforced_inside_recommender_api():
    first = rank_candidates(model, hero_ids, pool, allies, enemies)
    banned = [hero for hero, _ in first]
    ranking = rank_candidates(model, hero_ids, pool, allies, enemies, banned=banned)
    assert not set(banned + allies + enemies) & {hero for hero, _ in ranking}
    assert {hero for hero, _ in ranking} <= set(pool)


def test_historical_input_width_is_enforced():
    from lastpick.models.carry import CarryModel

    network = CarryModel(hero_ids, historical_dim=20)
    with pytest.raises(ValueError, match="wrong width"):
        network(
            torch.tensor([1]),
            torch.ones(1, 4, dtype=torch.long),
            torch.ones(1, 4, dtype=torch.long),
            torch.zeros(1, 19),
        )


def test_training_defaults_protect_isp_artifacts():
    assert training.MODEL_PATH.name == "foundation_carry_recommender.pt"
    assert training.RESULTS_PATH.name == "foundation_carry_training_results.json"
