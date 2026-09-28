import math

from scripts.carry_historical_features import FEATURE_FIELDS
from scripts.hero_lookup import hero_id_to_name, names_by_id
from scripts.recommend_carry import DEFAULT_MODEL, load_model, rank_candidates, score_candidates


# loads the final model once for these tests
model, hero_ids, pool = load_model(DEFAULT_MODEL)
allies = [2, 5, 25, 86]
enemies = [14, 26, 74, 99]


# checks the saved model and its five valid recommendations
def test_final_model_and_top_five():
    assert DEFAULT_MODEL.name == "final_carry_recommender.pt"
    assert model.use_static and model.historical_dim == 20
    assert model.historical_bundle["fields"] == FEATURE_FIELDS
    assert set(hero_ids) == set(names_by_id)
    ranking = rank_candidates(model, hero_ids, pool, allies, enemies)
    ids = [hero for hero, _ in ranking]
    assert len(ids) == len(set(ids)) == 5
    assert not set(ids) & set(allies + enemies)
    assert set(ids) <= set(pool)
    for hero, score in ranking:
        assert hero_id_to_name(hero)
        assert math.isfinite(score) and 0 <= score <= 1


# checks that banned or unavailable heroes are not suggested
def test_unavailable_heroes_are_excluded():
    banned = [hero for hero, _ in rank_candidates(model, hero_ids, pool, allies, enemies)]
    available = [hero for hero in pool if hero not in banned]
    ranking = rank_candidates(model, hero_ids, available, allies, enemies)
    assert len(ranking) == 5
    assert not set(banned) & {hero for hero, _ in ranking}
    assert rank_candidates(model, hero_ids, [], allies, enemies) == []


# checks that changing the draft changes carry scores and order
def test_draft_changes_scores_and_ranking():
    other_allies = [3, 7, 9, 13]
    other_enemies = [15, 16, 17, 18]
    picked = set(allies + enemies + other_allies + other_enemies)
    candidates = [hero for hero in pool if hero not in picked]
    first = score_candidates(model, hero_ids, candidates, allies, enemies)
    second = score_candidates(model, hero_ids, candidates, other_allies, other_enemies)
    assert [hero for hero, _ in first] != [hero for hero, _ in second]
    before, after = dict(first), dict(second)
    assert any(abs(before[hero] - after[hero]) > 1e-6 for hero in candidates)
