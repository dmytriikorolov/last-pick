#!/usr/bin/env python3
# loads the carry model and ranks available heroes for a visible draft

import torch

from lastpick.config import project_root
from lastpick.features.carry import FEATURE_FIELDS, feature_vector
from lastpick.legacy.hero_lookup import hero_name_to_id
from lastpick.models.carry import CarryModel, encode_draft

DEFAULT_MODEL = project_root() / "models/final_carry_recommender.pt"


# converts comma separated hero names or ids into a list of ids
def parse_heroes(text):
    heroes = []
    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        hero = int(value) if value.isdecimal() else hero_name_to_id(value)
        heroes.append(hero)
    return heroes


# loads the saved model and its supported carry heroes
def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    history = checkpoint.get("historical_bundle")
    # recreates the model with the settings used during training
    model = CarryModel(
        checkpoint["hero_ids"],
        use_static=checkpoint["use_static"],
        embedding_dim=checkpoint.get("embedding_dim", 16),
        historical_dim=len(history["fields"]) if history else 0,
        hidden_sizes=checkpoint.get("hidden_sizes", (64, 32)),
    )
    model.load_state_dict(checkpoint["model_state"])
    # switches to prediction mode
    model.eval()
    model.historical_bundle = history
    return model, checkpoint["hero_ids"], checkpoint["candidate_pool"]


# scores available carry heroes and sorts them from highest to lowest
def score_candidates(model, hero_ids, candidates, allies, enemies):
    # index zero is reserved for empty picks
    mapping = {hero: index for index, hero in enumerate(hero_ids, 1)}
    encoded = [encode_draft(hero, allies, enemies, mapping) for hero in candidates]
    if not encoded:
        return []
    # creates one model input row for each candidate
    carry = torch.tensor([row[0] for row in encoded])
    ally = torch.tensor([row[1] for row in encoded])
    enemy = torch.tensor([row[2] for row in encoded])
    inputs = [carry, ally, enemy]
    history = getattr(model, "historical_bundle", None)
    if history:
        # selects and scales the historical inputs saved with the model
        indices = [FEATURE_FIELDS.index(field) for field in history["fields"]]
        values = torch.tensor(
            [feature_vector(history["stats"], hero, allies, enemies) for hero in candidates]
        )[:, indices]
        inputs.append((values - torch.tensor(history["mean"])) / torch.tensor(history["scale"]))
    # skips gradient tracking and turns raw scores into probabilities
    with torch.no_grad():
        scores = model(*inputs).sigmoid().tolist()
    # sorts equal scores by hero id for a consistent order
    return sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]))


# removes picked heroes and returns the top recommendations
def rank_candidates(model, hero_ids, pool, allies, enemies, top_n=5, banned=()):
    if top_n < 0:
        raise ValueError("top_n must be nonnegative")
    visible = set(allies + enemies) | set(banned)
    candidates = sorted(set(pool) - visible)
    return score_candidates(model, hero_ids, candidates, allies, enemies)[:top_n]
