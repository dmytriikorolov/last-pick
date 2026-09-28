# defines the carry model and converts visible drafts into model inputs

import csv
from pathlib import Path

import torch
from torch import nn


ROLE_FIELDS = ("is_carry", "is_support", "is_nuker", "is_disabler",
               "is_durable", "is_escape", "is_pusher", "is_initiator")
ATTRS = ("agi", "int", "str", "all")
ATTACKS = ("Melee", "Ranged")


# loads each hero attribute, attack type and roles as numbers
def static_hero_features(hero_ids, path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data/constants/heroes.csv"
    with path.open(newline="", encoding="utf-8") as source:
        rows = {int(row["hero_id"]): row for row in csv.DictReader(source)}
    # reserves row zero for missing picks and 14 columns for hero details
    features = torch.zeros(len(hero_ids) + 1, 14)
    for index, hero_id in enumerate(hero_ids, 1):
        row = rows[hero_id]
        # marks the matching attribute and attack type with 1 and the others with 0
        features[index] = torch.tensor(
            [float(row["primary_attr"] == attr) for attr in ATTRS] +
            [float(row["attack_type"] == attack) for attack in ATTACKS] +
            [float(row[field]) for field in ROLE_FIELDS])
    return features


# averages hero vectors while ignoring empty pick slots
def masked_mean(values, indices):
    # adds a dimension so the mask applies to every number in each hero vector
    mask = (indices != 0).unsqueeze(-1)
    # divides by at least one so an empty group returns zeros
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


# learns a carry score from the candidate and visible draft
class CarryModel(nn.Module):
    # creates hero vectors and the layers that turn draft inputs into a score
    def __init__(self, hero_ids, use_static=False, embedding_dim=16,
                 historical_dim=0, hidden_sizes=(64, 32)):
        super().__init__()
        self.use_static = use_static
        self.historical_dim = historical_dim
        # learns a vector for each hero and keeps empty slot zero as a zero vector
        self.embedding = nn.Embedding(len(hero_ids) + 1, embedding_dim, padding_idx=0)
        # saves fixed hero details with the model without training them
        self.register_buffer("hero_features", static_hero_features(hero_ids))
        # uses five learned vectors and optionally three groups of 14 fixed details
        input_size = embedding_dim * 5 + (42 if use_static else 0) + historical_dim
        layers = []
        # each hidden layer mixes the inputs and applies relu to its output
        for size in hidden_sizes:
            layers.extend([nn.Linear(input_size, size), nn.ReLU()])
            input_size = size
        # the last layer produces one raw score per draft
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    # combines hero vectors and candidate interactions into one draft vector
    def representations(self, candidate, allies, enemies):
        carry = self.embedding(candidate)
        ally = self.embedding(allies)
        enemy = self.embedding(enemies)
        # includes the candidate, average allies and average enemies
        # multiplying candidate and hero vectors adds interaction inputs
        parts = [carry, masked_mean(ally, allies), masked_mean(enemy, enemies),
                 masked_mean(carry.unsqueeze(1) * ally, allies),
                 masked_mean(carry.unsqueeze(1) * enemy, enemies)]
        # optionally adds fixed details for the candidate and both visible groups
        if self.use_static:
            parts.extend([self.hero_features[candidate],
                          masked_mean(self.hero_features[allies], allies),
                          masked_mean(self.hero_features[enemies], enemies)])
        # joins the inputs into columns while keeping one row per draft
        return torch.cat(parts, dim=1)

    # calculates raw scores from draft inputs and optional historical stats
    def forward(self, candidate, allies, enemies, historical=None):
        values = self.representations(candidate, allies, enemies)
        if self.historical_dim:
            if historical is None or historical.shape[1] != self.historical_dim:
                raise ValueError("historical feature vector missing or wrong width")
            values = torch.cat((values, historical), dim=1)
        # removes the extra output dimension and leaves probability conversion to the caller
        return self.network(values).squeeze(1)


# converts hero ids into model indices and fills empty slots with zeros
def encode_draft(candidate_id, ally_ids, enemy_ids, hero_to_index):
    # accepts the visible pick counts used by the training dataset
    if (len(ally_ids), len(enemy_ids)) not in {(2, 2), (3, 2), (4, 4)}:
        raise ValueError("draft needs 2/2, 3/2, or 4/4 visible allies/enemies")
    visible = [candidate_id] + ally_ids + enemy_ids
    # a hero cannot appear more than once in the same draft
    if len(set(visible)) != len(visible):
        raise ValueError("candidate and visible heroes must be unique")
    # both hero groups always have four slots even when fewer picks are visible
    try:
        return (hero_to_index[candidate_id],
                [hero_to_index[hero] for hero in ally_ids] + [0] * (4 - len(ally_ids)),
                [hero_to_index[hero] for hero in enemy_ids] + [0] * (4 - len(enemy_ids)))
    except KeyError as error:
        raise ValueError(f"unknown hero ID: {error.args[0]}") from error
