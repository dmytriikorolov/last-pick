# defines a model that predicts the result from both complete teams

import torch
from torch import nn

# these positions describe all ten unique pairs within a five hero team
pair_left = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
pair_right = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]


# averages the multiplied hero vectors for every ally pair
def ally_interactions(team):
    return (team[:, pair_left] * team[:, pair_right]).mean(dim=1)


# averages the multiplied hero vectors for all 25 enemy pairs
def cross_team_interactions(radiant, dire):
    # extra dimensions let each radiant hero combine with every dire hero
    return (radiant[:, :, None] * dire[:, None, :]).mean(dim=(1, 2))


# learns a score from both teams and optionally their hero interactions
class FullDraftSynergyModel(nn.Module):
    # creates the learned hero vectors and prediction layers
    def __init__(self, hero_count, embedding_dim=16, use_interactions=True):
        super().__init__()
        self.use_interactions = use_interactions
        self.embedding = nn.Embedding(hero_count, embedding_dim)
        # uses two team averages and optionally three interaction vectors
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * (5 if use_interactions else 2), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    # turns a complete draft into one raw radiant win score
    def forward(self, hero_indices):
        heroes = self.embedding(hero_indices)
        # the first five heroes are radiant and the remaining five are dire
        radiant, dire = heroes[:, :5], heroes[:, 5:]
        parts = [radiant.mean(dim=1), dire.mean(dim=1)]
        if self.use_interactions:
            parts += [
                ally_interactions(radiant),
                ally_interactions(dire),
                cross_team_interactions(radiant, dire),
            ]
        # joins the vectors into one input row per match
        features = torch.cat(parts, dim=1)
        # leaves probability conversion to the code that uses the model
        return self.network(features).squeeze(1)
