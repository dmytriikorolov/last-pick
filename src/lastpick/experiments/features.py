# explicit feature groups and training-only, completed-match historical inputs

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from lastpick.data.schemas import match_from_record
from lastpick.experiments.data import draft_input
from lastpick.features.asof import AsOfHistory
from lastpick.features.carry import FEATURE_FIELDS, feature_vector
from lastpick.models.carry import ATTACKS, ATTRS, ROLE_FIELDS

HISTORY_GROUPS = {
    "profile": ("hero_win_rate",),
    "synergy": ("synergy_mean_delta", "synergy_max_delta"),
    "matchup": ("matchup_mean_delta", "matchup_min_delta"),
    "farming_stage": (
        "early_win_rate",
        "mid_win_rate",
        "late_win_rate",
        "avg_gpm",
        "avg_gold_10",
        "avg_lh_10",
    ),
    "support": (
        "hero_log_games",
        "early_reliability",
        "mid_reliability",
        "late_reliability",
        "farm_reliability",
        "synergy_mean_log_games",
        "synergy_mean_reliability",
        "matchup_mean_log_games",
        "matchup_mean_reliability",
        "gpm_log_support",
        "gold_10_log_support",
        "lh_10_log_support",
    ),
}


class HistoricalFeatures:
    def __init__(self, directory, training_ids, cutoff, decisions):
        matches = []
        for match_id in sorted(training_ids):
            match = match_from_record(json.loads((directory / f"{match_id}.json").read_text()))
            if match.start_time is None or match.start_time >= cutoff:
                raise ValueError("evaluation match in historical training allowlist")
            # no item/upgrade features in this benchmark; retain canonical farm snapshots
            players = tuple(
                replace(p, purchases=None, final_items=(), ability_upgrades=None)
                for p in match.players
            )
            matches.append(replace(match, players=players))
        history = AsOfHistory(matches, set(training_ids))
        self.observed = {}
        self.observed_contexts = {}
        self.availability = []
        for d in decisions:
            as_of = min(d.start_time, cutoff)
            history.advance(as_of)
            self.observed[(d.match_id, d.side)] = self.values(
                history.stats, d.chosen_hero, d.visible_allies, d.visible_enemies
            )
            key = (
                d.match_id,
                d.start_time,
                d.chosen_hero,
                tuple(sorted(d.visible_allies)),
                tuple(sorted(d.visible_enemies)),
            )
            self.observed_contexts[key] = self.observed[(d.match_id, d.side)]
            latest = history.matches[history.position - 1][0] if history.position else None
            if latest is not None and latest >= as_of:
                raise ValueError("historical completion leakage")
            self.availability.append((d.match_id, d.side, as_of, history.position, latest))
        # the frozen state is used for alternate candidate scoring, never test outcomes
        history.advance(cutoff)
        self.stats = history.stats
        self.cutoff = cutoff
        self.source_ids = tuple(history.ids)

    @staticmethod
    def values(stats, candidate, allies, enemies):
        values = dict(
            zip(
                FEATURE_FIELDS,
                feature_vector(stats, candidate, sorted(allies), sorted(enemies)),
                strict=True,
            )
        )
        profile = stats["heroes"].get(str(candidate), [0.0] * 14)
        for name, i in (("gpm", 9), ("gold_10", 11), ("lh_10", 13)):
            values[f"{name}_log_support"] = math.log1p(profile[i])
        return values

    def for_decision(self, d, candidate=None):
        if d.start_time >= self.cutoff:
            return self.values(
                self.stats,
                d.chosen_hero if candidate is None else candidate,
                d.visible_allies,
                d.visible_enemies,
            )
        if candidate is not None and candidate != d.chosen_hero:
            raise ValueError("alternate candidate scoring requires frozen evaluation rows")
        # side is metadata and cannot choose another candidate's cached historical inputs
        key = (
            d.match_id,
            d.start_time,
            d.chosen_hero,
            tuple(sorted(d.visible_allies)),
            tuple(sorted(d.visible_enemies)),
        )
        if key not in self.observed_contexts:
            raise ValueError("training history requires an original observed draft and timestamp")
        return self.observed_contexts[key]


def static_features(path):
    with Path(path).open(newline="") as stream:
        return {
            int(row["hero_id"]): [float(row["primary_attr"] == a) for a in ATTRS]
            + [float(row["attack_type"] == a) for a in ATTACKS]
            + [float(row[k]) for k in ROLE_FIELDS]
            for row in csv.DictReader(stream)
        }


class FeatureEncoder:
    def __init__(self, groups, static=None):
        self.groups = tuple(groups)
        self.static = static or {}
        self.vectorizer = DictVectorizer(sparse=True, sort=True)
        self.scaler = StandardScaler(with_mean=False)

    def record(self, d, historical=None, candidate=None):
        context = draft_input(d, candidate)
        values = {"candidate": str(context.candidate)}
        if "draft" in self.groups:
            for group, heroes in (
                ("ally", context.allies),
                ("enemy", context.enemies),
                ("ban", context.bans),
            ):
                values.update({f"{group}:{h}": 1.0 for h in heroes})
        if "context" in self.groups:
            values.update(
                {
                    "side": str(context.side),
                    "patch": str(context.patch),
                    "rank": context.rank_bracket or "unknown",
                    "region": str(context.region),
                }
            )
            values["avg_rank_tier"] = context.avg_rank_tier or 0.0
            values["rank_missing"] = float(context.avg_rank_tier is None)
        if "static" in self.groups:
            for group, heroes in (
                ("candidate", (context.candidate,)),
                ("ally", context.allies),
                ("enemy", context.enemies),
            ):
                if group != "candidate" and "draft" not in self.groups:
                    continue
                matrix = [self.static.get(h, [0.0] * 14) for h in heroes]
                for i, value in enumerate(np.mean(matrix, axis=0)):
                    values[f"static:{group}:{i}"] = float(value)
        for group, names in HISTORY_GROUPS.items():
            if group in self.groups:
                if historical is None:
                    raise ValueError("historical feature group requires safe history")
                values.update({f"history:{name}": historical[name] for name in names})
        return values

    def fit(self, records):
        matrix = self.vectorizer.fit_transform(records)
        self.scaler.fit(matrix)
        return self.scaler.transform(matrix).astype(np.float32)

    def transform(self, records):
        return self.scaler.transform(self.vectorizer.transform(records)).astype(np.float32)

    def raw(self, records):
        # trees use original units. Binary hero indicators avoid ordinal hero IDs
        return self.vectorizer.transform(records).toarray().astype(np.float32)
