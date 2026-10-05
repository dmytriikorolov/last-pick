# probability metrics, calibration and match-clustered uncertainty

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def calibration(y, probabilities, bins=10):
    if bins < 1:
        raise ValueError("positive calibration bin count required")
    y, p = np.asarray(y), np.asarray(probabilities)
    if len(y) != len(p) or not len(y) or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid calibration inputs")
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for i in range(bins):
        mask = assignment == i
        count = int(mask.sum())
        predicted = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(predicted - observed)
        table.append(
            {
                "bin": i,
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": count,
                "mean_prediction": predicted,
                "observed_win_rate": observed,
            }
        )
    return {
        "ece": ece,
        "bins": table,
        "definition": "10 equal-width bins, weighted absolute calibration gap"
        if bins == 10
        else f"{bins} equal-width bins",
    }


def classification(y, probabilities):
    y, p = np.asarray(y), np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    if not len(y):
        return {"count": 0, "reliable": False}
    return {
        "count": len(y),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece": calibration(y, p)["ece"],
    }


def cluster_indices(match_ids, rng):
    ids = np.asarray(match_ids)
    unique = np.unique(ids)
    groups = {m: np.flatnonzero(ids == m) for m in unique}
    return np.concatenate([groups[m] for m in rng.choice(unique, len(unique), replace=True)])


def bootstrap(y, p, match_ids, seed=42, repetitions=200):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(match_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    samples = {name: [] for name in ("accuracy", "roc_auc", "log_loss", "brier", "ece")}
    for _ in range(repetitions):
        index = np.concatenate([groups[i] for i in rng.integers(0, len(unique), len(unique))])
        metrics = classification(y[index], p[index])
        for name in samples:
            if metrics[name] is not None:
                samples[name].append(metrics[name])
    return {
        "unit": "match",
        "repetitions": repetitions,
        "seed": seed,
        "intervals_95": {
            name: [float(v) for v in np.quantile(values, [0.025, 0.975])] if values else None
            for name, values in samples.items()
        },
    }


def paired_bootstrap(y, first, second, match_ids, seed=42, repetitions=200):
    # Negative differences favor the first model for log loss and Brier.
    y, first, second = np.asarray(y), np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(match_ids, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    values = {"log_loss": [], "brier": []}
    for _ in range(repetitions):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(unique), len(unique))])
        a, b = (
            classification(y[indices], first[indices]),
            classification(y[indices], second[indices]),
        )
        for name in values:
            values[name].append(a[name] - b[name])
    return {
        name: {
            "difference": classification(y, first)[name] - classification(y, second)[name],
            "interval_95": [float(v) for v in np.quantile(vs, [0.025, 0.975])],
        }
        for name, vs in values.items()
    }


def ranking_agreement(ranks, eligible, total):
    ranks = np.asarray(ranks, dtype=float)
    return {
        "name": "observed-pick agreement",
        "total_decisions": total,
        "eligible_chosen_in_pool": eligible,
        "coverage": eligible / total if total else None,
        "eligible_metrics": {
            **{
                f"recall_at_{k}": float((ranks <= k).mean()) if len(ranks) else None
                for k in (1, 3, 5)
            },
            "mrr": float((1 / ranks).mean()) if len(ranks) else None,
            "mean_rank": float(ranks.mean()) if len(ranks) else None,
            "ndcg_at_5": float(np.where(ranks <= 5, 1 / np.log2(ranks + 1), 0).mean())
            if len(ranks)
            else None,
        },
        "all_decision_recall": {
            f"recall_at_{k}": int((ranks <= k).sum()) / total if total else None for k in (1, 3, 5)
        },
        "caveat": "Behavioral agreement with one observed pick. Unchosen carry outcomes are unknown; this is not recommendation quality or counterfactual regret.",
    }


def subgroup_metrics(rows, y, p, minimum=100):
    result = {}
    for name in ("patch", "rank_bracket", "side", "time_bucket"):
        groups = {}
        for i, row in enumerate(rows):
            value = row.start_time // (30 * 86400) if name == "time_bucket" else getattr(row, name)
            groups.setdefault(str(value), []).append(i)
        result[name] = {}
        for value, indices in sorted(groups.items()):
            indices = np.asarray(indices)
            observed = np.asarray(y)[indices]
            reliable = (
                len(indices) >= minimum
                and min(Counter(observed).values()) >= 20
                and len(set(observed)) == 2
            )
            entry = {
                "count": len(indices),
                "matches": len({rows[i].match_id for i in indices}),
                "reliable": reliable,
                "observed_win_rate": float(observed.mean()),
                "mean_prediction": float(np.asarray(p)[indices].mean()),
            }
            if reliable:
                entry.update(classification(observed, np.asarray(p)[indices]))
            else:
                entry["reason"] = "low support: require 100 rows and >=20 examples of each outcome"
            result[name][value] = entry
    return result
