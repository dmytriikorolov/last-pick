# Check how rankings respond to changes in a draft
# Measure observed-pick agreement and shifts in the data

from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
from scipy.stats import spearmanr

from lastpick.experiments.data import evaluation_pool
from lastpick.experiments.metrics import ranking_agreement


def score_drafts(model, rows, pool, history, batch_size=128):
    output = []
    for start in range(0, len(rows), batch_size):
        block = rows[start : start + batch_size]
        contexts, candidates, lengths = [], [], []
        for row in block:
            available = evaluation_pool(row, pool)
            lengths.append(len(available))
            contexts.extend([row] * len(available))
            candidates.extend(available)
        scores = model.ranking_scores(contexts, history, candidates) if candidates else []
        offset = 0
        for length in lengths:
            output.append(
                dict(
                    zip(
                        candidates[offset : offset + length],
                        map(float, scores[offset : offset + length]),
                        strict=True,
                    )
                )
            )
            offset += length
    return output


def ranking_report(rows, scores):
    ranks, winning, losing = [], [], []
    top1, top5 = Counter(), Counter()
    dispersions, nearly_equal = [], 0
    for d, values in zip(rows, scores, strict=True):
        ordered = sorted(values, key=lambda h: (-values[h], h))
        if ordered:
            top1[ordered[0]] += 1
            top5.update(ordered[:5])
            spread = max(values.values()) - min(values.values())
            dispersions.append(spread)
            nearly_equal += spread < 0.001
        if d.chosen_hero in ordered:
            rank = ordered.index(d.chosen_hero) + 1
            ranks.append(rank)
            (winning if d.outcome else losing).append(rank)
    total = len(rows)
    probabilities = (
        np.asarray(list(top1.values()), dtype=float) / sum(top1.values())
        if top1
        else np.asarray([])
    )
    return {
        **ranking_agreement(ranks, len(ranks), total),
        "winning_observed_picks": ranking_agreement(
            winning, len(winning), sum(d.outcome for d in rows)
        ),
        "losing_observed_picks": ranking_agreement(
            losing, len(losing), sum(not d.outcome for d in rows)
        ),
        "top1_unique_heroes": len(top1),
        "top5_unique_heroes": len(top5),
        "top1_percent": {str(h): 100 * count / total for h, count in top1.most_common()},
        "top5_appearances": dict(top5),
        "top1_effective_heroes": float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        if len(probabilities)
        else 0,
        "mean_score_range": float(np.mean(dispersions)) if dispersions else None,
        "nearly_identical_scores_fraction": nearly_equal / total if total else None,
        "tie_breaking": "descending score, then ascending hero ID; ties make global-prior ranking arbitrary",
    }


def sensitivity(model, rows, pool, history, hero_universe, seed=42, size=256):
    rng = np.random.default_rng(seed)
    selected = [rows[i] for i in sorted(rng.choice(len(rows), min(size, len(rows)), replace=False))]
    original = score_drafts(model, selected, pool, history)
    result = {"sample_size": len(selected), "seed": seed}
    for group in ("ally", "enemy"):
        mutated = []
        for d in selected:
            visible = set(d.visible_allies + d.visible_enemies + d.visible_bans + (d.chosen_hero,))
            replacements = sorted(set(hero_universe) - visible)
            if not replacements:
                raise ValueError("insufficient heroes for context sensitivity")
            new = int(rng.choice(replacements))
            heroes = list(d.visible_allies if group == "ally" else d.visible_enemies)
            heroes[int(rng.integers(0, 4))] = new
            mutated.append(
                replace(
                    d, **{"visible_allies" if group == "ally" else "visible_enemies": tuple(heroes)}
                )
            )
        altered = score_drafts(model, mutated, pool, history)
        changes, changes_common, correlations, max_deltas = [], [], [], []
        for before, after in zip(original, altered, strict=True):
            if not before or not after:
                continue

            def top(scores):
                return min(scores, key=lambda h: (-scores[h], h))

            changes.append(top(before) != top(after))
            common = sorted(set(before) & set(after))
            if not common:
                continue
            a, b = {h: before[h] for h in common}, {h: after[h] for h in common}
            changes_common.append(top(a) != top(b))
            if (
                np.std(list(a.values())) > 1e-10
                and np.std(list(b.values())) > 1e-10
                and len(common) >= 3
            ):
                correlations.append(float(spearmanr(list(a.values()), list(b.values())).statistic))
            max_deltas.append(max(abs(a[h] - b[h]) for h in common))
        result[group] = {
            "top1_change_fraction": float(np.mean(changes)) if changes else None,
            "top1_change_common_pool": float(np.mean(changes_common)) if changes_common else None,
            "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
            "mean_max_score_change": float(np.mean(max_deltas)) if max_deltas else None,
            "common_pool_note": "Common candidates isolate scoring changes from a newly picked hero leaving the evaluation pool.",
        }
    different = []
    for a, b in zip(original[::2], original[1::2]):
        common = sorted(set(a) & set(b))
        av, bv = [a[h] for h in common], [b[h] for h in common]
        if len(common) >= 3 and np.std(av) > 1e-10 and np.std(bv) > 1e-10:
            different.append(float(spearmanr(av, bv).statistic))
    result["different_drafts_mean_rank_correlation"] = (
        float(np.mean(different)) if different else None
    )
    # explicit alternate-candidate path recomputes historical pair
    # features for changed drafts
    candidates = [d.chosen_hero for d in selected]
    base = model.predict(selected, history, candidates)
    swapped = model.predict([replace(d, side=1 - d.side) for d in selected], history, candidates)
    delta = np.asarray(
        [
            swapped[i] - base[i] if d.side == 1 else base[i] - swapped[i]
            for i, d in enumerate(selected)
        ]
    )
    result["side_effect"] = {
        "side_is_input": "context" in model.groups,
        "mean_radiant_minus_dire": float(delta.mean()),
        "mean_absolute_effect": float(np.abs(delta).mean()),
        "max_absolute_effect": float(np.abs(delta).max()),
    }
    ally_permuted = model.predict(
        [replace(d, visible_allies=tuple(reversed(d.visible_allies))) for d in selected],
        history,
        candidates,
    )
    enemy_permuted = model.predict(
        [replace(d, visible_enemies=tuple(reversed(d.visible_enemies))) for d in selected],
        history,
        candidates,
    )
    result["invariance"] = {
        "ally_max_difference": float(np.max(np.abs(base - ally_permuted))),
        "enemy_max_difference": float(np.max(np.abs(base - enemy_permuted))),
    }
    return result


def drift_report(parts, history):
    result = {
        "selection": "All changes are observational; collection policy, role filters and patch mix can confound them."
    }
    summaries = {}
    for name, rows in parts.items():
        heroes = defaultdict(lambda: Counter())
        for d in rows:
            heroes[d.chosen_hero]["count"] += 1
            heroes[d.chosen_hero]["wins"] += d.outcome
        summaries[name] = {
            str(h): {
                "count": v["count"],
                "pick_rate": v["count"] / len(rows),
                "win_rate": v["wins"] / v["count"],
                "reliable": v["count"] >= 100,
            }
            for h, v in sorted(heroes.items())
        }
    result["candidate_distributions"] = summaries
    training = summaries["train"]
    for name in ("validation", "test"):
        other = summaries[name]
        result[f"train_to_{name}_pick_total_variation"] = 0.5 * sum(
            abs(training.get(h, {}).get("pick_rate", 0) - other.get(h, {}).get("pick_rate", 0))
            for h in set(training) | set(other)
        )
    names = sorted(next(iter(history.observed.values())))
    train_features = np.asarray(
        [[history.for_decision(d)[n] for n in names] for d in parts["train"]]
    )
    mean, std = train_features.mean(axis=0), train_features.std(axis=0)
    result["historical_feature_shift"] = {}
    for split in ("validation", "test"):
        matrix = np.asarray([[history.for_decision(d)[n] for n in names] for d in parts[split]])
        result["historical_feature_shift"][split] = {
            n: {
                "train_mean": float(mean[i]),
                "evaluation_mean": float(matrix[:, i].mean()),
                "mean_shift_in_train_std": float((matrix[:, i].mean() - mean[i]) / std[i])
                if std[i] > 0
                else None,
            }
            for i, n in enumerate(names)
        }
    return result
