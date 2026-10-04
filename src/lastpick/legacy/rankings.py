#!/usr/bin/env python3
# checks how carry rankings change between test drafts
# I added this file to just check if the model changes it opinion when we change allies and enemies
import csv
import json
from itertools import combinations

from scipy.stats import spearmanr

from lastpick.config import project_root

if __package__:
    from lastpick.legacy.hero_lookup import hero_id_to_name
    from lastpick.legacy.logistic import load_rows, split_rows
    from lastpick.recommender.carry import DEFAULT_MODEL, load_model, score_candidates
else:
    from lastpick.legacy.hero_lookup import hero_id_to_name
    from lastpick.legacy.logistic import load_rows, split_rows
    from lastpick.recommender.carry import DEFAULT_MODEL, load_model, score_candidates


ROOT = project_root()


# gets the visible ally and enemy hero ids from a training row
def visible(row):
    return (
        [int(row[f"ally_{i}"]) for i in range(1, 5) if row[f"ally_{i}"]],
        [int(row[f"enemy_{i}"]) for i in range(1, 5) if row[f"enemy_{i}"]],
    )


# selects test drafts with different pick counts and match times
def sample_contexts(test):
    shapes = [(2, 2), (3, 2), (4, 4)]
    first_by_shape = []
    # takes one draft for each pick count pattern from separate matches
    for shape in shapes:
        first_by_shape.append(
            next(
                row
                for row in test
                if (int(row["known_ally_count"]), int(row["known_enemy_count"])) == shape
                and row["match_id"] not in {chosen["match_id"] for chosen in first_by_shape}
            )
        )
    # selects 20 evenly spaced rows across the test data
    spread = [test[round(i * (len(test) - 1) / 19)] for i in range(20)]
    return first_by_shape + [row for row in spread if row not in first_by_shape]


# compares candidate rankings between drafts and shows example recommendations
def rankings_report(model, hero_ids, pool, contexts):
    changed = 0
    correlations = []
    # compares each pair of sampled drafts once
    for left, right in combinations(contexts, 2):
        la, le = visible(left)
        ra, re = visible(right)
        # uses the same candidates for both drafts and removes heroes picked in either
        blocked = set(la + le + ra + re)
        candidates = [hero for hero in pool if hero not in blocked]
        if len(candidates) < 3:
            continue
        left_ranks = score_candidates(model, hero_ids, candidates, la, le)
        right_ranks = score_candidates(model, hero_ids, candidates, ra, re)
        # counts pairs where the highest ranked hero changes
        changed += left_ranks[0][0] != right_ranks[0][0]
        positions = {hero: i for i, (hero, _) in enumerate(left_ranks)}
        # spearman compares ranking order with 1 meaning identical and -1 reversed
        correlations.append(
            float(
                spearmanr(
                    list(range(len(candidates))), [positions[hero] for hero, _ in right_ranks]
                ).statistic
            )
        )
    examples = []
    # excludes every hero visible in the first three drafts for a shared candidate pool
    common_visible = {hero for row in contexts[:3] for group in visible(row) for hero in group}
    common_candidates = [hero for hero in pool if hero not in common_visible]
    for row in contexts[:3]:
        allies, enemies = visible(row)
        ranking = score_candidates(model, hero_ids, common_candidates, allies, enemies)
        examples.append(
            {
                "match_id": row["match_id"],
                "allies": [hero_id_to_name(hero) for hero in allies],
                "enemies": [hero_id_to_name(hero) for hero in enemies],
                "top_five": [
                    (hero_id_to_name(hero), round(score, 4)) for hero, score in ranking[:5]
                ],
            }
        )
    return {
        "draft_pairs": len(correlations),
        "different_top_one": changed,
        "average_spearman": sum(correlations) / len(correlations),
        "example_candidate_count": len(common_candidates),
        "examples": examples,
    }


# loads the model and test drafts and prints ranking and replacement checks
def main():
    model, hero_ids, pool = load_model(DEFAULT_MODEL)
    rows = load_rows(ROOT / "data/processed/training_examples.csv")
    _, test, _ = split_rows(rows)
    # removes examples that the quality check excluded from training and evaluation
    with (ROOT / "data/processed/training_quality.csv").open(newline="") as source:
        excluded = {
            (row["match_id"], row["side"])
            for row in csv.DictReader(source)
            if row["training_weight"] == "0.0"
        }
    test = [row for row in test if (row["match_id"], row["side"]) not in excluded]
    contexts = sample_contexts(test)
    report = rankings_report(model, hero_ids, pool, contexts)
    first = contexts[0]
    allies, enemies = visible(first)
    # picks an available candidate and two replacement heroes without duplicates
    candidate = next(hero for hero in pool if hero not in set(allies + enemies))
    alternative_ally = next(
        hero for hero in hero_ids if hero not in set(allies + enemies + [candidate])
    )
    alternative_enemy = next(
        hero
        for hero in hero_ids
        if hero not in set(allies + enemies + [candidate, alternative_ally])
    )
    # scores the same candidate before and after replacing one visible hero
    base = score_candidates(model, hero_ids, [candidate], allies, enemies)[0][1]
    ally_changed = score_candidates(
        model, hero_ids, [candidate], allies[:-1] + [alternative_ally], enemies
    )[0][1]
    enemy_changed = score_candidates(
        model, hero_ids, [candidate], allies, enemies[:-1] + [alternative_enemy]
    )[0][1]
    # saves the replaced hero names and resulting scores alongside the base score
    report["interaction_check"] = {
        "candidate": hero_id_to_name(candidate),
        "base_score": round(base, 4),
        "replace_ally": [
            hero_id_to_name(allies[-1]),
            hero_id_to_name(alternative_ally),
            round(ally_changed, 4),
        ],
        "replace_enemy": [
            hero_id_to_name(enemies[-1]),
            hero_id_to_name(alternative_enemy),
            round(enemy_changed, 4),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
