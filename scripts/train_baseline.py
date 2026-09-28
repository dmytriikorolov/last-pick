"""trains a logistic regression model from visible draft heroes"""
import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from scripts.hero_lookup import format_hero

root = Path(__file__).resolve().parent.parent
input_path = root / "data" / "processed" / "training_examples.csv"
model_path = root / "models" / "baseline_model.pkl"


# reads training examples from a csv file
def load_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


# splits earlier matches into training data and later matches into test data
def split_rows(rows, train_fraction=0.8):
    # equal timestamps stay together so a match cannot appear in both groups
    ordered_times = sorted({int(row["start_time"]) for row in rows})
    cutoff = ordered_times[int(len(ordered_times) * train_fraction)]
    train = [row for row in rows if int(row["start_time"]) < cutoff]
    test = [row for row in rows if int(row["start_time"]) >= cutoff]
    return train, test, cutoff


# collects the unique hero ids present in the training data
def hero_ids(rows):
    found = set()
    for row in rows:
        found.add(int(row["candidate_hero_id"]))
        for side in ("ally", "enemy"):
            for i in range(1, 5):
                value = row[f"{side}_{i}"]
                if value:
                    found.add(int(value))
    return sorted(found)


# creates one signed column per hero with allies positive and enemies negative
def features(rows, hero_to_index):
    # stores only nonzero values to avoid a large mostly empty table
    indices, data, indptr = [], [], [0]
    for row in rows:
        for field, sign in [("candidate_hero_id", 1)] + [
            (f"{side}_{i}", 1 if side == "ally" else -1)
            for side in ("ally", "enemy") for i in range(1, 5)
        ]:
            value = row[field]
            if value and int(value) in hero_to_index:
                indices.append(hero_to_index[int(value)])
                data.append(sign)
        # records where the next row begins in the stored values
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), shape=(len(rows), len(hero_to_index)), dtype=float)


# compares candidate rankings for two different drafts
def ranking_changes(model, rows, hero_to_index, candidates_seen):
    first = rows[0]
    second = next(row for row in rows if row["match_id"] != first["match_id"] and
                  any(row[f"ally_{i}"] != first[f"ally_{i}"] for i in range(1, 5)))
    # removes heroes already present in either draft
    blocked = set(hero_ids([first, second]))
    candidates = sorted(candidates_seen - blocked)
    rankings = []
    for context in (first, second):
        hypothetical = [dict(context, candidate_hero_id=str(hero)) for hero in candidates]
        scores = model.decision_function(features(hypothetical, hero_to_index))
        rankings.append(sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0])))
    changes = sum(a[0] != b[0] for a, b in zip(*rankings))
    return len(candidates), changes, first["match_id"], second["match_id"], rankings[0][0][0]


# formats a timestamp as a readable utc date
def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# trains the baseline model and prints its test results
def main():
    rows = load_rows(input_path)
    train, test, cutoff = split_rows(rows)
    hero_to_index = {hero: i for i, hero in enumerate(hero_ids(train))}
    x_train, x_test = features(train, hero_to_index), features(test, hero_to_index)
    y_train = np.array([int(row["win"]) for row in train])
    y_test = np.array([int(row["win"]) for row in test])
    # predicts win probability from the signed hero inputs
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(x_train, y_train)
    # column one contains the probability of winning
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, probability >= 0.5),
        "roc_auc": roc_auc_score(y_test, probability),
        "log_loss": log_loss(y_test, probability),
        "brier_score": brier_score_loss(y_test, probability),
    }
    candidates_seen = {int(row["candidate_hero_id"]) for row in train}
    candidate_count, changed, first_id, second_id, top_hero_id = ranking_changes(
        model, test, hero_to_index, candidates_seen)
    top_hero = format_hero(top_hero_id, include_id=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # saves the model and hero column mapping for later use
    with model_path.open("wb") as output:
        pickle.dump({"model": model, "hero_to_index": hero_to_index,
                     "cutoff": cutoff, "feature_rule": "candidate + allies - enemies"}, output)

    print(json.dumps({"train_rows": len(train), "test_rows": len(test), "cutoff_utc": utc(cutoff),
                      "metrics": metrics, "ranking_changes": changed,
                      "top_candidate": top_hero}, indent=2))
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
