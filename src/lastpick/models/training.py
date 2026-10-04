# trains one carry neural network with hero details and historical features
import csv
import json
import time
from collections import Counter

import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lastpick.config import load_config, project_root
from lastpick.features.carry import (
    FEATURE_FIELDS,
    feature_bundle,
    features_for_rows,
    load_training_matches,
    normalize,
    training_features,
)
from lastpick.legacy.hero_lookup import names_by_id
from lastpick.legacy.logistic import load_rows, split_rows
from lastpick.models.carry import CarryModel, encode_draft

ROOT = project_root()
CONFIG_PATH = ROOT / "config/default.toml"
CONFIG = load_config(CONFIG_PATH if CONFIG_PATH.exists() else None)
TRAINING = CONFIG.training
DATA = ROOT / "data/processed"
MODELS = ROOT / "models"
MIN_CANDIDATE_EXAMPLES = TRAINING.min_candidate_examples
MAX_EPOCHS = TRAINING.max_epochs
PATIENCE = TRAINING.patience
BATCH_SIZE = TRAINING.batch_size
LEARNING_RATE = TRAINING.learning_rate
MODEL_PATH = ROOT / TRAINING.model_path
RESULTS_PATH = ROOT / TRAINING.results_path


# loads quality rows using match id and team as the key
def quality_by_decision(path):
    with path.open(newline="", encoding="utf-8") as source:
        quality = {}
        for row in csv.DictReader(source):
            key = (row["match_id"], row["side"])
            quality[key] = row
    return quality


# adds the saved quality weight to each training example
def join_quality(rows, quality):
    for row in rows:
        key = (row["match_id"], row["side"])
        row["training_weight"] = float(quality[key]["training_weight"])
    return rows


# converts draft rows and labels into numeric model inputs
def tensors(rows, hero_to_index):
    candidates, allies, enemies, labels, weights = [], [], [], [], []
    for row in rows:
        ally_ids = [int(row[f"ally_{i}"]) for i in range(1, 5) if row[f"ally_{i}"]]
        enemy_ids = [int(row[f"enemy_{i}"]) for i in range(1, 5) if row[f"enemy_{i}"]]
        candidate, ally, enemy = encode_draft(
            int(row["candidate_hero_id"]), ally_ids, enemy_ids, hero_to_index
        )
        candidates.append(candidate)
        allies.append(ally)
        enemies.append(enemy)
        labels.append(int(row["win"]))
        weights.append(row["training_weight"])
    return (
        torch.tensor(candidates, dtype=torch.long),
        torch.tensor(allies, dtype=torch.long),
        torch.tensor(enemies, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


# calculates prediction loss while giving questionable examples less weight
def weighted_loss(logits, labels, weights):
    if (
        weights.shape != labels.shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("training weights must be finite, nonnegative and have positive sum")
    # keeps each example loss separate before applying its weight
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum()


# gets model win probabilities in batches
def probabilities(model, data, history):
    # switches to prediction mode without changing learned weights
    model.eval()
    output = []
    # skips gradient tracking because no training happens here
    with torch.no_grad():
        for start in range(0, len(data[0]), 2048):
            batch = [part[start : start + 2048] for part in data[:3]]
            output.append(model(*batch, history[start : start + 2048]).sigmoid())
    return torch.cat(output).numpy()


# calculates accuracy, auc, log loss and brier score
def metrics(labels, scores):
    return {
        "accuracy": float(accuracy_score(labels, scores >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }


# trains one neural network and keeps the epoch with the best validation loss
def train_model(train_data, validation_data, train_history, validation_history, hero_ids, bundle):
    torch.manual_seed(CONFIG.seed)
    # includes hero details, draft interactions and every historical feature
    model = CarryModel(hero_ids, use_static=True, historical_dim=len(FEATURE_FIELDS))
    model.historical_bundle = bundle
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(
        TensorDataset(*train_data[:3], train_history, train_data[3], train_data[4]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(CONFIG.seed),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    epochs = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for candidate, allies, enemies, history, labels, weights in loader:
            optimizer.zero_grad()
            loss = weighted_loss(model(candidate, allies, enemies, history), labels, weights)
            loss.backward()
            optimizer.step()
        scores = probabilities(model, validation_data, validation_history)
        validation_metrics = metrics(validation_data[3].numpy(), scores)
        val_loss = validation_metrics["log_loss"]
        epochs.append({"epoch": epoch, **validation_metrics})
        print(f"epoch {epoch}: validation loss {val_loss:.6f}", flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            # copies the best weights so later epochs cannot change them
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "epochs": epochs,
        "training_seconds": round(time.perf_counter() - started, 2),
    }


# builds historical inputs and trains the model used by the website
def main():
    protected = {
        ROOT / "models/final_carry_recommender.pt",
        ROOT / "models/final_carry_training_results.json",
    }
    if MODEL_PATH.resolve() in protected or RESULTS_PATH.resolve() in protected:
        raise ValueError("ISP artifacts are protected; choose separate experiment output paths")
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    rows = join_quality(
        load_rows(DATA / "training_examples.csv"),
        quality_by_decision(DATA / "training_quality.csv"),
    )
    # keeps later matches separate for validation and testing
    early, test_rows, test_cutoff = split_rows(rows)
    train_rows, validation_rows, validation_cutoff = split_rows(early, 0.85)
    train_rows = [row for row in train_rows if row["training_weight"] > 0]
    validation_rows = [row for row in validation_rows if row["training_weight"] > 0]
    test_rows = [row for row in test_rows if row["training_weight"] > 0]
    counts = Counter(int(row["candidate_hero_id"]) for row in train_rows)
    pool = sorted(hero for hero, count in counts.items() if count >= MIN_CANDIDATE_EXAMPLES)
    hero_ids = sorted(names_by_id)
    mapping = {hero: index for index, hero in enumerate(hero_ids, 1)}
    train_data = tensors(train_rows, mapping)
    validation_data = tensors(validation_rows, mapping)
    test_data = tensors(test_rows, mapping)

    # historical training inputs use earlier matches rather than their own results
    print("building historical inputs", flush=True)
    matches = load_training_matches(DATA, train_rows, validation_cutoff)
    train_values, stats, blocks = training_features(
        train_rows, matches, freeze_before=validation_cutoff, settings=CONFIG.features
    )
    bundle = feature_bundle(stats, train_values, FEATURE_FIELDS)
    train_history = torch.tensor(normalize(train_values, bundle))
    # validation and test inputs use only the completed training history
    validation_history = torch.tensor(normalize(features_for_rows(validation_rows, stats), bundle))
    test_history = torch.tensor(normalize(features_for_rows(test_rows, stats), bundle))
    model, training = train_model(
        train_data, validation_data, train_history, validation_history, hero_ids, bundle
    )

    # measures test performance after the best epoch has been selected
    test_metrics = metrics(test_data[3].numpy(), probabilities(model, test_data, test_history))
    checkpoint = {
        "model_state": model.state_dict(),
        "hero_ids": hero_ids,
        "candidate_pool": pool,
        "minimum_examples": MIN_CANDIDATE_EXAMPLES,
        "use_static": True,
        "embedding_dim": 16,
        "hidden_sizes": [64, 32],
        "historical_bundle": bundle,
        "name": "historical_nn",
        "seed": CONFIG.seed,
        "split_cutoffs": {"validation_cutoff": validation_cutoff, "test_cutoff": test_cutoff},
    }

    # writes a complete checkpoint before replacing the website model
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(MODEL_PATH)
    result = {
        "name": "historical_nn",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "validation_cutoff": validation_cutoff,
        "test_cutoff": test_cutoff,
        "candidate_pool_size": len(pool),
        "historical_fields": FEATURE_FIELDS,
        "historical_blocks": blocks,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "test_metrics": test_metrics,
        "model_path": str(MODEL_PATH),
        **training,
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
