# Small models for repeatable research runs
# Keep their outputs away from the protected ISP model files

import copy
import ctypes
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lastpick.experiments.features import HISTORY_GROUPS, FeatureEncoder
from lastpick.experiments.metrics import classification


class PriorModel:
    def __init__(self, kind, shrinkage=100):
        self.kind = kind
        self.shrinkage = shrinkage
        self.groups = ("candidate", "context") if kind == "hero_side" else ("candidate",)

    def fit(self, rows):
        started = time.perf_counter()
        self.base = float(np.mean([d.outcome for d in rows]))
        self.counts, self.wins = Counter(), Counter()
        self.side_counts, self.side_wins = Counter(), Counter()
        for d in rows:
            self.counts[d.chosen_hero] += 1
            self.wins[d.chosen_hero] += d.outcome
            self.side_counts[(d.chosen_hero, d.side)] += 1
            self.side_wins[(d.chosen_hero, d.side)] += d.outcome
        self.parameters = {"kind": self.kind, "shrinkage": self.shrinkage, "hero_side_minimum": 100}
        self.train_seconds = time.perf_counter() - started
        return self

    def predict(self, rows, history=None, candidates=None):
        output = []
        for i, d in enumerate(rows):
            h = candidates[i] if candidates is not None else d.chosen_hero
            if self.kind in ("global", "popularity"):
                value = self.base
            else:
                value = (self.wins[h] + self.shrinkage * self.base) / (
                    self.counts[h] + self.shrinkage
                )
                key = (h, d.side)
                # sparse hero/side cells fall back to the shrunk hero
                # prior
                if self.kind == "hero_side" and self.side_counts[key] >= 100:
                    value = (self.side_wins[key] + self.shrinkage * value) / (
                        self.side_counts[key] + self.shrinkage
                    )
            output.append(value)
        return np.asarray(output)

    def ranking_scores(self, rows, history, candidates):
        if self.kind == "popularity":
            return np.asarray([self.counts[h] for h in candidates], dtype=float)
        return self.predict(rows, history, candidates)


class TabularModel:
    def __init__(self, kind, groups, seed=42, static=None):
        self.kind, self.groups, self.seed = kind, tuple(groups), seed
        self.encoder = FeatureEncoder(groups, static)

    def records(self, rows, history, candidates=None):
        needs_history = any(g in HISTORY_GROUPS for g in self.groups)
        return [
            self.encoder.record(
                d,
                history.for_decision(d, None if candidates is None else candidates[i])
                if needs_history
                else None,
                None if candidates is None else candidates[i],
            )
            for i, d in enumerate(rows)
        ]

    def cat_matrix(self, rows, records, candidates=None):
        numeric = self.encoder.raw(records)[:, self.numeric_indices]
        categorical = []
        for i, d in enumerate(rows):
            row = [str(d.chosen_hero if candidates is None else candidates[i])]
            if "context" in self.groups:
                row += [str(d.side), str(d.patch), d.rank_bracket or "unknown", str(d.region)]
            categorical.append(row)
        return np.concatenate(
            [np.asarray(categorical, dtype=object), numeric.astype(object)], axis=1
        )

    def fit(self, train, validation, history):
        started = time.perf_counter()
        records, valid_records = self.records(train, history), self.records(validation, history)
        x = self.encoder.fit(records)
        xv = self.encoder.transform(valid_records)
        y = np.asarray([d.outcome for d in train], dtype=int)
        yv = np.asarray([d.outcome for d in validation], dtype=int)
        self.search = []
        best = float("inf")
        if self.kind == "catboost":
            from catboost import CatBoostClassifier

            self.numeric_indices = [
                i
                for i, name in enumerate(self.encoder.vectorizer.get_feature_names_out())
                if "=" not in name
            ]
            cat_count = 5 if "context" in self.groups else 1
            x, xv = self.cat_matrix(train, records), self.cat_matrix(validation, valid_records)
            options = [{"depth": 4, "l2_leaf_reg": 10}, {"depth": 6, "l2_leaf_reg": 20}]
        elif self.kind == "lightgbm":
            # macOS wheels need libomp
            # PyTorch often supplies a compatible copy
            if sys.platform == "darwin":
                import torch

                library = Path(torch.__file__).parent / "lib/libomp.dylib"
                if library.exists():
                    ctypes.CDLL(str(library))
            from lightgbm import LGBMClassifier, early_stopping, log_evaluation

            x, xv = self.encoder.raw(records), self.encoder.raw(valid_records)
            options = [{"num_leaves": 7}, {"num_leaves": 15}]
        else:
            options = [{"C": c} for c in (0.01, 0.1, 1.0)]
        for parameters in options:
            if self.kind == "lr":
                model = LogisticRegression(
                    **parameters, max_iter=400, solver="lbfgs", random_state=self.seed, tol=1e-5
                )
                model.fit(x, y)
            elif self.kind == "catboost":
                model = CatBoostClassifier(
                    **parameters,
                    iterations=400,
                    learning_rate=0.04,
                    loss_function="Logloss",
                    random_seed=self.seed,
                    thread_count=4,
                    verbose=False,
                    allow_writing_files=False,
                )
                model.fit(
                    x,
                    y,
                    cat_features=list(range(cat_count)),
                    eval_set=(xv, yv),
                    early_stopping_rounds=40,
                )
            elif self.kind == "lightgbm":
                model = LGBMClassifier(
                    **parameters,
                    n_estimators=400,
                    learning_rate=0.03,
                    min_child_samples=100,
                    reg_lambda=10,
                    colsample_bytree=0.8,
                    random_state=self.seed,
                    n_jobs=4,
                    verbosity=-1,
                    deterministic=True,
                    force_col_wise=True,
                )
                model.fit(
                    x,
                    y,
                    eval_set=[(xv, yv)],
                    callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
                )
            else:
                raise ValueError(self.kind)
            loss = classification(yv, model.predict_proba(xv)[:, 1])["log_loss"]
            self.search.append({"parameters": parameters, "validation_log_loss": loss})
            if loss < best:
                best, self.model, self.parameters = loss, model, parameters
        self.parameters = {
            **self.parameters,
            "estimator_parameters": self.model.get_params(),
            "search": self.search,
            "seed": self.seed,
            "max_iterations": 400,
        }
        if self.kind == "catboost":
            self.parameters["tree_count"] = self.model.tree_count_
        elif self.kind == "lightgbm":
            self.parameters["best_iteration"] = self.model.best_iteration_
        self.train_seconds = time.perf_counter() - started
        return self

    def predict(self, rows, history, candidates=None):
        records = self.records(rows, history, candidates)
        if self.kind == "catboost":
            x = self.cat_matrix(rows, records, candidates)
        elif self.kind == "lightgbm":
            x = self.encoder.raw(records)
        else:
            x = self.encoder.transform(records)
        return self.model.predict_proba(x)[:, 1]

    ranking_scores = predict


class NeuralModel:
    def __init__(self, groups, seed=42, max_epochs=20, patience=4):
        self.kind, self.groups, self.seed = "nn", tuple(groups), seed
        self.max_epochs, self.patience = max_epochs, patience

    def inputs(self, rows, history, candidates=None):
        import torch

        candidate, allies, enemies, records = [], [], [], []
        needs_history = any(g in HISTORY_GROUPS for g in self.groups)
        for i, d in enumerate(rows):
            h = d.chosen_hero if candidates is None else candidates[i]
            candidate.append(self.hero_to_index.get(h, 0))
            allies.append(
                [self.hero_to_index.get(a, 0) for a in sorted(d.visible_allies)]
                if "draft" in self.groups
                else [0] * 4
            )
            enemies.append(
                [self.hero_to_index.get(a, 0) for a in sorted(d.visible_enemies)]
                if "draft" in self.groups
                else [0] * 4
            )
            records.append(
                self.encoder.record(
                    d,
                    history.for_decision(d, None if candidates is None else h)
                    if needs_history
                    else None,
                    None if candidates is None else h,
                )
            )
        numeric = self.encoder.vectorizer.transform(records).toarray()[:, self.numeric_indices]
        numeric = (
            self.numeric_scaler.transform(numeric)
            if len(self.numeric_indices)
            else np.empty((len(rows), 0))
        )
        return tuple(torch.as_tensor(v, dtype=torch.long) for v in (candidate, allies, enemies)) + (
            torch.as_tensor(numeric, dtype=torch.float32),
        )

    def fit(self, train, validation, history):
        import torch

        from lastpick.models.carry import CarryModel

        started = time.perf_counter()
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        heroes = sorted(
            {h for d in train for h in (d.chosen_hero,) + d.visible_allies + d.visible_enemies}
        )
        self.hero_to_index = {h: i for i, h in enumerate(heroes, 1)}
        # candidate/draft identity comes from the existing
        # permutation-invariant embeddings
        self.encoder = FeatureEncoder(
            [g for g in self.groups if g not in ("candidate", "draft", "static")]
        )
        records = [self.encoder.record(d, history.for_decision(d)) for d in train]
        self.encoder.fit(records)
        names = self.encoder.vectorizer.get_feature_names_out()
        self.numeric_indices = [
            i for i, name in enumerate(names) if not name.startswith("candidate=")
        ]
        numeric = self.encoder.vectorizer.transform(records).toarray()[:, self.numeric_indices]
        self.numeric_scaler = StandardScaler()
        if len(self.numeric_indices):
            self.numeric_scaler.fit(numeric)
        self.model = CarryModel(
            heroes,
            use_static="static" in self.groups,
            embedding_dim=16,
            historical_dim=len(self.numeric_indices),
            hidden_sizes=(64, 32),
        )
        inputs, valid = self.inputs(train, history), self.inputs(validation, history)
        targets = torch.tensor([float(d.outcome) for d in train])
        valid_targets = torch.tensor([float(d.outcome) for d in validation])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.001)
        generator = torch.Generator().manual_seed(self.seed)
        best, stale, epochs = float("inf"), 0, []
        for epoch in range(self.max_epochs):
            self.model.train()
            for indices in torch.randperm(len(train), generator=generator).split(512):
                optimizer.zero_grad()
                logits = self.model(*(v[indices] for v in inputs))
                # every accepted canonical label has weight one
                # confidence is not a target probability
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, targets[indices]
                )
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                loss = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        self.model(*valid), valid_targets
                    )
                )
            epochs.append({"epoch": epoch + 1, "validation_log_loss": loss})
            if loss < best - 1e-6:
                best, state, stale, self.best_epoch = (
                    loss,
                    copy.deepcopy(self.model.state_dict()),
                    0,
                    epoch + 1,
                )
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model.load_state_dict(state)
        self.model.eval()
        self.parameters = {
            "embedding_dim": 16,
            "hidden_sizes": [64, 32],
            "learning_rate": 0.001,
            "weight_decay": 0.001,
            "batch_size": 512,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "best_epoch": self.best_epoch,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "epochs": epochs,
            "seed": self.seed,
        }
        self.train_seconds = time.perf_counter() - started
        return self

    def predict(self, rows, history, candidates=None):
        import torch

        self.model.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(rows), 4096):
                stop = start + 4096
                inputs = self.inputs(
                    rows[start:stop],
                    history,
                    None if candidates is None else candidates[start:stop],
                )
                result.extend(torch.sigmoid(self.model(*inputs)).tolist())
        return np.asarray(result)

    ranking_scores = predict
