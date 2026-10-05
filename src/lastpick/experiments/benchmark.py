# reproducible observational benchmark orchestration

import csv
import importlib.metadata
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lastpick.data.io import atomic_json
from lastpick.evaluation.splits import temporal_split
from lastpick.experiments.audit import utc
from lastpick.experiments.data import (
    candidate_pool,
    dataset_identity,
    digest,
    load_decisions,
    verify_canonical,
)
from lastpick.experiments.diagnostics import drift_report, ranking_report, score_drafts, sensitivity
from lastpick.experiments.features import HistoricalFeatures, static_features
from lastpick.experiments.metrics import (
    bootstrap,
    calibration,
    classification,
    paired_bootstrap,
    subgroup_metrics,
)
from lastpick.experiments.models import NeuralModel, PriorModel, TabularModel

BASE = ("candidate", "draft")
RICH = BASE + ("static", "profile", "synergy", "matchup", "farming_stage", "support", "context")
EXPLORATORY_MODELS = {"catboost_static", "catboost_profile", "catboost_pairs"}
SUITE = {
    "global": ("prior", ("candidate",)),
    "hero": ("prior", ("candidate",)),
    "hero_side": ("prior", ("candidate", "context")),
    "popularity": ("prior", ("candidate",)),
    "lr_candidate": ("lr", ("candidate",)),
    "lr_draft": ("lr", BASE),
    "lr_context": ("lr", BASE + ("context",)),
    "lr_static": ("lr", BASE + ("static",)),
    "lr_profile": ("lr", BASE + ("static", "profile", "support")),
    "lr_synergy": ("lr", BASE + ("static", "profile", "synergy", "support")),
    "lr_matchup": ("lr", BASE + ("static", "profile", "synergy", "matchup", "support")),
    "lr_farming": ("lr", tuple(g for g in RICH if g != "context")),
    "lr_history": ("lr", tuple(g for g in RICH if g not in ("static", "context"))),
    "lr_full": ("lr", RICH),
    "catboost_raw": ("catboost", BASE + ("context",)),
    "catboost_static": ("catboost", BASE + ("static", "context")),
    "catboost_profile": ("catboost", BASE + ("static", "profile", "support", "context")),
    "catboost_pairs": (
        "catboost",
        BASE + ("static", "profile", "synergy", "matchup", "support", "context"),
    ),
    "catboost_full": ("catboost", RICH),
    "lightgbm_raw": ("lightgbm", BASE + ("context",)),
    "lightgbm_full": ("lightgbm", RICH),
    "nn_draft": ("nn", BASE),
    "nn_full": ("nn", RICH),
}


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def split_summary(parts, cutoff, test_cutoff):
    return {
        "strategy": "chronological matches; equal start timestamps remain together",
        "validation_cutoff": cutoff,
        "test_cutoff": test_cutoff,
        "validation_cutoff_utc": utc(cutoff),
        "test_cutoff_utc": utc(test_cutoff),
        "partitions": {
            name: {
                "rows": len(rows),
                "matches": len({d.match_id for d in rows}),
                "start": utc(min(d.start_time for d in rows)),
                "end": utc(max(d.start_time for d in rows)),
                "patch_counts": {
                    str(p): sum(d.patch == p for d in rows)
                    for p in sorted({d.patch for d in rows}, key=str)
                },
            }
            for name, rows in parts.items()
        },
    }


def evaluate_predictions(rows, probabilities, seed, repetitions):
    y = np.asarray([int(d.outcome) for d in rows])
    return {
        "metrics": classification(y, probabilities),
        "calibration": calibration(y, probabilities),
        "subgroups": subgroup_metrics(rows, y, probabilities),
        "bootstrap": bootstrap(y, probabilities, [d.match_id for d in rows], seed, repetitions)
        if repetitions
        else None,
        "role_sensitivity": {
            str(threshold): {
                "count": sum(d.role.confidence >= threshold for d in rows),
                "hero_counts": {
                    str(h): sum(d.chosen_hero == h and d.role.confidence >= threshold for d in rows)
                    for h in sorted({d.chosen_hero for d in rows if d.role.confidence >= threshold})
                },
                "metrics": classification(
                    y[[d.role.confidence >= threshold for d in rows]],
                    np.asarray(probabilities)[[d.role.confidence >= threshold for d in rows]],
                ),
            }
            for threshold in (0.65, 0.75, 0.85, 0.9)
        },
    }


def benchmark(
    root: Path,
    build: Path,
    canonical: Path,
    output: Path,
    models: list[str] | None = None,
    seed: int = 42,
    minimum: int = 100,
    repetitions: int = 200,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    patch_holdout: bool = False,
    max_epochs: int = 20,
    holdout_start: int | None = None,
) -> Path:
    selected = models or [name for name in SUITE if name not in EXPLORATORY_MODELS]
    if unknown := set(selected) - set(SUITE):
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if output.exists():
        raise FileExistsError(f"use a fresh run directory: {output}")
    rows = load_decisions(build)
    canonical_verification = verify_canonical(build, canonical)
    if patch_holdout and holdout_start is not None:
        raise ValueError("choose patch holdout or a time holdout")
    if patch_holdout or holdout_start is not None:
        patches = sorted({d.patch for d in rows if d.patch is not None})
        supported = [p for p in patches if sum(d.patch == p for d in rows) >= 1000]
        if patch_holdout and len(supported) < 2:
            raise ValueError("insufficient patch support for separate holdout")
        if holdout_start is not None:
            test_cutoff = holdout_start
        else:
            last = supported[-1]
            test_cutoff = min(d.start_time for d in rows if d.patch == last)
        earlier = [d for d in rows if d.start_time < test_cutoff]
        later = [d for d in rows if d.start_time >= test_cutoff]
        if len(earlier) < 3000 or len(later) < 1000:
            raise ValueError("insufficient chronological patch holdout support")
        # reserve the last 20% of earlier timestamps for validation
        times = sorted({d.start_time for d in earlier})
        cutoff = times[int(0.8 * len(times))]
        parts = {
            "train": [d for d in earlier if d.start_time < cutoff],
            "validation": [d for d in earlier if d.start_time >= cutoff],
            "test": later,
        }
    else:
        split = temporal_split(rows, train_fraction, validation_fraction)
        cutoff, test_cutoff = split.validation_cutoff, split.test_cutoff
        parts = {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        }
    train, validation, test = parts.values()
    pool = candidate_pool(train, minimum)
    if len(pool) < 2:
        raise ValueError("insufficient training-supported candidates")
    output.mkdir(parents=True)
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    source_files = sorted((root / "src/lastpick").rglob("*.py")) + [root / "pyproject.toml"]
    config = {
        "experiment": "modeling-evaluation-v2",
        "git_commit": version,
        "git_dirty": bool(status),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in source_files},
        "dataset": dataset_identity(build),
        "canonical_verification": canonical_verification,
        "seed": seed,
        "models": selected,
        "candidate_minimum_training_rows": minimum,
        "evaluation_candidate_pool": list(pool),
        "split": split_summary(parts, cutoff, test_cutoff),
        "patch_holdout": patch_holdout,
        "holdout_start": holdout_start,
        "history_scope": "training match IDs only; completion < decision start; evaluation frozen at validation cutoff",
        "training_ids_sha256": None,
        "bootstrap_repetitions": repetitions,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "torch")
        },
        "max_nn_epochs": max_epochs,
    }
    for dependency in ("catboost", "lightgbm"):
        try:
            config["versions"][dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            config["versions"][dependency] = "unavailable"
    atomic_json(output / "training_ids.json", sorted({d.match_id for d in train}))
    config["training_ids_sha256"] = digest(output / "training_ids.json")
    atomic_json(output / "config.json", config)
    write_csv(
        output / "split.csv",
        [
            {"match_id": d.match_id, "side": d.side, "split": name, "start_time": d.start_time}
            for name, partition in parts.items()
            for d in partition
        ],
    )
    print(
        f"building history: {len(train)} training rows, {len(pool)} evaluation candidates",
        flush=True,
    )
    history = HistoricalFeatures(
        canonical, {d.match_id for d in train}, cutoff, train + validation + test
    )
    with (output / "history.pkl").open("wb") as stream:
        pickle.dump(history, stream)
    write_csv(
        output / "history_availability.csv",
        [
            {"match_id": m, "side": s, "as_of": t, "source_matches": n, "latest_completion": latest}
            for m, s, t, n, latest in history.availability
        ],
    )
    atomic_json(output / "drift.json", drift_report(parts, history))
    static = static_features(root / "data/constants/heroes.csv")
    hero_universe = sorted(
        {h for d in train for h in (d.chosen_hero,) + d.visible_allies + d.visible_enemies}
    )
    results, predictions, comparisons = {}, {}, []
    for name in selected:
        kind, groups = SUITE[name]
        print(f"training {name}: {groups}", flush=True)
        directory = output / name
        directory.mkdir()
        if kind == "prior":
            model = PriorModel(name).fit(train)
        elif kind == "nn":
            model = NeuralModel(groups, seed, max_epochs=max_epochs).fit(train, validation, history)
        else:
            try:
                model = TabularModel(kind, groups, seed, static).fit(train, validation, history)
            except (ImportError, OSError) as error:
                # an unavailable optional dependency is an explicit failed model, not a hidden fallback
                results[name] = {"status": "unavailable", "error": str(error)}
                atomic_json(directory / "metrics.json", results[name])
                print(f"{name} unavailable: {error}", flush=True)
                continue
        with (directory / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream)
        entry = {
            "status": "complete",
            "feature_groups": groups,
            "parameters": model.parameters,
            "train_seconds": model.train_seconds,
            "partitions": {},
        }
        probability_rows, rank_rows = [], []
        for split_name in ("validation", "test"):
            partition = parts[split_name]
            p = model.predict(partition, history)
            predictions[(name, split_name)] = p
            evaluation = evaluate_predictions(
                partition, p, seed, repetitions if split_name == "test" else 0
            )
            scores = score_drafts(model, partition, pool, history)
            evaluation["ranking"] = ranking_report(partition, scores)
            if split_name == "test":
                evaluation["sensitivity"] = sensitivity(
                    model, partition, pool, history, hero_universe, seed
                )
            entry["partitions"][split_name] = evaluation
            for d, probability, values in zip(partition, p, scores, strict=True):
                probability_rows.append(
                    {
                        "split": split_name,
                        "match_id": d.match_id,
                        "side": d.side,
                        "chosen_hero": d.chosen_hero,
                        "outcome": int(d.outcome),
                        "probability": float(probability),
                    }
                )
                order = sorted(values, key=lambda h: (-values[h], h))
                rank_rows.append(
                    {
                        "split": split_name,
                        "match_id": d.match_id,
                        "side": d.side,
                        "chosen_rank": order.index(d.chosen_hero) + 1
                        if d.chosen_hero in order
                        else None,
                        "pool_size": len(order),
                        "top5": json.dumps(order[:5]),
                        "score_range": max(values.values()) - min(values.values())
                        if values
                        else None,
                    }
                )
            comparisons.append(
                {
                    "model": name,
                    "split": split_name,
                    **evaluation["metrics"],
                    "train_seconds": model.train_seconds,
                }
            )
            print(f"{name} {split_name}: {evaluation['metrics']}", flush=True)
        write_csv(directory / "predictions.csv", probability_rows)
        write_csv(directory / "rankings.csv", rank_rows)
        atomic_json(directory / "metrics.json", entry)
        for artifact, key in (
            ("ranking_metrics", "ranking"),
            ("subgroup_metrics", "subgroups"),
            ("calibration", "calibration"),
            ("role_sensitivity", "role_sensitivity"),
        ):
            atomic_json(
                directory / f"{artifact}.json", {s: v[key] for s, v in entry["partitions"].items()}
            )
        results[name] = entry
        atomic_json(output / "metrics.json", results)
    complete = [
        name for name in selected if results[name]["status"] == "complete" and name != "popularity"
    ]
    winner = min(
        complete, key=lambda n: results[n]["partitions"]["validation"]["metrics"]["log_loss"]
    )

    # fit sigmoid calibration on validation only, after raw model selection
    def logits(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1000, random_state=seed).fit(
        logits(predictions[(winner, "validation")]), [int(d.outcome) for d in validation]
    )
    calibrated = calibrator.predict_proba(logits(predictions[(winner, "test")]))[:, 1]
    calibration_result = {
        "base_model": winner,
        "fit_split": "validation only",
        "validation_metrics": "omitted because validation was used to fit the calibrator",
        "test": evaluate_predictions(test, calibrated, seed, repetitions),
        "slope": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    atomic_json(output / "sigmoid_calibration.json", calibration_result)
    write_csv(
        output / "sigmoid_predictions.csv",
        [
            {
                "match_id": d.match_id,
                "side": d.side,
                "probability": float(p),
                "outcome": int(d.outcome),
            }
            for d, p in zip(test, calibrated, strict=True)
        ],
    )
    with (output / "sigmoid_model.pkl").open("wb") as stream:
        pickle.dump(calibrator, stream)
    y = np.asarray([int(d.outcome) for d in test])
    intervals = (
        {
            name: paired_bootstrap(
                y,
                predictions[(winner, "test")],
                predictions[(name, "test")],
                [d.match_id for d in test],
                seed,
                repetitions,
            )
            for name in complete
            if name != winner
        }
        if repetitions
        else {}
    )
    atomic_json(
        output / "paired_comparisons.json",
        {"validation_selected_winner": winner, "test_differences": intervals},
    )
    winners = {
        "selection_metric": "validation log loss",
        "validation_selected": winner,
        "test_log_loss": min(
            complete, key=lambda n: results[n]["partitions"]["test"]["metrics"]["log_loss"]
        ),
        "test_auc": max(
            complete, key=lambda n: results[n]["partitions"]["test"]["metrics"]["roc_auc"] or 0
        ),
        "test_ece": min(complete, key=lambda n: results[n]["partitions"]["test"]["metrics"]["ece"]),
        "calibrated_test": calibration_result["test"]["metrics"],
        "context_sensitive": max(
            complete,
            key=lambda n: (
                results[n]["partitions"]["test"]["sensitivity"]["ally"]["top1_change_common_pool"]
                + results[n]["partitions"]["test"]["sensitivity"]["enemy"][
                    "top1_change_common_pool"
                ]
            ),
        ),
        "caution": "test minima are descriptive comparisons, not independent evidence from subsequent test-based tuning",
    }
    atomic_json(output / "winners.json", winners)
    write_csv(output / "comparison.csv", comparisons)
    report = render_report(config, results, winners)
    (output / "report.md").write_text(report)
    atomic_json(
        output / "artifacts.json",
        {
            str(p.relative_to(output)): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifacts.json"
        },
    )
    print(f"completed {output}; validation-selected winner: {winner}", flush=True)
    return output


def render_report(config, results, winners):
    lines = [
        "# modeling v2 benchmark",
        "",
        f"git commit `{config['git_commit']}`, seed {config['seed']}, dataset `{config['dataset']['decisions_sha256']}`",
        "",
        "## split",
        "",
        "```json",
        json.dumps(config["split"], indent=2),
        "```",
        "",
        "## probability comparison",
        "",
        "| model | split | accuracy | auc | log loss | brier | ece |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        if result["status"] != "complete":
            continue
        for split, entry in result["partitions"].items():
            m = entry["metrics"]
            lines.append(
                f"| {name} | {split} | {m['accuracy']:.5f} | {m['roc_auc']:.5f} | {m['log_loss']:.5f} | {m['brier']:.5f} | {m['ece']:.5f} |"
            )
    lines += [
        "",
        "## observed-pick agreement and behavior",
        "",
        "| model | recall@1 | recall@3 | recall@5 | mrr | top1 heroes | ally changes | enemy changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        if result["status"] != "complete":
            continue
        entry = result["partitions"]["test"]
        r, s = entry["ranking"], entry["sensitivity"]
        m = r["eligible_metrics"]
        lines.append(
            f"| {name} | {m['recall_at_1']:.5f} | {m['recall_at_3']:.5f} | {m['recall_at_5']:.5f} | {m['mrr']:.5f} | {r['top1_unique_heroes']} | {s['ally']['top1_change_common_pool']:.5f} | {s['enemy']['top1_change_common_pool']:.5f} |"
        )
    lines += [
        "",
        "observed-pick agreement measures imitation of the historically selected hero, not causal recommendation quality or counterfactual regret",
        "",
        "## winners",
        "",
        "```json",
        json.dumps(winners, indent=2),
        "```",
        "",
        "test minima are descriptive; the benchmark selection rule uses validation log loss",
        "",
    ]
    return "\n".join(lines)


def evaluate_run(
    run: Path, build: Path, output: Path, seed: int = 42, repetitions: int = 200
) -> dict:
    config = json.loads((run / "config.json").read_text())
    if (
        dataset_identity(build)["decisions_sha256"] != config["dataset"]["decisions_sha256"]
        or dataset_identity(build)["manifest_sha256"] != config["dataset"]["manifest_sha256"]
    ):
        raise ValueError("evaluation dataset differs from recorded run")
    checksums = json.loads((run / "artifacts.json").read_text())
    for name, expected in checksums.items():
        if digest(run / name) != expected:
            raise ValueError(f"run artifact digest mismatch: {name}")
    rows = [d for d in load_decisions(build) if d.start_time >= config["split"]["test_cutoff"]]
    # pickles are local experiment artifacts; never load an untrusted downloaded model
    with (run / "history.pkl").open("rb") as stream:
        history = pickle.load(stream)
    result = {}
    for name in config["models"]:
        path = run / name / "model.pkl"
        if not path.exists():
            continue
        with path.open("rb") as stream:
            model = pickle.load(stream)
        p = model.predict(rows, history)
        result[name] = evaluate_predictions(rows, p, seed, repetitions)
    atomic_json(output, result)
    return result
