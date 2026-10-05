import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from test_foundation_drafts import canonical

from lastpick.config import Config
from lastpick.data.dataset import build_dataset, decisions_for_match
from lastpick.data.io import atomic_json
from lastpick.data.schemas import Rejection, record
from lastpick.evaluation.splits import temporal_split
from lastpick.experiments.audit import audit_dataset
from lastpick.experiments.benchmark import benchmark, evaluate_run
from lastpick.experiments.data import (
    candidate_pool,
    dataset_identity,
    draft_input,
    evaluation_pool,
    load_decisions,
)
from lastpick.experiments.features import FeatureEncoder, HistoricalFeatures
from lastpick.experiments.metrics import (
    bootstrap,
    calibration,
    classification,
    cluster_indices,
    ranking_agreement,
    subgroup_metrics,
)
from lastpick.experiments.models import NeuralModel, PriorModel, TabularModel
from lastpick.experiments.review import role_review_sample


def examples(count=30):
    match, _, _ = canonical(both_teams=True)
    matches, rows = [], []
    for i in range(count):
        m = replace(
            match,
            match_id=100 + i,
            start_time=10000 + (i // 2) * 5000,
            duration=1000,
            radiant_win=i % 3 == 0,
        )
        matches.append(m)
        rows.extend(decisions_for_match(m, Config())[0])
    return matches, rows


def corpus(tmp_path, count=30):
    matches, _ = examples(count)
    directory = tmp_path / Config().paths.normalized
    files = []
    for m in matches:
        name = f"{m.match_id}.json"
        atomic_json(directory / name, record(m))
        files.append(name)
    atomic_json(
        directory / "index.json",
        {
            "schema_version": "2.0",
            "files": files,
            "raw_matches": count,
            "accepted": count,
            "rejected": 0,
            "raw_digest": "fixture",
        },
    )
    return directory, build_dataset(tmp_path, Config(), "fixture")


def safe_history(tmp_path, count=30):
    canonical_dir, build = corpus(tmp_path, count)
    rows = load_decisions(build)
    split = temporal_split(rows)
    history = HistoricalFeatures(
        canonical_dir,
        {d.match_id for d in split.train},
        split.validation_cutoff,
        list(split.train + split.validation + split.test),
    )
    return split, history


def test_split_matches_equal_times_and_candidate_support():
    _, rows = examples()
    split = temporal_split(rows)
    groups = [split.train, split.validation, split.test]
    ids = [{d.match_id for d in g} for g in groups]
    times = [{d.start_time for d in g} for g in groups]
    for i in range(3):
        for j in range(i):
            assert not ids[i] & ids[j]
            assert not times[i] & times[j]
    modified = [replace(d, chosen_hero=99) for d in split.test]
    assert 99 not in candidate_pool(split.train, 1)
    assert 99 in candidate_pool(modified, 1)


def test_candidate_exclusions_and_unknown_bans():
    _, rows = examples()
    d = replace(rows[0], visible_bans=(99,), unlocated_bans=(100,))
    pool = tuple(sorted(set(d.visible_allies + d.visible_enemies + (d.chosen_hero, 99, 100))))
    assert evaluation_pool(d, pool) == tuple(sorted((d.chosen_hero, 100)))
    with pytest.raises(Rejection):
        draft_input(d, d.visible_allies[0])


def test_encoder_deterministic_order_and_training_only_preprocessing():
    _, rows = examples()
    d = rows[0]
    encoder = FeatureEncoder(("candidate", "draft", "context"))
    record = encoder.record(d)
    reordered = replace(
        d,
        visible_allies=tuple(reversed(d.visible_allies)),
        visible_enemies=tuple(reversed(d.visible_enemies)),
    )
    assert record == encoder.record(reordered)
    encoder.fit([record])
    vocab = dict(encoder.vectorizer.vocabulary_)
    scaling = encoder.scaler.scale_.copy()
    encoder.transform([encoder.record(replace(d, patch=999, chosen_hero=99))])
    assert vocab == encoder.vectorizer.vocabulary_
    np.testing.assert_array_equal(scaling, encoder.scaler.scale_)
    assert "candidate=99" not in vocab and "patch=999" not in vocab
    forbidden = ("confidence", "role", "outcome", "gpm", "position_est", "unlocated")
    assert all(not any(f in key for f in forbidden) for key in record)


def test_history_frozen_and_completion_safe(tmp_path):
    split, history = safe_history(tmp_path)
    allowed = {d.match_id for d in split.train}
    assert set(history.source_ids) <= allowed
    for _, _, cutoff, _, latest in history.availability:
        assert latest is None or latest < cutoff
    assert all(
        cutoff == split.validation_cutoff
        for _, _, cutoff, _, _ in history.availability[len(split.train) :]
    )
    with pytest.raises(ValueError, match="evaluation match"):
        HistoricalFeatures(
            tmp_path / Config().paths.normalized,
            {d.match_id for d in split.test},
            split.validation_cutoff,
            list(split.test),
        )


def test_calibration_ece_and_probability_endpoints():
    result = calibration([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["ece"] == 0
    assert sum(r["count"] for r in result["bins"]) == 4
    assert calibration([0, 0, 1, 1], [0.5] * 4)["ece"] == 0
    assert calibration([0, 0], [0.75, 0.75])["ece"] == 0.75
    with pytest.raises(ValueError):
        calibration([0], [np.nan])


def test_match_cluster_bootstrap_and_reproducibility():
    ids = [1, 1, 2, 2, 3, 3]
    indices = cluster_indices(ids, np.random.default_rng(42))
    for m in set(ids):
        selected = [i for i in indices if ids[i] == m]
        assert selected.count(ids.index(m)) == selected.count(ids.index(m) + 1)
    a = bootstrap([0, 1] * 3, [0.3, 0.7] * 3, ids, repetitions=10)
    assert a == bootstrap([0, 1] * 3, [0.3, 0.7] * 3, ids, repetitions=10)
    assert a["unit"] == "match"


def test_ranking_agreement_and_support():
    metrics = ranking_agreement([1, 2, 6], 3, 4)
    assert metrics["eligible_metrics"]["recall_at_1"] == 1 / 3
    assert metrics["eligible_metrics"]["recall_at_3"] == 2 / 3
    assert metrics["all_decision_recall"]["recall_at_1"] == 1 / 4
    assert metrics["coverage"] == 0.75
    _, rows = examples()
    low = subgroup_metrics(rows[:2], [0, 1], [0.4, 0.6])
    assert not low["patch"]["60"]["reliable"]
    assert "log_loss" not in low["patch"]["60"]


def test_models_reproducible_and_order_side_invariant(tmp_path):
    split, history = safe_history(tmp_path)
    train, valid, test = list(split.train), list(split.validation), list(split.test)
    models = [
        PriorModel("hero").fit(train),
        TabularModel("lr", ("candidate", "draft")).fit(train, valid, history),
        NeuralModel(("candidate", "draft"), max_epochs=2).fit(train, valid, history),
    ]
    repeat = TabularModel("lr", ("candidate", "draft")).fit(train, valid, history)
    np.testing.assert_allclose(models[1].predict(test, history), repeat.predict(test, history))
    for model in models:
        candidates = [d.chosen_hero for d in test]
        p = model.predict(test, history, candidates)
        for field in ("visible_allies", "visible_enemies"):
            permuted = [replace(d, **{field: tuple(reversed(getattr(d, field)))}) for d in test]
            np.testing.assert_allclose(p, model.predict(permuted, history, candidates), atol=1e-7)
        np.testing.assert_allclose(
            p,
            model.predict([replace(d, side=1 - d.side) for d in test], history, candidates),
            atol=1e-7,
        )


def test_dataset_manifest_digest_integrity_and_audit(tmp_path):
    canonical_dir, build = corpus(tmp_path)
    identity = dataset_identity(build)
    assert len(identity["decisions_sha256"]) == 64
    audit = audit_dataset(build, canonical_dir, tmp_path / "audit")
    assert audit["last_pick_decisions"] == 60
    assert audit["accepted_match_contribution"] == {"2": 30}
    with (build / "last_pick_decisions.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(Rejection, match="dataset_digest"):
        load_decisions(build)


def test_role_review_deterministic_and_human_labels_blank(tmp_path):
    canonical_dir, _ = corpus(tmp_path)
    first, second = tmp_path / "review.csv", tmp_path / "review2.csv"
    role_review_sample(canonical_dir, first, Config(), size=10)
    role_review_sample(canonical_dir, second, Config(), size=10)
    assert first.read_bytes() == second.read_bytes()
    with first.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert all(
        not row[k] for row in rows for k in ("reviewed_pos1_slot", "reviewer_confidence", "notes")
    )


def test_tiny_end_to_end_run_and_reload(tmp_path):
    canonical_dir, build = corpus(tmp_path)
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    output = benchmark(
        root,
        build,
        canonical_dir,
        tmp_path / "run",
        models=["global", "hero", "lr_draft"],
        minimum=1,
        repetitions=2,
    )
    original = json.loads((output / "metrics.json").read_text())
    repeated = evaluate_run(output, build, tmp_path / "evaluation.json", repetitions=2)
    assert repeated["lr_draft"]["metrics"] == original["lr_draft"]["partitions"]["test"]["metrics"]
    assert (output / "artifacts.json").exists()
    assert classification([0, 1], [0.2, 0.8])["log_loss"] < 0.3
