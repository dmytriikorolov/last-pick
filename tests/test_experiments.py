import csv
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_foundation_drafts import canonical

from lastpick.config import Config
from lastpick.data.dataset import build_dataset, decisions_for_match
from lastpick.data.io import atomic_json
from lastpick.data.schemas import Rejection, record
from lastpick.evaluation.splits import temporal_split
from lastpick.experiments.audit import audit_dataset
from lastpick.experiments.benchmark import benchmark, completed_partition, evaluate_run
from lastpick.experiments.data import (
    candidate_pool,
    dataset_identity,
    draft_input,
    evaluation_pool,
    load_decisions,
    verify_canonical,
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
from lastpick.experiments.reporting import export_report
from lastpick.experiments.review import role_review_pilot, role_review_sample
from lastpick.experiments.rolling import DAY, WEEK, rolling_benchmark


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
            "rejection_reasons": {},
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


def test_label_cutoff_excludes_both_sides_and_unknown_duration(tmp_path):
    matches, decisions = examples(4)
    cutoff = 16_000
    for match, duration in zip(matches, (5_999, 6_000, None, 1_001), strict=True):
        atomic_json(
            tmp_path / f"{match.match_id}.json",
            record(replace(match, duration=duration)),
        )
    retained, exclusions = completed_partition(decisions, tmp_path, cutoff)
    assert [(row.match_id, row.side) for row in retained] == [(100, 0), (100, 1)]
    assert exclusions == {
        "unfinished_at_cutoff": {"matches": 2, "rows": 4},
        "unknown_completion": {"matches": 1, "rows": 2},
    }


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
        TabularModel("lr", ("candidate", "draft", "profile", "synergy", "matchup", "support")).fit(
            train, valid, history
        ),
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
        for partition in (train, test):
            base = model.predict(partition, history)
            np.testing.assert_allclose(
                base,
                model.predict([replace(d, side=1 - d.side) for d in partition], history),
                atol=1e-7,
            )


def test_dataset_manifest_digest_integrity_and_audit(tmp_path):
    canonical_dir, build = corpus(tmp_path)
    identity = dataset_identity(build)
    assert len(identity["decisions_sha256"]) == 64
    assert (
        verify_canonical(build, canonical_dir)["input_digest"] == identity["canonical_input_digest"]
    )
    source = canonical_dir / "100.json"
    content = source.read_bytes()
    source.write_bytes(content + b" ")
    with pytest.raises(Rejection, match="canonical_build_digest"):
        verify_canonical(build, canonical_dir)
    source.write_bytes(content)
    audit = audit_dataset(build, canonical_dir, tmp_path / "audit")
    assert audit["last_pick_decisions"] == 60
    assert audit["accepted_match_contribution"] == {"2": 30}
    decisions = build / "last_pick_decisions.jsonl"
    original = decisions.read_text()
    with decisions.open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(Rejection, match="dataset_digest"):
        load_decisions(build)
    rows = [json.loads(line) for line in original.splitlines()]
    assert rows[0]["match_id"] == rows[1]["match_id"]
    rows[1]["start_time"] += 1
    decisions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(Rejection, match="inconsistent_match_timestamp"):
        load_decisions(build, verify=False)


def test_role_review_deterministic_and_human_labels_blank(tmp_path):
    canonical_dir, _ = corpus(tmp_path)
    files = json.loads((canonical_dir / "index.json").read_text())["files"]
    for i, filename in enumerate(files):
        value = json.loads((canonical_dir / filename).read_text())
        for p in value["players"]:
            if i < 10:
                p["lane_role"] = None
            elif i < 20:
                carry = p["hero_id"] in (25, 17)
                p["lh_t"] = [65 if carry else 15] * 16
                p["gold_t"] = [4000 if carry else 2000] * 16
            else:
                carry = p["hero_id"] in (25, 17)
                p["lh_t"] = [None] * 10 + [30 if carry else 15]
                p["gold_t"] = [None] * 10 + [2400 if carry else 2000]
        atomic_json(canonical_dir / filename, value)
    first, second = tmp_path / "review.csv", tmp_path / "review2.csv"
    review = role_review_sample(canonical_dir, first, Config(), size=10)
    role_review_sample(canonical_dir, second, Config(), size=10)
    assert first.read_bytes() == second.read_bytes()
    with first.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert review["bucket_counts"] == {"high": 4, "medium": 3, "abstention": 3}
    assert all(
        not row[k] for row in rows for k in ("reviewed_pos1_slot", "reviewer_confidence", "notes")
    )


def test_role_review_pilot_hides_heuristic_and_preserves_reviewers_work(tmp_path):
    source = Path(__file__).resolve().parents[1] / "reports/role_review_sample.csv"
    output = tmp_path / "pilot.csv"
    assert role_review_pilot(source, output)["cases"] == 30
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len({row["match_id"] for row in rows}) == 30
    assert set(rows[0]) == {
        "match_id",
        "side",
        "patch",
        "rank_bracket",
        "hero_ids",
        "reviewed_pos1_slot",
        "reviewer_confidence",
        "notes",
    }
    with source.open() as stream:
        strata = {(row["match_id"], row["side"]): row["stratum"] for row in csv.DictReader(stream)}
    by_bucket = [strata[(row["match_id"], row["side"])].split(":")[0] for row in rows]
    assert {bucket: by_bucket.count(bucket) for bucket in set(by_bucket)} == {
        "abstention": 10,
        "medium": 10,
        "high": 10,
    }
    assert all(
        not row[field]
        for row in rows
        for field in ("reviewed_pos1_slot", "reviewer_confidence", "notes")
    )
    with pytest.raises(FileExistsError):
        role_review_pilot(source, output)


def test_rolling_benchmark_reserves_last_eligible_week(tmp_path, monkeypatch):
    _, source_rows = examples()
    anchor = 4 * DAY  # Monday January 5 1970.
    rows = [
        replace(source_rows[0], start_time=anchor + day * DAY)
        for day in range(42)
        for _ in range(160)
    ]
    monkeypatch.setattr("lastpick.experiments.rolling.load_decisions", lambda _build: rows)
    runs = []

    def capture(_root, _build, _canonical, output, **kwargs):
        runs.append((output.name, kwargs["window"]))

    monkeypatch.setattr("lastpick.experiments.benchmark.benchmark", capture)
    monkeypatch.setattr(
        "lastpick.experiments.rolling.summarize_week", lambda run: {"run": run.name}
    )
    rolling_benchmark(tmp_path, tmp_path, tmp_path, tmp_path / "rolling", False)
    report = json.loads((tmp_path / "rolling/coverage.json").read_text())
    assert report["reserved_week"] == "1970-02-09"
    assert report["eligible_weeks"] == 2
    assert len(runs) == 2
    assert runs[-1][1] == (anchor - WEEK, anchor + 3 * WEEK, anchor + 4 * WEEK, anchor + 5 * WEEK)
    assert sorted(report["test_results"]) == sorted(name for name, _ in runs)


@pytest.mark.parametrize("bounded", [False, True])
def test_tiny_end_to_end_run_and_reload(tmp_path, bounded):
    canonical_dir, build = corpus(tmp_path)
    root = Path(__file__).resolve().parents[1]
    output = benchmark(
        root,
        build,
        canonical_dir,
        tmp_path / "run",
        models=["global", "hero", "lr_draft"],
        minimum=1,
        repetitions=2,
        window=(None, 55_000, 70_000, 80_000) if bounded else None,
    )
    original = json.loads((output / "metrics.json").read_text())
    repeated = evaluate_run(output, build, tmp_path / "evaluation.json", repetitions=2)
    assert repeated["lr_draft"]["metrics"] == original["lr_draft"]["partitions"]["test"]["metrics"]
    if bounded:
        assert repeated["lr_draft"]["metrics"]["count"] == 8
        from lastpick.experiments.rolling import summarize_week

        weekly = summarize_week(output)
        assert weekly["models"]["lr_draft"]["picked_hero_coverage"] == 1
        assert (
            weekly["models"]["lr_draft"]["log_loss"] == repeated["lr_draft"]["metrics"]["log_loss"]
        )
    assert (output / "artifacts.json").exists()
    assert classification([0, 1], [0.2, 0.8])["log_loss"] < 0.3
    audit_dataset(build, canonical_dir, tmp_path / "audit")
    export_report(
        output,
        tmp_path / "audit/audit.json",
        tmp_path / "summary",
        tmp_path / "docs/MODELING_V2.md",
    )
    exported = json.loads((tmp_path / "summary/benchmark.json").read_text())
    assert exported["primary_config"]["dataset"] == dataset_identity(build)
    assert "lr_draft" in (tmp_path / "docs/MODELING_V2.md").read_text()


@pytest.mark.parametrize("kind", ["catboost", "lightgbm"])
def test_optional_tree_smoke_and_order_invariance(tmp_path, kind):
    if importlib.util.find_spec(kind) is None:
        pytest.skip(f"optional {kind} research dependency is absent")
    split, history = safe_history(tmp_path)
    model = TabularModel(kind, ("candidate", "draft", "context")).fit(
        list(split.train), list(split.validation), history
    )
    rows = list(split.test)
    candidates = [d.chosen_hero for d in rows]
    p = model.predict(rows, history, candidates)
    reordered = [
        replace(
            d,
            visible_allies=tuple(reversed(d.visible_allies)),
            visible_enemies=tuple(reversed(d.visible_enemies)),
        )
        for d in rows
    ]
    np.testing.assert_allclose(p, model.predict(reordered, history, candidates), atol=1e-10)
    assert np.all((p > 0) & (p < 1))
