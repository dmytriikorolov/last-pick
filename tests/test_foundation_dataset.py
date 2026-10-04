import json
from dataclasses import replace

import pytest
from test_draft_parsing import make_match
from test_foundation_drafts import canonical

from lastpick.config import Config, load_config
from lastpick.data.dataset import build_dataset, decisions_for_match
from lastpick.data.io import atomic_json
from lastpick.data.pipeline import normalize_directory
from lastpick.data.schemas import DraftEvent, Rejection, decision_from_record, record


def test_4v4_labels_and_metadata():
    match, _, _ = canonical(both_teams=True)
    for winner in (True, False):
        decisions, _ = decisions_for_match(replace(match, radiant_win=winner), Config())
        assert len(decisions) == 2
        for d in decisions:
            assert len(d.visible_allies) == len(d.visible_enemies) == 4
            assert len(set(d.visible_allies + d.visible_enemies + (d.chosen_hero,))) == 9
            assert d.outcome == (winner if d.side == 0 else not winner)
            assert d.patch == 60 and d.rank_bracket == "immortal" and d.region == 3
            assert d.start_time == 1000 and d.avg_rank_tier == 80
            assert d.team_pick_number == 5 and d.phase == 3
            assert decision_from_record(json.loads(json.dumps(record(d)))) == d


def make_corpus(tmp_path):
    config = Config()
    normalized = tmp_path / config.paths.normalized
    filenames = []
    for i, order in enumerate((4, 6, 8)):
        match, _, _ = canonical(order)
        match = replace(
            match, match_id=100 + i, events=match.events + (DraftEvent(10, 0, 99, False),)
        )
        name = f"{match.match_id}.json"
        atomic_json(normalized / name, record(match))
        filenames.append(name)
    atomic_json(normalized / "index.json", {"files": filenames, "raw_matches": 3})
    return config


def test_deterministic_artifacts_separate_contexts_and_complete_manifest(tmp_path):
    config = make_corpus(tmp_path)
    first = build_dataset(tmp_path, config, "fixed")
    second = build_dataset(tmp_path, config, "fixed")
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    general = [
        json.loads(line) for line in (first / "carry_decisions.jsonl").read_text().splitlines()
    ]
    lastpick = [
        json.loads(line) for line in (first / "last_pick_decisions.jsonl").read_text().splitlines()
    ]
    assert len(general) == 3 and len(lastpick) == 1
    assert {d["team_pick_number"] for d in general} == {3, 4, 5}
    assert lastpick[0]["unlocated_bans"] == [99]
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["counts_by_team_pick"] == {"3": 1, "4": 1, "5": 1}
    assert manifest["decisions"] == 3 and manifest["last_pick_decisions"] == 1
    assert manifest["schema_version"] == "2.0" and manifest["config"]["seed"] == 42
    assert manifest["role_confidence_distribution"] == {"high": 3}
    assert len(manifest["artifact_sha256"]) == 5
    assert (
        json.loads((tmp_path / config.paths.output / "latest.json").read_text())["build"]
        == second.name
    )


def test_duplicate_canonical_match_does_not_duplicate_decisions(tmp_path):
    config = make_corpus(tmp_path)
    directory = tmp_path / config.paths.normalized
    duplicate = json.loads((directory / "100.json").read_text())
    atomic_json(directory / "duplicate.json", duplicate)
    index = json.loads((directory / "index.json").read_text())
    atomic_json(directory / "index.json", dict(index, files=index["files"] + ["duplicate.json"]))
    output = build_dataset(tmp_path, config)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["decisions"] == 3
    assert manifest["rejection_reasons"] == {"duplicate_match_id": 1}


def test_normalization_preserves_discovery_rank_without_inventing_mmr(tmp_path):
    config = Config()
    raw, _, _ = make_match(8)
    atomic_json(tmp_path / config.paths.raw / "123.json", raw)
    atomic_json(
        tmp_path / config.paths.ingestion / "metadata/123.json",
        {"discovery": {"avg_rank_tier": 75, "num_rank_tier": 8}},
    )
    report = normalize_directory(tmp_path, config)
    assert report["accepted"] == 1
    saved = json.loads((tmp_path / config.paths.normalized / "123.json").read_text())
    assert saved["avg_rank_tier"] == 75 and saved["rank_bracket"] == "divine"
    assert saved["rank_source"] == "discovery" and saved["region"] is None
    assert "mmr" not in saved


def test_raw_duplicates_and_invalid_json_reported(tmp_path):
    config = Config()
    raw, _, _ = make_match(8)
    directory = tmp_path / config.paths.raw
    atomic_json(directory / "a.json", raw)
    atomic_json(directory / "b.json", raw)
    (directory / "c.json").write_text("broken")
    report = normalize_directory(tmp_path, config)
    assert report["raw_matches"] == 3 and report["accepted"] == 1
    assert report["rejection_reasons"] == {"duplicate_match_id": 1, "invalid_json_or_metadata": 1}


def test_missing_rank_is_explicit_and_can_be_included_by_policy():
    match, _, _ = canonical()
    match = replace(match, avg_rank_tier=None, rank_bracket=None, num_rank_tier=None)
    with pytest.raises(Rejection, match="missing_rank"):
        decisions_for_match(match, Config())
    config = replace(Config(), filters=replace(Config().filters, require_rank=False))
    decisions, _ = decisions_for_match(match, config)
    assert decisions[0].avg_rank_tier is None and decisions[0].rank_bracket is None


def test_config_defaults_and_unknown_keys_are_validated(tmp_path):
    from pathlib import Path

    assert load_config(Path("config/default.toml")) == Config()
    path = tmp_path / "invalid.toml"
    path.write_text("[roles]\nunknown_setting = 1\n")
    with pytest.raises(TypeError):
        load_config(path)


def test_zero_decision_build_keeps_csv_headers_and_rejection_counts(tmp_path):
    config = make_corpus(tmp_path)
    config = replace(config, roles=replace(config.roles, min_confidence=1.0))
    output = build_dataset(tmp_path, config)
    csv = (output / "last_pick_decisions.csv").read_text()
    assert len(csv.splitlines()) == 1 and "candidate_hero_id" in csv
    manifest = json.loads((output / "manifest.json").read_text())
    assert (
        manifest["decisions"] == 0
        and manifest["decision_rejection_reasons"]["below_role_confidence"] == 3
    )
