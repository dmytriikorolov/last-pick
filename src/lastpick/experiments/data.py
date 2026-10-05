# validated experiment contract, with label evidence separated from draft inputs

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lastpick.data.schemas import DraftDecision, Rejection, decision_from_record


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve_build(root: Path, build: Path | None = None) -> Path:
    if build is not None:
        return build if build.is_absolute() else root / build
    directory = root / "data/processed/v2"
    name = str(json.loads((directory / "latest.json").read_text())["build"])
    return directory / name


@dataclass(frozen=True)
class DraftInput:
    candidate: int
    allies: tuple[int, ...]
    enemies: tuple[int, ...]
    bans: tuple[int, ...]
    side: int
    patch: int | None
    rank_bracket: str | None
    avg_rank_tier: float | None
    region: int | None

    def __post_init__(self) -> None:
        heroes = (self.candidate,) + self.allies + self.enemies
        if len(self.allies) != 4 or len(self.enemies) != 4:
            raise Rejection("experiment_requires_4v4")
        if self.side not in (0, 1) or any(h <= 0 for h in heroes):
            raise Rejection("invalid_experiment_draft")
        if len(set(heroes)) != 9 or set(heroes) & set(self.bans):
            raise Rejection("duplicate_or_banned_hero")


def draft_input(decision: DraftDecision, candidate: int | None = None) -> DraftInput:
    # Deliberate allowlist: no outcome, role evidence, telemetry or unlocated bans.
    return DraftInput(
        candidate if candidate is not None else decision.chosen_hero,
        tuple(sorted(decision.visible_allies)),
        tuple(sorted(decision.visible_enemies)),
        tuple(sorted(decision.visible_bans)),
        decision.side,
        decision.patch,
        decision.rank_bracket,
        decision.avg_rank_tier,
        decision.region,
    )


def load_decisions(build: Path, verify: bool = True) -> list[DraftDecision]:
    manifest = json.loads((build / "manifest.json").read_text())
    if manifest["schema_version"] != "2.0":
        raise Rejection("unsupported_experiment_schema")
    if verify:
        for name, expected in manifest["artifact_sha256"].items():
            if digest(build / name) != expected:
                raise Rejection("dataset_digest_mismatch", name)
    rows = []
    seen: set[tuple[int, int]] = set()
    with (build / "last_pick_decisions.jsonl").open() as stream:
        for line in stream:
            row = decision_from_record(json.loads(line))
            draft_input(row)
            if row.start_time is None:
                raise Rejection("missing_decision_time")
            key = (row.match_id, row.side)
            if key in seen:
                raise Rejection("duplicate_experiment_decision")
            seen.add(key)
            rows.append(row)
    if len(rows) != manifest["last_pick_decisions"]:
        raise Rejection("manifest_row_count_mismatch")
    return sorted(rows, key=lambda d: (d.start_time or 0, d.match_id, d.side))


def candidate_pool(
    training: tuple[DraftDecision, ...] | list[DraftDecision], minimum: int = 100
) -> tuple[int, ...]:
    if minimum < 1:
        raise ValueError("minimum candidate support must be positive")
    counts = Counter(d.chosen_hero for d in training)
    return tuple(sorted(h for h, count in counts.items() if count >= minimum))


def evaluation_pool(decision: DraftDecision, pool: tuple[int, ...]) -> tuple[int, ...]:
    excluded = set(decision.visible_allies + decision.visible_enemies + decision.visible_bans)
    return tuple(h for h in pool if h not in excluded)


def dataset_identity(build: Path) -> dict[str, Any]:
    manifest = json.loads((build / "manifest.json").read_text())
    return {
        "build": str(build.resolve()),
        "manifest_sha256": digest(build / "manifest.json"),
        "decisions_sha256": digest(build / "last_pick_decisions.jsonl"),
        "canonical_input_digest": manifest["input_digest"],
        "raw_digest": manifest["normalization"]["raw_digest"],
    }
