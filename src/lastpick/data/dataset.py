"""Deterministic general carry and dedicated 4v4 decision artifacts."""

import csv
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lastpick import __version__
from lastpick.config import Config
from lastpick.data.io import atomic_json
from lastpick.data.schemas import (
    SCHEMA_VERSION,
    DraftDecision,
    Match,
    Rejection,
    match_from_record,
    record,
)
from lastpick.data.validation import validate_filters, validate_integrity
from lastpick.drafts.reconstruction import reconstruct, visible_picks
from lastpick.roles.inference import RoleInferer, make_inferer

CSV_COLUMNS = [
    "match_id",
    "source",
    "side",
    "candidate_hero_id",
    "candidate_player_slot",
    "candidate_team_pick_number",
    "candidate_global_pick_order",
    "phase",
    "known_ally_count",
    "known_enemy_count",
    "pos1_confidence",
    "role_method",
    "patch",
    "rank_bracket",
    "avg_rank_tier",
    "region",
    "start_time",
    "win",
    "visible_bans",
    "unlocated_bans",
] + [f"{group}_{i}" for group in ("ally", "enemy") for i in range(1, 5)]


def decisions_for_match(
    match: Match, config: Config, inferer: RoleInferer | None = None
) -> tuple[list[DraftDecision], Counter[str]]:
    validate_integrity(match)
    if match.source != config.source:
        raise Rejection("source_mismatch")
    validate_filters(match, config.filters)
    draft = reconstruct(match)
    inferer = inferer or make_inferer(config.roles)
    decisions = []
    rejected: Counter[str] = Counter()
    for side in (0, 1):
        role = inferer.predict(match, side)
        if role.predicted_role != 1 or role.hero_id is None:
            rejected[role.reason or "role_abstained"] += 1
            continue
        if role.confidence < config.roles.min_confidence:
            rejected["below_role_confidence"] += 1
            continue
        pick = next(e for e in draft.picks if e.hero_id == role.hero_id and e.team == side)
        allies, enemies = visible_picks(draft, pick)
        number = len(allies) + 1
        if number not in config.dataset.picks:
            rejected[f"pos1_pick_{number}"] += 1
            continue
        phase = pick.phase
        assert phase is not None
        if config.dataset.exclude_candidate_phase_attempts and any(
            e.is_pick and not e.successful and e.phase == phase for e in draft.events
        ):
            rejected["candidate_phase_extra_attempt"] += 1
            continue
        bans = tuple(
            sorted(
                {
                    e.hero_id
                    for e in draft.events
                    if not e.is_pick and e.phase is not None and e.phase < phase
                }
            )
        )
        unlocated = tuple(
            sorted({e.hero_id for e in draft.events if not e.is_pick and e.phase is None})
        )
        # Simultaneous enemy choices are hidden: exact legal universe is unknowable
        # unless a source supplies legality at decision time. Never exclude hidden heroes.
        visible = allies + enemies + (role.hero_id,)
        if len(set(visible)) != len(visible) or set(visible) & set(bans):
            raise Rejection("invalid_visible_heroes")
        decisions.append(
            DraftDecision(
                match_id=match.match_id,
                source=match.source,
                side=side,
                phase=phase,
                team_pick_number=number,
                event_order=pick.order,
                visible_allies=allies,
                visible_enemies=enemies,
                visible_bans=bans,
                unlocated_bans=unlocated,
                chosen_hero=role.hero_id,
                role=role,
                patch=match.patch,
                rank_bracket=match.rank_bracket,
                avg_rank_tier=match.avg_rank_tier,
                region=match.region,
                start_time=match.start_time,
                outcome=bool(match.radiant_win) if side == 0 else not bool(match.radiant_win),
                candidate_universe=config.dataset.hero_universe or None,
            )
        )
    return decisions, rejected


def csv_row(decision: DraftDecision) -> dict[str, Any]:
    row: dict[str, Any] = {
        "match_id": decision.match_id,
        "source": decision.source,
        "side": "radiant" if decision.side == 0 else "dire",
        "candidate_hero_id": decision.chosen_hero,
        "candidate_player_slot": decision.role.player_slot,
        "candidate_team_pick_number": decision.team_pick_number,
        "candidate_global_pick_order": decision.event_order,
        "phase": decision.phase,
        "known_ally_count": len(decision.visible_allies),
        "known_enemy_count": len(decision.visible_enemies),
        "pos1_confidence": decision.role.confidence,
        "role_method": decision.role.method,
        "patch": decision.patch,
        "rank_bracket": decision.rank_bracket,
        "avg_rank_tier": decision.avg_rank_tier,
        "region": decision.region,
        "start_time": decision.start_time,
        "win": int(decision.outcome),
        "visible_bans": json.dumps(decision.visible_bans),
        "unlocated_bans": json.dumps(decision.unlocated_bans),
    }
    for group, heroes in (("ally", decision.visible_allies), ("enemy", decision.visible_enemies)):
        for i in range(4):
            row[f"{group}_{i + 1}"] = heroes[i] if i < len(heroes) else None
    return row


def build_dataset(root: Path, config: Config, build_timestamp: str | None = None) -> Path:
    source = root / config.paths.normalized
    index = json.loads((source / "index.json").read_text())
    output = root / config.paths.output
    output.mkdir(parents=True, exist_ok=True)
    # Every build is retained. latest.json is published only after its manifest is durable.
    generation = Path(tempfile.mkdtemp(prefix="build-", dir=output))
    rejected: Counter[str] = Counter()
    decision_reasons: Counter[str] = Counter()
    phases: Counter[int] = Counter()
    patches: Counter[str] = Counter()
    ranks: Counter[str] = Counter()
    confidences: Counter[str] = Counter()
    seen: set[int] = set()
    accepted = 0
    lastpick_count = 0
    inferer = make_inferer(config.roles)
    inputs = hashlib.sha256()
    from contextlib import ExitStack

    with ExitStack() as stack:
        general = stack.enter_context((generation / "carry_decisions.jsonl").open("w"))
        lastpick = stack.enter_context((generation / "last_pick_decisions.jsonl").open("w"))
        reports = stack.enter_context((generation / "rejections.jsonl").open("w"))
        csv_general = stack.enter_context(
            (generation / "carry_decisions.csv").open("w", newline="")
        )
        csv_last = stack.enter_context(
            (generation / "last_pick_decisions.csv").open("w", newline="")
        )
        writers = [
            csv.DictWriter(stream, fieldnames=CSV_COLUMNS) for stream in (csv_general, csv_last)
        ]
        for writer in writers:
            writer.writeheader()
        for filename in sorted(index["files"]):
            try:
                content = (source / filename).read_bytes()
                inputs.update(filename.encode() + hashlib.sha256(content).digest())
                match = match_from_record(json.loads(content))
                if match.match_id in seen:
                    raise Rejection("duplicate_match_id")
                seen.add(match.match_id)
                decisions, reasons = decisions_for_match(match, config, inferer)
            except (Rejection, OSError, ValueError, TypeError, KeyError) as error:
                reason = (
                    error.reason if isinstance(error, Rejection) else "invalid_canonical_record"
                )
                rejected[reason] += 1
                reports.write(
                    json.dumps({"file": filename, "stage": "match", "reason": reason}) + "\n"
                )
                continue
            accepted += 1
            decision_reasons.update(reasons)
            for reason, count in sorted(reasons.items()):
                reports.write(
                    json.dumps(
                        {
                            "match_id": match.match_id,
                            "stage": "decision",
                            "reason": reason,
                            "count": count,
                        }
                    )
                    + "\n"
                )
            for decision in decisions:
                text = json.dumps(record(decision), sort_keys=True, allow_nan=False) + "\n"
                general.write(text)
                row = csv_row(decision)
                writers[0].writerow(row)
                if len(decision.visible_allies) == len(decision.visible_enemies) == 4:
                    lastpick.write(text)
                    writers[1].writerow(row)
                    lastpick_count += 1
                phases[decision.team_pick_number] += 1
                patches[str(decision.patch) if decision.patch is not None else "unknown"] += 1
                ranks[decision.rank_bracket or "unknown"] += 1
                bucket = (
                    "high"
                    if decision.role.confidence >= 0.85
                    else "medium"
                    if decision.role.confidence >= 0.65
                    else "low"
                )
                confidences[bucket] += 1
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generation.iterdir())
    }
    manifest = {
        "implementation_version": __version__,
        "python_version": platform.python_version(),
        "dataset_version": config.dataset.version,
        "schema_version": SCHEMA_VERSION,
        "source": config.source,
        "build_timestamp": build_timestamp or datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "raw_matches": index["raw_matches"],
        "normalization": {k: v for k, v in index.items() if k != "files"},
        "canonical_matches": len(index["files"]),
        "accepted_matches": accepted,
        "rejected_matches": sum(rejected.values()),
        "rejection_reasons": dict(rejected),
        "decision_rejection_reasons": dict(decision_reasons),
        "decisions": sum(phases.values()),
        "last_pick_decisions": lastpick_count,
        "counts_by_team_pick": dict(phases),
        "patch_distribution": dict(patches),
        "rank_distribution": dict(ranks),
        "role_confidence_distribution": dict(confidences),
        "input_digest": inputs.hexdigest(),
        "artifact_sha256": checksums,
    }
    atomic_json(generation / "manifest.json", manifest)
    atomic_json(output / "latest.json", {"build": generation.name, "manifest": manifest})
    return generation
