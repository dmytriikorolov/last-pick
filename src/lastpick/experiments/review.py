# Sample team cases for independent role review
# Keep sampling repeatable and leave human labels blank

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from lastpick.config import Config
from lastpick.data.io import atomic_json
from lastpick.data.schemas import Rejection
from lastpick.data.validation import validate_filters
from lastpick.experiments.audit import canonical_records
from lastpick.roles.inference import make_inferer


def role_review_pilot(source: Path, output: Path) -> dict[str, Any]:
    """Select distinct matches and hide the heuristic and its input telemetry."""
    fields = (
        "match_id",
        "side",
        "patch",
        "rank_bracket",
        "hero_ids",
        "reviewed_pos1_slot",
        "reviewer_confidence",
        "notes",
    )
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected: dict[str, dict[str, str]] = {}
    for bucket in ("abstention", "medium", "high"):
        count = 0
        for row in rows:
            match_id = row["match_id"]
            if row["stratum"].split(":")[0] == bucket and match_id not in selected:
                selected[match_id] = row
                count += 1
                if count == 10:
                    break
        if count < 10:
            raise ValueError(f"fewer than ten distinct matches available for {bucket}")

    # Mix confidence groups without revealing how rows were selected.
    ordered = sorted(selected, key=lambda mid: hashlib.sha256(mid.encode()).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for match_id in ordered:
            row = selected[match_id]
            writer.writerow(
                {
                    **{field: row[field] for field in fields[:5]},
                    "reviewed_pos1_slot": "",
                    "reviewer_confidence": "",
                    "notes": "",
                }
            )
    return {"output": str(output), "cases": len(ordered), "reviewers": 2}


def role_review_sample(
    canonical: Path, output: Path, config: Config, size: int = 400, seed: int = 42
) -> dict[str, Any]:
    if size < 1:
        raise ValueError("sample size must be positive")
    inferer = make_inferer(config.roles)
    strata: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    per_cell = max(10, size // 3 + 1)
    for match in canonical_records(canonical):
        try:
            validate_filters(match, config.filters)
        except Rejection:
            continue
        for side in (0, 1):
            prediction = inferer.predict(match, side)
            team = [p for p in match.players if p.side == side]
            safelane = [p for p in team if p.lane_role == 1]
            bucket = (
                "abstention"
                if prediction.hero_id is None
                else "high"
                if prediction.confidence >= 0.85
                else "medium"
            )
            difficult = len(safelane) != 2 or prediction.reason == "ambiguous_pos1"
            stratum = f"{bucket}:{int(difficult)}:{match.patch}:{match.rank_bracket}:{prediction.hero_id}:{prediction.reason or 'accepted'}"
            counts[stratum] += 1
            key = hashlib.sha256(f"{seed}:{match.match_id}:{side}".encode()).hexdigest()
            row: dict[str, Any] = {
                "match_id": match.match_id,
                "side": side,
                "patch": match.patch,
                "rank_bracket": match.rank_bracket,
                "avg_rank_tier": match.avg_rank_tier,
                "stratum": stratum,
                "hero_ids": json.dumps({p.slot: p.hero_id for p in team}, sort_keys=True),
                "lane_roles": json.dumps({p.slot: p.lane_role for p in team}, sort_keys=True),
                "position_est": json.dumps({p.slot: p.position_est for p in team}, sort_keys=True),
                "heuristic_pos1_slot": prediction.player_slot,
                "heuristic_hero": prediction.hero_id,
                "confidence": prediction.confidence,
                "method": prediction.method,
                "abstention_reason": prediction.reason,
                "difficult_safelane": difficult,
                "gpm": json.dumps({p.slot: p.gpm for p in team}, sort_keys=True),
                "xpm": json.dumps({p.slot: p.xpm for p in team}, sort_keys=True),
                "evidence": json.dumps(prediction.evidence, sort_keys=True),
                "reviewed_pos1_slot": "",
                "reviewer_confidence": "",
                "notes": "",
            }
            for minute in (5, 10, 15):
                for field in ("lh_t", "gold_t", "xp_t"):
                    row[f"{field}_{minute}"] = json.dumps(
                        {p.slot: p.snapshot(field, minute) for p in team}, sort_keys=True
                    )
            strata[stratum].append((key, row))
            strata[stratum].sort(key=lambda pair: pair[0])
            # retain enough candidates per cell for a balanced sample
            # without holding the corpus
            strata[stratum] = strata[stratum][:per_cell]
    ordered = sorted(strata, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    chosen = []
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for depth in range(per_cell):
        for stratum in ordered:
            if len(strata[stratum]) > depth:
                reservoirs[stratum.split(":")[0]].append(strata[stratum][depth][1])
    # equal primary quotas keep abstentions represented despite their
    # smaller cell count
    buckets = ("high", "medium", "abstention")
    for i, bucket in enumerate(buckets):
        quota = size // 3 + int(i < size % 3)
        chosen.extend(reservoirs[bucket][:quota])
        reservoirs[bucket] = reservoirs[bucket][quota:]
    while len(chosen) < size and any(reservoirs.values()):
        for bucket in buckets:
            if reservoirs[bucket] and len(chosen) < size:
                chosen.append(reservoirs[bucket].pop(0))
    output.parent.mkdir(parents=True, exist_ok=True)
    if not chosen:
        raise ValueError("no eligible review examples")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(chosen[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(chosen)
    report = {
        "seed": seed,
        "sample_size": len(chosen),
        "eligible_team_count": sum(counts.values()),
        "strata_counts": dict(counts),
        "represented_strata": len({r["stratum"] for r in chosen}),
        "bucket_counts": {
            bucket: sum(r["stratum"].startswith(bucket + ":") for r in chosen) for bucket in buckets
        },
        "sampling": "equal high/medium/abstention quotas, then hash-priority round-robin across difficult/patch/rank/hero/reason cells; shortages redistribute; not population-weighted",
        "labels": "Human review columns are intentionally blank. Heuristic confidence is not role ground truth.",
    }
    atomic_json(output.with_suffix(".json"), report)
    return report
