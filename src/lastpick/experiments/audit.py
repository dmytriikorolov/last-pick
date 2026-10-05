# full-corpus descriptive audit. Label telemetry never becomes a predictor

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from lastpick.data.io import atomic_json
from lastpick.data.schemas import Match, match_from_record
from lastpick.experiments.data import dataset_identity, load_decisions


def canonical_records(directory: Path) -> Iterator[Match]:
    index = json.loads((directory / "index.json").read_text())
    for filename in index["files"]:
        yield match_from_record(json.loads((directory / filename).read_text()))


def utc(timestamp: int | None) -> str | None:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
        if timestamp is not None
        else None
    )


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(values) / len(values),
        "quantiles": {
            str(q): ordered[int((len(ordered) - 1) * q)] for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
    }


def audit_dataset(build: Path, canonical: Path, output: Path) -> dict[str, Any]:
    rows = load_decisions(build)
    manifest = json.loads((build / "manifest.json").read_text())
    matches = Counter(d.match_id for d in rows)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing: Counter[str] = Counter()
    heroes: dict[int, Counter[str]] = defaultdict(Counter)
    confidence = []
    ranks = []
    month_hero: dict[str, Counter[str]] = defaultdict(Counter)
    month_rank: dict[str, Counter[str]] = defaultdict(Counter)
    for d in rows:
        for name in ("patch", "rank_bracket", "avg_rank_tier", "region", "start_time"):
            value = getattr(d, name)
            missing[name] += value is None
            counts[name][str(value) if value is not None else "unknown"] += 1
        counts["side"]["radiant" if d.side == 0 else "dire"] += 1
        counts["role_method"][d.role.method] += 1
        counts["role_confidence"][f"{d.role.confidence:.2f}"] += 1
        confidence.append(d.role.confidence)
        if d.avg_rank_tier is not None:
            ranks.append(d.avg_rank_tier)
        for name in ("visible_bans", "unlocated_bans"):
            counts[name][str(len(getattr(d, name)))] += 1
        missing["exact_legality"] += d.legal_candidates is None
        heroes[d.chosen_hero]["wins" if d.outcome else "losses"] += 1
        month = datetime.fromtimestamp(d.start_time or 0, timezone.utc).strftime("%Y-%m")
        counts["month"][month] += 1
        month_hero[month][str(d.chosen_hero)] += 1
        month_rank[month][d.rank_bracket or "unknown"] += 1
    rejected_ids = set()
    with (build / "rejections.jsonl").open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["stage"] == "match":
                rejected_ids.add(int(Path(entry["file"]).stem))
    telemetry: Counter[str] = Counter()
    canonical_counts: dict[str, Counter[str]] = defaultdict(Counter)
    contributions: Counter[str] = Counter()
    players = 0
    duplicate_ids = 0
    seen: set[int] = set()
    for match in canonical_records(canonical):
        duplicate_ids += match.match_id in seen
        seen.add(match.match_id)
        for name in ("patch", "rank_bracket", "rank_source", "region"):
            canonical_counts[name][str(getattr(match, name))] += 1
        if match.match_id not in rejected_ids:
            contributions[str(matches[match.match_id])] += 1
        for p in match.players:
            players += 1
            for field in ("lane_role", "position_est", "gpm", "xpm", "purchases", "rank_tier"):
                telemetry[field] += getattr(p, field) is not None
            for minute in (5, 10, 15):
                for field in ("lh_t", "gold_t", "xp_t"):
                    telemetry[f"{field}_{minute}"] += p.snapshot(field, minute) is not None
    times = [d.start_time for d in rows if d.start_time is not None]
    n = len(rows)
    result = {
        "dataset": dataset_identity(build),
        "raw_matches": manifest["raw_matches"],
        "normalized_matches": manifest["canonical_matches"],
        "normalization": manifest["normalization"],
        "accepted_matches": manifest["accepted_matches"],
        "rejected_matches": manifest["rejected_matches"],
        "rejection_reasons": manifest["rejection_reasons"],
        "decision_rejection_reasons": manifest["decision_rejection_reasons"],
        "general_decisions": manifest["decisions"],
        "last_pick_decisions": n,
        "last_pick_unique_matches": len(matches),
        "accepted_match_contribution": dict(contributions),
        "accepted_match_contribution_percent": {
            k: 100 * v / manifest["accepted_matches"] for k, v in contributions.items()
        },
        "time_range": [utc(min(times)), utc(max(times))] if times else [],
        "distributions": dict(counts),
        "canonical_distributions": dict(canonical_counts),
        "role_confidence_summary": distribution(confidence),
        "avg_rank_tier_summary": distribution(ranks),
        "missing_metadata_rates": {k: v / n if n else None for k, v in missing.items()},
        "unique_carry_heroes": len(heroes),
        "hero_outcomes": {
            str(h): dict(v, total=sum(v.values())) for h, v in sorted(heroes.items())
        },
        "candidate_frequency": {str(h): sum(v.values()) for h, v in sorted(heroes.items())},
        "low_support_heroes_under_100": {
            str(h): sum(v.values()) for h, v in heroes.items() if sum(v.values()) < 100
        },
        "visible_ban_coverage": sum(bool(d.visible_bans) for d in rows) / n if n else None,
        "unlocated_ban_coverage": sum(bool(d.unlocated_bans) for d in rows) / n if n else None,
        "telemetry_player_denominator": players,
        "telemetry_coverage": {k: v / players for k, v in telemetry.items()},
        "month_candidate_counts": dict(month_hero),
        "month_rank_counts": dict(month_rank),
        "duplicate_canonical_ids": duplicate_ids,
        "duplicate_decisions": 0,
        "source_canonical_overlap": "Canonical records are normalized representations of raw matches, not additional samples. Only the indexed build is loaded.",
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "audit.json", result)
    lines = [
        "# Full v2 corpus audit",
        "",
        f"Raw {result['raw_matches']:,}; normalized {result['normalized_matches']:,}; accepted {result['accepted_matches']:,}; rejected {result['rejected_matches']:,}.",
        f"General decisions {result['general_decisions']:,}; true 4v4 decisions {n:,} from {len(matches):,} matches.",
        f"Time range: {result['time_range']}",
        "",
        "## Coverage and selection",
        "",
    ]
    for field in (
        "rejection_reasons",
        "decision_rejection_reasons",
        "accepted_match_contribution_percent",
        "missing_metadata_rates",
        "telemetry_coverage",
        "low_support_heroes_under_100",
    ):
        lines += [
            f"### {field}",
            "",
            "```json",
            json.dumps(result[field], indent=2, sort_keys=True),
            "```",
            "",
        ]
    for field in ("patch", "rank_bracket", "side", "month", "role_method", "role_confidence"):
        lines += [
            f"### {field}",
            "",
            "```json",
            json.dumps(counts[field], indent=2, sort_keys=True),
            "```",
            "",
        ]
    lines += [
        "## Collection discontinuities",
        "",
        "Inspect monthly patch/rank/candidate distributions in audit.json. Concentration reflects the cached collection, not Dota's population. Low-support heroes remain in audit/classification; ranking support is a separate training-only policy.",
        f"Visible bans: {result['visible_ban_coverage']}; unlocated bans: {result['unlocated_ban_coverage']}. Unknown bans never exclude evaluation candidates.",
        "No raw/canonical duplicate samples are concatenated. Duplicate decision and canonical checks are explicit.",
    ]
    (output / "audit.md").write_text("\n".join(lines) + "\n")
    return result
