"""Compare the same models on consecutive UTC weeks."""

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lastpick.data.io import atomic_json
from lastpick.experiments.data import load_decisions

DAY = 86_400
WEEK = 7 * DAY
MODELS = ["global", "hero_side", "lr_context", "catboost_full"]


def summarize_week(run: Path) -> dict[str, Any]:
    models = json.loads((run / "metrics.json").read_text())
    calibrated = json.loads((run / "sigmoid_calibration.json").read_text())
    results = {}
    for name, entry in models.items():
        if entry["status"] != "complete":
            results[name] = entry
            continue
        test = entry["partitions"]["test"]
        ranking = test["ranking"]
        results[name] = {
            **test["metrics"],
            "picked_hero_coverage": ranking["coverage"],
            "most_recommended_hero_percent": max(ranking["top1_percent"].values(), default=0),
            "unique_top_recommendations": ranking["top1_unique_heroes"],
        }
    return {
        "models": results,
        "calibrated_model": calibrated["base_model"],
        "calibrated_metrics": calibrated["test"]["metrics"],
    }


def rolling_benchmark(
    root: Path, build: Path, canonical: Path, output: Path, include_final: bool
) -> None:
    from lastpick.experiments.benchmark import InsufficientBenchmarkData, benchmark

    rows = load_decisions(build)
    if not rows:
        raise ValueError("rolling benchmark requires last-pick decisions")
    daily: Counter[int] = Counter()
    patches: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        assert row.start_time is not None
        day = row.start_time // DAY
        daily[day] += 1
        patches[day - (day + 3) % 7][str(row.patch)] += 1
    first = min(daily) - (min(daily) + 3) % 7
    last = max(daily) - (max(daily) + 3) % 7
    weeks: list[dict[str, Any]] = []
    eligible: list[int] = []
    for week in range(first, last + 1, 7):
        counts = [daily[d] for d in range(week, week + 7)]
        validation = [daily[d] for d in range(week - 7, week)]
        recent = sum(daily[d] for d in range(week - 35, week - 7))
        reason = None
        if not all(counts) or sum(counts) < 1_000:
            reason = "incomplete_test_week"
        elif not all(validation) or sum(validation) < 1_000:
            reason = "incomplete_validation_week"
        elif recent < 3_000:
            reason = "insufficient_recent_training"
        else:
            eligible.append(week)
        weeks.append(
            {
                "week": datetime.fromtimestamp(week * DAY, UTC).date().isoformat(),
                "daily_rows": counts,
                "patch_counts": dict(patches[week]),
                "validation_rows": sum(validation),
                "recent_training_rows": recent,
                "skip_reason": reason,
            }
        )

    output.mkdir(parents=True)
    reserved = eligible[-1] if eligible else None
    runs: list[str] = []
    skipped_runs: dict[str, str] = {}
    test_results: dict[str, dict[str, Any]] = {}
    report = {
        "models": MODELS,
        "weeks": weeks,
        "eligible_weeks": len(eligible),
        "reserved_week": datetime.fromtimestamp(reserved * DAY, UTC).date().isoformat()
        if reserved is not None
        else None,
        "final_requested": include_final,
        "runs": runs,
        "skipped_runs": skipped_runs,
        "test_results": test_results,
    }
    atomic_json(output / "coverage.json", report)
    for week in eligible:
        if week == reserved and not include_final:
            continue
        test_start = week * DAY
        validation_start = test_start - WEEK
        name = datetime.fromtimestamp(test_start, UTC).date().isoformat()
        for training, start in (("all", None), ("recent", validation_start - 4 * WEEK)):
            run = output / f"{name}-{training}"
            try:
                benchmark(
                    root,
                    build,
                    canonical,
                    run,
                    models=MODELS,
                    window=(start, validation_start, test_start, test_start + WEEK),
                )
            except InsufficientBenchmarkData as error:
                skipped_runs[run.name] = str(error)
            else:
                runs.append(run.name)
                test_results[run.name] = summarize_week(run)
            atomic_json(output / "coverage.json", report)
