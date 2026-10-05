"""Small orchestration commands; package modules provide the reusable APIs."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from lastpick.config import load_config, project_root
from lastpick.data.dataset import build_dataset
from lastpick.data.io import atomic_json
from lastpick.data.pipeline import normalize_directory
from lastpick.features.asof import AsOfHistory, load_canonical
from lastpick.ingestion.collector import Collector
from lastpick.ingestion.opendota import OpenDotaSource


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--max-pages", type=int)
    commands.add_parser("normalize")
    commands.add_parser("dataset")
    history = commands.add_parser("history")
    history.add_argument("--as-of", type=int, required=True)
    history.add_argument(
        "--training-ids",
        type=Path,
        help="JSON array of training match IDs; required for model evaluation",
    )
    history.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit-dataset")
    audit.add_argument("--build", type=Path)
    audit.add_argument("--output", type=Path, default=Path("reports/modeling-v2/generated"))
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--build", type=Path)
    benchmark.add_argument("--models", nargs="+")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--minimum-candidate-support", type=int, default=100)
    benchmark.add_argument("--bootstrap", type=int, default=200)
    benchmark.add_argument("--train-fraction", type=float, default=0.7)
    benchmark.add_argument("--validation-fraction", type=float, default=0.15)
    benchmark.add_argument("--patch-holdout", action="store_true")
    benchmark.add_argument(
        "--holdout-start", type=int, help="UTC Unix time; reserve all later matches for test"
    )
    benchmark.add_argument("--max-nn-epochs", type=int, default=20)
    benchmark.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--build", type=Path)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--bootstrap", type=int, default=200)
    review = commands.add_parser("role-review-sample")
    review.add_argument("--size", type=int, default=400)
    review.add_argument("--seed", type=int, default=42)
    review.add_argument("--output", type=Path, default=Path("reports/role_review_sample.csv"))
    report_command = commands.add_parser("benchmark-report")
    report_command.add_argument("--run", type=Path, required=True)
    report_command.add_argument("--holdout", type=Path)
    report_command.add_argument("--reevaluation", type=Path)
    report_command.add_argument("--ablations", type=Path)
    report_command.add_argument(
        "--audit", type=Path, default=Path("reports/modeling-v2/generated/audit.json")
    )
    report_command.add_argument("--output", type=Path, default=Path("reports/modeling-v2"))
    report_command.add_argument("--document", type=Path, default=Path("docs/MODELING_V2.md"))
    args = parser.parse_args(argv)
    config_path = args.config or args.root / "config/default.toml"
    config = load_config(config_path if args.config is not None or config_path.exists() else None)
    if args.command in {"ingest", "normalize"} and config.source != "opendota":
        raise ValueError("CLI currently provides only the OpenDota source/adapter")
    if args.command == "ingest":
        collector = Collector(args.root, config)
        collector.source = OpenDotaSource(
            config.ingestion, config.filters, collector.consume_request
        )
        print(json.dumps(collector.run(args.max_pages), indent=2))
    elif args.command == "normalize":
        report = normalize_directory(args.root, config)
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    elif args.command == "dataset":
        path = build_dataset(args.root, config)
        print(f"Dataset build: {path}")
        print((path / "manifest.json").read_text())
    elif args.command == "history":
        allowed = set(json.loads(args.training_ids.read_text())) if args.training_ids else None
        engine = AsOfHistory(
            load_canonical(args.root / config.paths.normalized), allowed, config.features
        )
        snapshot = engine.snapshot(args.as_of)
        value = asdict(snapshot)
        value["missing_availability"] = engine.missing_availability
        value["scope"] = "training_only" if allowed is not None else "descriptive"
        atomic_json(args.output, value)
        print(f"Historical snapshot: {args.output}")
    elif args.command == "audit-dataset":
        from lastpick.experiments.audit import audit_dataset
        from lastpick.experiments.data import resolve_build

        result = audit_dataset(
            resolve_build(args.root, args.build), args.root / config.paths.normalized, args.output
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "raw_matches",
                        "normalized_matches",
                        "accepted_matches",
                        "last_pick_decisions",
                        "time_range",
                    )
                },
                indent=2,
            )
        )
    elif args.command == "benchmark":
        from lastpick.experiments.benchmark import benchmark as run_benchmark
        from lastpick.experiments.data import resolve_build

        run_benchmark(
            args.root,
            resolve_build(args.root, args.build),
            args.root / config.paths.normalized,
            args.output,
            args.models,
            args.seed,
            args.minimum_candidate_support,
            args.bootstrap,
            args.train_fraction,
            args.validation_fraction,
            args.patch_holdout,
            args.max_nn_epochs,
            args.holdout_start,
        )
    elif args.command == "evaluate":
        from lastpick.experiments.benchmark import evaluate_run
        from lastpick.experiments.data import resolve_build

        evaluate_run(
            args.run, resolve_build(args.root, args.build), args.output, repetitions=args.bootstrap
        )
        print(f"Evaluation: {args.output}")
    elif args.command == "role-review-sample":
        from lastpick.experiments.review import role_review_sample

        result = role_review_sample(
            args.root / config.paths.normalized, args.output, config, args.size, args.seed
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("sample_size", "eligible_team_count", "represented_strata")
                },
                indent=2,
            )
        )

    elif args.command == "benchmark-report":
        from lastpick.experiments.reporting import export_report

        export_report(
            args.run,
            args.audit,
            args.output,
            args.document,
            args.holdout,
            args.reevaluation,
            args.ablations,
        )
        print(f"Report: {args.document}")


if __name__ == "__main__":
    main()


def ingest_main(argv: list[str] | None = None) -> None:
    """Historical collector entry points accept global options after the script name."""
    parser = argparse.ArgumentParser(description="Unified OpenDota collector")
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args(argv)
    forwarded = ["--root", str(args.root)]
    if args.config:
        forwarded += ["--config", str(args.config)]
    forwarded += ["ingest"]
    if args.max_pages is not None:
        forwarded += ["--max-pages", str(args.max_pages)]
    main(forwarded)
