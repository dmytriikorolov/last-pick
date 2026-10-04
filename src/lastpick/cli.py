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
    args = parser.parse_args(argv)
    config_path = args.config or args.root / "config/default.toml"
    config = load_config(config_path)
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
