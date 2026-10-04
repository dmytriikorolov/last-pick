"""Normalize raw source data once; all v2 downstream stages read canonical records."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lastpick.config import Config
from lastpick.data.adapter import MatchAdapter
from lastpick.data.io import atomic_json
from lastpick.data.opendota import OpenDotaAdapter
from lastpick.data.schemas import SCHEMA_VERSION, Rejection, record
from lastpick.data.validation import validate_integrity


def normalize_directory(
    root: Path, config: Config, adapter: MatchAdapter | None = None
) -> dict[str, Any]:
    if adapter is None and config.source != "opendota":
        raise ValueError("an adapter is required for this source")
    adapter = adapter or OpenDotaAdapter()
    if adapter.name != config.source:
        raise ValueError("adapter/source mismatch")
    raw_dir = root / config.paths.raw
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    output = root / config.paths.normalized
    metadata_dir = root / config.paths.ingestion / "metadata"
    accepted = []
    reasons: Counter[str] = Counter()
    rejections = []
    seen: set[int] = set()
    digest = hashlib.sha256()
    count = 0
    for path in sorted(raw_dir.glob("*.json")):
        count += 1
        try:
            content = path.read_bytes()
            digest.update(path.name.encode() + hashlib.sha256(content).digest())
            payload = json.loads(content)
            discovery = None
            identifier = adapter.identifier(payload)
            if identifier is not None:
                meta_path = metadata_dir / f"{identifier}.json"
                if meta_path.exists():
                    metadata_content = meta_path.read_bytes()
                    digest.update(hashlib.sha256(metadata_content).digest())
                    discovery = json.loads(metadata_content).get("discovery")
            match = adapter.normalize(payload, discovery)
            validate_integrity(match)
            if match.match_id in seen:
                raise Rejection("duplicate_match_id")
            seen.add(match.match_id)
            target = output / f"{match.match_id}.json"
            value = record(match)
            # Avoid rewriting unchanged canonical data on rerun.
            expected = json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n"
            if not target.exists() or target.read_text() != expected:
                atomic_json(target, value)
            accepted.append(target.name)
        except Rejection as error:
            reasons[error.reason] += 1
            rejections.append({"file": path.name, "reason": error.reason})
        except (OSError, ValueError, TypeError) as error:
            reasons["invalid_json_or_metadata"] += 1
            rejections.append(
                {"file": path.name, "reason": "invalid_json_or_metadata", "detail": str(error)}
            )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": config.source,
        "raw_matches": count,
        "accepted": len(accepted),
        "rejected": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "raw_digest": digest.hexdigest(),
        "files": sorted(accepted),
    }
    atomic_json(output / "normalization_rejections.json", rejections)
    # Authoritative input index, committed after every accepted record is durable.
    atomic_json(output / "index.json", report)
    return report
