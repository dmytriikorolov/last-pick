"""Atomic artifact writes; failures never truncate the previous artifact."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    # Retain an incomplete temporary file if replacement fails, for diagnosis.
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n")
