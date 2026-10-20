"""Atomic artifact writes
failures never truncate the previous artifact
"""

import gzip
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    # Retain an incomplete temporary file if replacement fails, for
    # diagnosis
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n")


def read_bytes(path: Path) -> bytes:
    """Read original bytes from either a plain or gzip artifact"""
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as stream:
                return stream.read()
        except (EOFError, zlib.error) as error:
            raise OSError(f"invalid gzip artifact: {path}") from error
    return path.read_bytes()


def raw_paths(directory: Path) -> list[Path]:
    """Prefer plain files during interrupted compression
    count each cache once
    """
    paths = {path.name.removesuffix(".gz"): path for path in directory.glob("*.json.gz")}
    paths.update({path.name: path for path in directory.glob("*.json")})
    return [paths[name] for name in sorted(paths)]


def atomic_gzip(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6
        ) as compressed:
            compressed.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def compress_raw_file(path: Path) -> tuple[int, int]:
    """Verify lossless compression before removing an explicitly
    selected
    original
    """
    content = path.read_bytes()
    target = path.with_suffix(path.suffix + ".gz")
    if not target.exists():
        atomic_gzip(target, content)
    if read_bytes(target) != content:
        raise ValueError(f"compressed bytes differ: {target}")
    if path.read_bytes() != content:
        raise ValueError(f"original changed during compression: {path}")
    sizes = (len(content), target.stat().st_size)
    path.unlink()
    return sizes
