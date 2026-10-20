import pytest
from test_draft_parsing import make_match

from lastpick.config import Config
from lastpick.data.io import atomic_gzip, atomic_json, compress_raw_file, raw_paths, read_bytes
from lastpick.data.pipeline import normalize_directory


def test_compression_preserves_normalization_and_digest(tmp_path):
    config = Config()
    path = tmp_path / config.paths.raw / "123.json"
    payload, _, _ = make_match(8)
    atomic_json(path, payload)
    original = path.read_bytes()
    report = normalize_directory(tmp_path, config)
    compress_raw_file(path)
    compressed = path.with_suffix(".json.gz")
    assert not path.exists()
    assert read_bytes(compressed) == original
    assert normalize_directory(tmp_path, config) == report
    # Interrupted conversion counts a plain/compressed pair only once
    path.write_bytes(original)
    assert raw_paths(path.parent) == [path]
    compress_raw_file(path)
    assert raw_paths(path.parent) == [compressed]


def test_conflicting_compressed_copy_preserves_original(tmp_path):
    path = tmp_path / "123.json"
    path.write_bytes(b"original")
    atomic_gzip(path.with_suffix(".json.gz"), b"different")
    with pytest.raises(ValueError, match="compressed bytes differ"):
        compress_raw_file(path)
    assert path.read_bytes() == b"original"


def test_corrupt_compressed_copy_preserves_original(tmp_path):
    path = tmp_path / "123.json"
    path.write_bytes(b"original")
    path.with_suffix(".json.gz").write_bytes(b"broken")
    with pytest.raises(OSError):
        compress_raw_file(path)
    assert path.read_bytes() == b"original"
