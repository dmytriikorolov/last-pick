# LastPick

A Dota 2 position-1 last-pick recommender. The credited ISP model/web demo is
preserved at `isp-v1`; Foundation v2 adds canonical data, draft/role inference,
reproducible 4v4 datasets and explicit temporal feature safety.

```sh
python -m pip install -e '.[dev,web]'
python -m web.app
python -m pytest -q
```

Read [documentation](docs/DOCUMENTATION.md),
[Foundation v2 architecture](docs/FOUNDATION_V2.md) and
[ISP reproduction notes](docs/ISP_V1.md). New data/model artifacts stay separate
from the original ISP result. No new production model was trained for Foundation v2.

Raw match caches support both `.json` and `.json.gz`; new collector downloads use
lossless gzip. To reclaim space from existing caches, run:

```sh
python -m scripts.compress_raw_matches
```

Each compressed copy is read back and compared byte for byte before its original
is removed. Interrupted runs can be resumed. Normalization digests use original
bytes and filenames, so compression preserves provenance. Canonical datasets and
ISP models are unchanged. Legacy ISP CSV extractors require plain JSON: restore
raw files before using them (allow enough disk space for the expanded files):

```sh
find data/raw/opendota/matches -name '*.json.gz' -exec gunzip {} +
```
