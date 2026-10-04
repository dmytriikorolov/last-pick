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
