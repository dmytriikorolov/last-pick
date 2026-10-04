# ISP carry demo

From the repository root, use a Python environment with the existing PyTorch dependencies:

```sh
python -m pip install -e '.[dev,web]'
python -m web.app
```

Open http://127.0.0.1:5000. Startup loads the final model, snapshot metadata, and
checkpoint candidate pool once. There is no training or dataset modification.

All 127 portraits are cached in `web/static/heroes/`. To reproduce or restore them:

```sh
python -m scripts.download_hero_portraits
```

The downloader uses image paths from the local OpenDota snapshot and skips existing
files. Once cached, the page and recommendations need no internet access.

Select YOUR TEAM or ENEMY TEAM, then left click portraits to fill four slots each.
Click a filled slot to remove it. Right click a portrait to toggle its ban; banning
a selected hero removes it from the draft. Clear draft resets picks and bans.
Example draft fills a sample 4v4. Search filters the full attribute-grouped roster.
The recommendation button enables only for 4v4; scores are estimated probabilities,
not guarantees. If bans leave fewer than five candidates, the API returns those
remaining (or an empty list).

```sh
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"allies":[2,5,25,86],"enemies":[14,26,74,99],"banned":[1]}'
```

Response: `{"recommendations":[{"hero_id":48,"hero_name":"Luna","model_score":0.55},...]}`
(example shape only; actual ranking and scores come from the checkpoint).
Invalid input returns HTTP 400 with an `error` message. IDs must be integers known
to the model. Both teams need exactly four unique heroes with no overlap or banned
picks. Recommendations exclude all picks and bans using the checkpoint pool.

Run integration tests against the actual final checkpoint and cached portraits:

```sh
python -m pytest tests/test_web.py tests/test_recommender.py -q
```

Manual UI check: add a hero, try adding it again, fill a team, switch teams, remove a
slot, toggle bans twice, search by name, use Example draft, and request the Top 5.
Check that changing picks or bans clears previous results and banned heroes never
appear. Resize to a phone width to check stacked team panels.

On an Apple Silicon Mac, if Python runs under Rosetta but PyTorch is arm64, prefix
these Python commands with `arch -arm64`.

## Hosted website

The committed standalone ISP snapshot is in `site/public/`. It uses the same UI
with checkpoint weights and aggregate historical statistics exported for browser
inference, and needs no running Python server. Those exported values are public.

Rebuild a standalone copy from the preserved checkpoint and current UI:

```sh
python -m scripts.export_web_site --output dist
python -m http.server 8000 --directory dist
```

Open http://localhost:8000, choose Example draft, then RECOMMEND CARRY.
Rebuilding locally does not publish a site. Backend tests use the tracked checkpoint;
the repository does not currently include a browser/PyTorch numerical parity test.
The Foundation smoke check exports the site and verifies its artifact structure.

See [Foundation v2](../docs/FOUNDATION_V2.md) for package/data commands. This UI
continues to serve `final_carry_recommender.pt`; Foundation training writes separate
artifacts and is not automatically deployed.
