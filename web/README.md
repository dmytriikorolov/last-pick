# LastPick website

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

## Draft and My Dota

Choose a team and fill four slots. Use the Pick/Ban switch or right-click a hero
to manage bans. Save and reopen named drafts, or copy a link that shares picks
and bans. Saved drafts and favorites stay in your browser.

Open My Dota for a six-month calendar and recent matches. Try the example or
load a public OpenDota account by ID, OpenDota player link, or numeric Steam profile link. The profile
makes one anonymous OpenDota request per load and caches the most recent
account's results locally for 30 minutes. Loading your history also selects
up to eight supported heroes by matches played in the displayed period, with
recent play breaking ties. The draft filter includes those heroes and your
starred favorites. You can turn it off to see the full model pool.

The recommendation button enables at 4v4 and ranks up to five available heroes
from the ISP checkpoint. The real profile needs an internet connection.

```sh
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"allies":[2,5,25,86],"enemies":[14,26,74,99],"banned":[1]}'
```

Response: `{"recommendations":[{"hero_id":48,"hero_name":"Luna","model_score":0.55},...]}`
(example shape only; actual ranking and scores come from the checkpoint).
Invalid input returns HTTP 400 with an `error` message. IDs must be integers known
to the model. Both teams need exactly four unique heroes with no overlap or banned
picks. Recommendations exclude all picks and bans from the checkpoint pool.
An optional `candidate_pool` array restricts the ranking to those IDs.

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
Rebuilding locally does not publish a site. The test suite compares the static
model with Flask, including the optional carry pool, when Node.js is installed.

See [Foundation v2](../docs/FOUNDATION_V2.md) for package/data commands. This UI
continues to serve `final_carry_recommender.pt`; Foundation training writes separate
artifacts and is not automatically deployed.
