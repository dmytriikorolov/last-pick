# Foundation v2 architecture and operations

## Scope and compatibility

The credited ISP implementation is tagged `isp-v1` at `7dd54e2`.
See [ISP_V1.md](ISP_V1.md) for dataset/model/results and reproduction requirements.
Its final checkpoint and result JSON, constants, portraits and frontend are retained.
Foundation v2 creates data and evaluation foundations; it does not claim model
improvements, replace the neural architecture or retrain the shipped model.

`scripts/` now exposes thin compatibility entry points. Reusable v1 extraction
and tools live in `lastpick.legacy`; model, recommendation and statistics logic
live in their package areas. Original script module commands still work. Collector
internals were replaced: both collector names invoke one v2 implementation.
The old pinned requirements are available at the baseline tag; current installation
uses the extras in `pyproject.toml`. The Flask API and browser export keep their
existing scoring behavior. Recommendation now enforces bans inside the package API.

## Package flow

```text
ingestion.source.MatchSource             data.adapter.MatchAdapter
  OpenDotaSource -> raw cache/state  ->    OpenDotaAdapter
                                                 |
                                          canonical Match
                                                 |
                                  integrity + selection validation
                                                 |
                           role inference + complete draft reconstruction
                                                 |
                                          DraftDecision
                                                 |
                            general pick-3/4/5 + primary 4v4 datasets
                                                 |
                              temporal splits / training-source allowlist
                                                 |
                             completed-match AsOfHistory + FeatureSpec
```

Provider discovery, transport, source normalization and domain processing are
independent interfaces. Only OpenDota is implemented; another provider needs a
MatchSource and MatchAdapter, not a new draft/dataset pipeline. The CLI selects
OpenDota explicitly. Configuration is frozen dataclasses + TOML, with checked-in
defaults covering paths, mode/lobby/rank/patch/duration filters, role thresholds,
dataset settings, random seed, transport budgets and training/statistics settings.
Unknown TOML keys fail; missing source data never becomes measured zero.

## Schemas

All records serialize to ordinary JSON, schema version `2.0`.

| Record | Key contents |
| --- | --- |
| Match | Source/id/time/patch/duration/mode/lobby/rank and provenance/region/cluster/winner, players, full draft events, ordering semantics |
| Player | Slot/side/hero/lane/position estimate, time/LH/gold/XP arrays, KDA/GPM/XPM/net worth, purchases, final items/backpack, upgrades/leaver/rank |
| DraftEvent | Source order/team/hero/pick-or-ban, recovered successful status, reliable/inferred phase and provenance |
| Draft | Every event plus ordered final played picks |
| RolePrediction | Predicted carry slot/hero/role, confidence, comparative slot scores, evidence, method/version/reason; optional future position probabilities |
| DraftDecision | Side/phase/team-pick/event-order, visible allies/enemies/bans, unknown-timing bans, chosen hero/role, outcome and patch/rank/region/time, optional universe/legality |

Canonical deserializers restore tuple records and integer score keys. Timelines
respect an explicit `times` array when available; otherwise minute indices match
the original OpenDota interpretation. Missing lists, missing entries and explicit
zero measurements remain distinct.

Rank tiers map tens digits 1..8 to Herald, Guardian, Crusader, Archon, Legend,
Ancient, Divine and Immortal. Average rank may come from detail, discovery metadata
or available player rank tiers, with provenance. No exact MMR is synthesized.
Unknown patch/rank/region stays unknown. No cluster-to-region guesses are made.

## Ingestion

```sh
python -m pip install -e '.[dev,web]'
export OPENDOTA_API_KEY=...  # optional, shell environment only
python -m lastpick --config config/default.toml ingest --max-pages 1
```

The default policy uses publicMatches, a daily UTC request budget and verified
HTTPS. `discovery = "explorer"` selects configured Explorer SQL discovery. Each
attempt consumes/persists budget before sending. HTTP 429/5xx and transport
failures have bounded retries; numeric Retry-After is honored. Permanent HTTP
errors and invalid response shapes are distinct. Failure reports do not print
exception URLs containing credentials.

Raw cache filenames remain `{match_id}.json`, so existing ISP downloads are reused.
V2 state lives separately in `data/raw/opendota/foundation_state`: pending page,
cursor, request budget/day, filter fingerprint, discovery metadata, outcomes and
failure reports. Cursor advances only after finishing the page. Failed downloads
stay pending; restart retries them. Invalid payloads/validation rejections are
recorded, with raw responses retained when identifiable. Invalid existing caches
are never overwritten. Policy changes require a new state directory, allowing
revalidation without deleting original data. The target counts raw cached matches,
including rejected matches; manifests count accepted data separately.

Atomic writes flush and fsync temporary files before replacement. State corruption
fails loudly rather than resetting budgets. Interrupted writes retain diagnostic
temporary files. The collector is sequential to make budget/state correctness
simple; high-throughput parallel downloads are deliberately deferred.

## Draft reconstruction and visibility

Final heroes are determined from the ten played slots, not pick-attempt count.
Final played heroes must each have exactly one successful event on the correct
team. Extra failed attempts survive reconstruction with inferred phases. Duplicate
orders, missing event positions, duplicate played heroes, wrong teams, mismatched
phase counts and extra attempts after the final pick are rejected.

| Team pick | Visible allies | Visible enemies | Phase |
| --- | ---: | ---: | ---: |
| 3 | 2 | 2 | 2 |
| 4 | 3 | 2 | 2 |
| 5 | 4 | 4 | 3 |

Enemy picks from the current simultaneous phase remain hidden even when source
ordering puts them earlier. Candidate-phase failed attempts exclude that decision
by default; a failure in a different phase does not exclude a valid last pick.

All bans are preserved. Local AP data frequently has picks followed by bans,
which does not establish chronological ban visibility. OpenDota also documents
ordering concerns in its [replay parser](https://github.com/odota/parser/blob/master/src/main/java/opendota/Parse.java).
The conservative v2 adapter leaves AP ban phases unknown. `unlocated_bans` is
audit context, never a draft-time feature. `visible_bans` requires proven earlier
phase metadata. Actual candidate legality cannot be reconstructed from hidden
enemy picks/unknown bans; `legal_candidates` is therefore None. A configured
hero universe is recorded separately and does not prove legality or patch eligibility.

## Role inference

LegacyPos1Heuristic retains the original exactly-two-safelane LH/gold-at-10 rule.
FarmRoleBaseline compares all available safelane candidates at 5/10/15 minutes:

- jointly separated LH and gold are required; identical snapshots add no evidence;
- comparative farm scores and position_est hints produce deterministic ordering;
- multiple corroborating snapshots strengthen confidence; conflicts reduce it;
- missing unrelated lane metadata is tolerated, and trilane cases can succeed;
- missing telemetry/insufficient lane evidence/ambiguous scores explicitly abstain.

XP/GPM/XPM and position estimates are retained in evidence. Final farm does not
override early lane evidence. Thresholds and minimum confidence are configurable.
Confidence is evidence strength, not a calibrated probability, and source
position_est is not a trustworthy supervised target. Role evidence is post-match
labeling information and must never enter the predicted match's draft features.

A future role classifier needs independently adjudicated roles, preferably replay
review plus multiple signals. Sample across patches/ranks, retain disagreement,
split by time and match, compare heuristic coverage/precision and abstention,
then measure probability calibration. Do not train/evaluate against these same
heuristics and present that as independent ground truth.

## Building datasets

```sh
python -m lastpick normalize
python -m lastpick dataset
```

Normalization reads raw source JSON once, preserves metadata and publishes
`data/interim/v2/matches/index.json` after canonical records are durable. Invalid
JSON/source records and duplicate IDs have a separate rejection report. Dataset
construction reads only that index and canonical objects, not raw source files.

Each build gets a new directory under `data/processed/v2/`:

- `carry_decisions.jsonl/.csv`: general valid carry picks 3–5, explicit phase;
- `last_pick_decisions.jsonl/.csv`: primary 4-allies/4-enemies chosen-carry dataset;
- `rejections.jsonl`: match- and decision-stage rejection reasons;
- `manifest.json`: schema/dataset/implementation/Python versions, build time,
  configuration, raw and canonical counts, normalization summary/digest,
  accepted/rejected counts, decision counts by team pick, patch/rank/confidence
  distributions, input digest and artifact SHA-256 checksums.

`latest.json` publishes the completed build; old builds remain. Sorted inputs,
side order and exports make dataset bytes deterministic. Directory names and
default build times intentionally differ. The builder API accepts an explicit
build timestamp for identical manifests. Empty datasets still have CSV headers.
There are no generated unpicked-candidate labels. V1 CSVs remain untouched.

Default filters require known high-rank context. Old ISP detail files often lack
discovery rank metadata; available player ranks may supply it. Set
`require_rank = false` in a separate config to include truly unknown rank, keeping
unknown metadata explicit and reviewing the resulting distribution. Restrict
`patches` or `min_patch` as needed; the default patch floor is source ID 59,
preserving the ISP collector's conservative scope for the supported draft layout.
Region is preserved, with no filter UI yet.

## Historical statistics and evaluation

AsOfHistory is the canonical v2 statistics engine. Both rich descriptive snapshots
and model input generation use it. A result is eligible only when its match
completed strictly before the decision cutoff: `start_time + duration < as_of`.
Equal timestamps cannot reveal each other's outcomes. Missing completion time
is excluded/reported; actual source publication delay is unavailable.

The shared counters retain hero/duration-group win rates, farming, symmetric
ally synergy and directional enemy matchups with shrunk sample support. Rich
extras keep independent measurement support, lane-pair farm/results and first
item purchase timing by hero/patch. Empty purchase logs contribute to item-rate
denominators; missing logs do not. Unknown values remain None in rich outputs.
The fixed 20-field ISP numeric selection uses neutral defaults and support values
for compatibility. FeatureSpec chooses fields separately from accumulation.

`build_feature_rows` requires a training match allowlist, restores input row order,
and uses per-decision cutoffs. Validation/test call it with `frozen_as_of` equal to
the training cutoff. TemporalSplit keeps entire matches and equal start timestamps
together. Returned snapshots are detached from subsequent engine updates.

```sh
python -m lastpick history --as-of UNIX_SECONDS \
  --training-ids train_ids.json --output snapshot.json
```

The ID file is a JSON integer array. Without it, CLI snapshots are explicitly
marked descriptive. Never feed legacy all-history aggregate CSVs into training.
Those exporters are retained solely for ISP reproduction. The corrected ISP CSV
trainer bridges its extracted history into this same completed-match engine,
freezes at validation cutoff, and passes real 1/.5 weights into minibatches.
It writes separate Foundation artifacts; updated metrics will differ from v1.
The trainer still accepts the ISP CSV/quality contract; automatic v2 JSONL model
training and a new model comparison are the next milestone.

## Checks and known limits

```sh
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

GitHub Actions installs CPU PyTorch and the package extras, then runs all checks.
Strict typing covers the v2 schemas/config/transport/drafts/roles/dataset/history
and split APIs. Retained dynamic ISP ML/legacy modules have lint/format and runtime
coverage, but are excluded from strict mypy to keep this milestone focused.
Tests use synthetic canonical data/mocked network plus the tracked model for
web/inference integration, without the large raw corpus or expensive training.

### Verification on 2026-10-04

| Check | Result |
| --- | --- |
| Full pytest suite | 78 passed in 4.21 seconds |
| Ruff lint | All checks passed |
| Ruff formatting | Passed |
| Strict mypy | No issues in 27 core source files |
| Editable installation | Passed, without reinstalling the existing ML stack |
| Core/collector CLI help | Passed for package, module and direct-script commands |
| Static export smoke | 127 heroes/portraits, 44 candidates, 142 first-layer inputs |
| Real-data smoke | 200 normalized; 193 accepted matches; 216 general and 81 last-pick decisions |
| Preservation diff | ISP model/results, constants, web static files and site/public unchanged |

The local pytest run emitted one existing Python importlib metadata deprecation
warning. Local checks used the existing arm64 Python 3.12.6 environment:

```sh
PYTHONPATH=/private/tmp/isp-web-deps arch -arm64 .venv/bin/python -m pytest -q
PYTHONPATH=/private/tmp/lastpick-dev arch -arm64 .venv/bin/python -m ruff check .
PYTHONPATH=/private/tmp/lastpick-dev arch -arm64 .venv/bin/python -m ruff format --check .
PYTHONPATH=/private/tmp/lastpick-dev arch -arm64 .venv/bin/python -m mypy
```

The 200-match smoke ran on copies under `/private/tmp`, with patch floor 59,
no API downloads and no full-corpus regeneration. Seven matches were rejected:
five for insufficient rank coverage, one short match and one ambiguous final draft.
General decisions split into 61 pick-3, 74 pick-4 and 81 pick-5 contexts. Historical
availability checks passed. This establishes pipeline operability, not corpus-level
model quality. GitHub Actions was added but has not been executed remotely.

Remaining limits: heuristic roles/confidence, incomplete AP ban timing, uncertain
source publication delay, inconsistent historical rank coverage, no patch-aware
hero legality registry, and no causal/counterfactual recommendation evaluation.
Rich statistics are observational; duration groups are outcome-conditioned
historical summaries, not causal early-game strength labels. Quality selection
can change sample composition. Full-corpus benchmarking and role adjudication
are needed before claims about v2 model quality.

Next: audit/build the full 4v4 corpus, compare chronological simple baselines and
the existing MLP, evaluate calibration/coverage and patch/rank drift, and establish
an independent role evaluation set. Complex neural models and live infrastructure
remain later milestones.
