# LastPick: canonical project knowledge

## Product and scope

This is `dmytriikorolov/lastPick`, a Dota 2 position-1 last-pick recommender.
The finished ISP submission earned credit. Foundation v2 supplies safe data,
engineering and evaluation foundations; the shipped UI still uses the ISP NN.
The eventual product includes strong baselines, learned roles, temporal meta,
matchups/synergy, items, personalization, explanations, live drafts and retraining.
Those are future research milestones, not requirements to implement opportunistically.

## Baseline preservation

- `isp-v1` identifies final ISP commit `7dd54e2`. Development branch: `foundation-v2`.
- Protect `models/final_carry_recommender.pt` and `models/final_carry_training_results.json`.
  Do not overwrite/retrain these files to run experiments. The corrected trainer
  writes `models/foundation_*` by default and refuses the ISP output paths.
- ISP reference: train 73,558; validation 12,275; test 22,205; candidates 44;
  accuracy .5345192524; AUC .5457609609; log loss .6899946928; Brier .2484227717.
  Validation log loss .6915258765, best epoch 2, 13,313 trained parameters.
- Tag checkout reproduces original logic, including its quality-weight bug.
  Data is gitignored; reproducing the exact experiment also needs original local
  raw/processed datasets. See `docs/ISP_V1.md`. Never delete them casually.
- The model objective is chosen carry + visible draft -> win/loss. Inference
  scores legal candidates. Unpicked carries have unknown counterfactual outcomes;
  NEVER label every unpicked carry as a negative example.

## Canonical architecture

`src/lastpick` is the importable package; `scripts` provides compatibility CLIs.

- `config.py`, `config/default.toml`: frozen typed configuration, root-relative paths.
- `ingestion/source.py`: provider interface; `data/adapter.py`: normalization interface.
- `ingestion/opendota.py`, `data/opendota.py`: sole implemented source and adapter.
- `ingestion/collector.py`: sequential collector, atomic caches/state, persistent
  unfinished pages, request budgets, retry policy, distinct recorded failure types.
- `data/schemas.py`: Match, Player, Purchase, DraftEvent, Draft, RolePrediction,
  DraftDecision; JSON schema version `2.0`. Missing fields remain None.
- `data/pipeline.py`: normalize raw files once, publish authoritative canonical index.
- `data/validation.py`: integrity checks and independent dataset selection policies.
- `drafts/reconstruction.py`: complete events, final-ten recovery and phase visibility.
- `roles/inference.py`: RoleInferer protocol, LegacyPos1Heuristic, FarmRoleBaseline.
- `data/dataset.py`: general carry and dedicated 4v4 dataset builds/manifests.
- `features/carry.py`: shared hero/pair counters and ISP numeric compatibility APIs.
- `features/asof.py`: canonical completed-match as-of engine, rich farm/lane/item
  statistics, immutable returned snapshots, training allowlists and frozen evaluation.
- `features/selection.py`: model-specific selection separate from accumulation.
- `evaluation/splits.py`: deterministic chronological match/timestamp grouping.
- `models/carry.py`, `models/training.py`: preserved MLP, corrected quality weights.
- `recommender/carry.py`: inference, candidate pool/pick/ban constraints.
- `legacy/`: quarantined ISP CSV extraction, descriptive aggregates and tooling.
  Its raw JSON and CSV interfaces exist for reproduction. Do not add new v2 logic here.
  `legacy/history_adapter.py` bridges ISP CSV history into the canonical as-of engine.
- `web/` and `site/public/`: existing Flask/static demos; no frontend replacement.

The v2 flow is provider -> cached raw -> adapter -> canonical Match -> validation
-> roles/reconstructed draft -> DraftDecision -> versioned dataset -> as-of features.
No new downstream component should re-parse OpenDota JSON independently.

## Data and invariants

- ISP raw: `data/raw/opendota/matches/*.json`; ISP processed CSVs:
  drafts, training_examples, full_draft_examples, match_features,
  player_match_features, training_quality and extracted event tables.
- Hero/item snapshots: tracked `data/constants/`, including `heroes.csv`.
- V2 normalized: `data/interim/v2/matches/{match_id}.json` plus `index.json` and
  normalization rejection report. Only indexed records are current inputs.
- V2 builds: `data/processed/v2/build-*/`, with general/last-pick JSONL and CSV,
  rejection JSONL and manifest. `latest.json` changes after completion; earlier
  builds are retained. Data bytes are deterministic; build timestamps/directories differ.
- Rank tier buckets: tens digit 1..8 = Herald..Immortal. Tier 75 is Divine,
  not an exact MMR or a trusted Immortal label. Discovery and available player
  ranks are preserved with provenance; unavailable rank/region remain None.
- Default patch floor is source ID 59, matching the original API collector's scope.
  New draft layouts require explicit visibility validation before broadening filters.
- Team `side`: 0 Radiant, 1 Dire. `slot`: 0..4 or 128..132. Names use snake_case;
  external hero IDs are distinct from model embedding indices (zero is padding).
- Final teams need ten distinct heroes and valid slots. Repeated played-hero
  attempts, inconsistent teams/orders and uncertain phase patterns are rejected.
- Phase visibility: team pick 3 => 2 allies/2 enemies; 4 => 3/2; 5 => 4/4.
  Current-phase enemies stay hidden even if earlier in raw source ordering.
- Full events retain bans AND failed attempts. Candidate-phase extra attempts
  exclude that decision by default. Never replace the stream with just final heroes.
- AP `picks_bans` often appends bans after all picks. This is not reliable ban
  chronology. Unknown ban phases remain unknown, are carried as `unlocated_bans`,
  and MUST NOT be fed into draft-time model inputs. Only reliably earlier phases
  enter `visible_bans`. `legal_candidates` stays None when legality is not proven;
  a configured candidate_universe does not establish exact legality.

## Roles and leakage

- Legacy rule: exactly two safelane players, all five players have lane data,
  >=15 LH and >=400 gold separation at minute 10; otherwise abstain.
- Farm v2 compares >=2 safelane candidates at available 5/10/15-minute snapshots,
  requires LH AND gold separation, uses position_est as a secondary score hint,
  discounts conflicting evidence and returns confidence/evidence/abstention reasons.
  Missing unrelated lane metadata and trilane cases no longer automatically fail.
  XP/GPM/XPM are preserved as evidence, not allowed to override early farm.
- These are heuristics, not supervised ground truth. Confidence is uncalibrated.
  Role telemetry is post-match label evidence, NEVER a current draft feature.
- V2 history requires `start_time + duration < as_of`; equal completion timestamps
  stay hidden. Unknown completion time is excluded/reported. No current/future
  outcome or telemetry may enter a feature. Actual API publication latency is unknown.
- Model feature building requires a training match allowlist. Validation/test
  must freeze at the training cutoff. Descriptive all-history snapshots are not
  approved training features. Do not load legacy aggregate CSVs into new models.
- Missing measurements have separate counts; rich stats preserve None. The fixed
  ISP numeric feature selection alone uses neutral numeric defaults with support.
- Quality weights are 1/.5/0; actual minibatches now carry the saved weights.
  Quality filters are sample selection policies, not predictive input features.

## Commands and tests

From the repository root, Python >=3.11:

```sh
python -m pip install -e '.[dev,web]'
python -m lastpick --config config/default.toml ingest --max-pages 1
python -m lastpick normalize
python -m lastpick dataset
python -m lastpick history --as-of UNIX_SECONDS --training-ids train_ids.json --output snapshot.json
python -m web.app
python -m scripts.export_web_site --output dist
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Use `--root` or LASTPICK_ROOT for assets outside a checkout. OPENDOTA_API_KEY is
an environment variable; do not commit credentials. V2 does not implicitly read .env.
Both historical collector script names run the same implementation; discovery
policy is `public` or `explorer` in TOML. V2 uses separate state, does not interpret
old ISP cursors/budgets. Start with one page and review reports before bulk ingestion.
Use a new state directory when changing source/discovery/filter policies.

Tests use small canonical fixtures and mocked transport; model/web integration
tests use only the tracked checkpoint and constants, never the large raw corpus.
CI runs lint, format, strict typing for the v2 domain/ingestion/dataset/history
APIs, and the full suite. Retained dynamic ISP ML/legacy modules are excluded from
strict mypy scope. Apple Silicon may require `arch -arm64` with the existing venv.
Final local verification: 78 tests passed, Ruff lint/format passed, strict mypy
passed for 27 core files. A 200-match copied-data smoke produced 216 general and
81 last-pick decisions, without altering ISP data/artifacts. Full details and the
environment warning are in `docs/FOUNDATION_V2.md`; remote CI has not yet run.

## Working rules and next milestones

Inspect before editing; keep changes focused, explicit, typed and deterministic.
Do not overwrite unrelated uncommitted changes, delete data/files without approval,
or introduce heavy infrastructure/fancy ML without a requested milestone. Record
rejection reasons; do not swallow broad exceptions or invent zero telemetry.
Keep future Codex knowledge and architecture docs synchronized with real commands.

Next: build/audit the full v2 4v4 corpus, establish temporal majority/hero-rate/LR
and existing MLP baselines, document ablations, coverage/support and calibration.
Create an independently adjudicated role evaluation set, stratified by patch/rank,
before training or claiming a supervised classifier. Later: meta drift, matchup
components, items, recommendation objective research, personalization, explanations,
live integration and scheduled retraining. No naive counterfactual labels.
