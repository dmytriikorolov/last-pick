# LastPick: canonical project knowledge

## Product and scope

This is `dmytriikorolov/lastPick`, a Dota 2 position-1 last-pick recommender.
The finished ISP submission earned credit. Foundation v2 supplies safe data,
engineering and evaluation foundations; the shipped UI still uses the ISP NN.
modeling & evaluation v2 now supplies a full-corpus observational benchmark
see `docs/MODELING_V2.md` and tracked `reports/modeling-v2/` summaries
The eventual product includes strong baselines, learned roles, temporal meta,
matchups/synergy, items, personalization, explanations, live drafts and retraining.
Those are future research milestones, not requirements to implement opportunistically.

## Baseline preservation

- `isp-v1` identifies final ISP commit `7dd54e2`
  foundation v2 merged into main at `f0444b1`; current branch is `modeling-evaluation-v2`
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
- `experiments/data.py`: verified 4v4 loader, draft input allowlist and training-only evaluation pools
- `experiments/audit.py`, `review.py`: full-corpus audit and balanced independent role-review pack
- `experiments/features.py`: unordered features and completed-match training-only AsOfHistory inputs
- `experiments/models.py`: priors, LR, CatBoost, LightGBM and the existing small CarryModel
- `experiments/metrics.py`, `diagnostics.py`: calibration, match bootstrap, subgroups, agreement and sensitivity
- `experiments/benchmark.py`, `reporting.py`: reproducible runs, reload checks and generated research reports
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

- Raw caches: `data/raw/opendota/matches/*.json` or `*.json.gz`; new downloads use
  gzip. `python -m scripts.compress_raw_matches` verifies original bytes before
  removing plain copies. V2 reads both formats and preserves normalization digests;
  legacy ISP extractors/tag require decompression first. ISP processed CSVs:
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
foundation verification was 78 tests and 27 strict core files, with a 200-match smoke
modeling verification is 91 tests with research dependencies, Ruff lint/format passed
and strict mypy passed for 32 files; without research packages, 89 pass and 2 skip
the core contract/audit/review/report APIs are strict; numerical research modules
follow the existing dynamic ML convention and are excluded from strict mypy
the suite remains synthetic and takes about 5 seconds locally; remote CI was not run

## Working rules and next milestones

Inspect before editing; keep changes focused, explicit, typed and deterministic.
Do not overwrite unrelated uncommitted changes, delete data/files without approval,
or introduce heavy infrastructure/fancy ML without a requested milestone. Record
rejection reasons; do not swallow broad exceptions or invent zero telemetry.
Keep future Codex knowledge and architecture docs synchronized with real commands.

## modeling v2 benchmark facts

- full build: `data/processed/v2/build-vl4kzp0e`, indexed canonical directory unchanged in layout
- raw 96,524; normalized 96,520; accepted 92,585; match rejections 3,935
- normalization rejections: 2 invalid durations and 2 invalid event teams
- dataset rejections: rank coverage 3,127; ambiguous final picks 428; short duration 269; rank below minimum 111
- general decisions 109,822; 4v4 decisions 44,329 from 36,644 matches
- time range 2026-06-04 to 2026-08-24 UTC; only patch ID 60, June/August collection windows with a 76.55-day gap
- 4v4 ranks: Divine 22,048 and Immortal 22,281; sides: Radiant 22,290 and Dire 22,039
- 111 chosen heroes, 71 with fewer than 100 total rows; those rows remain in classification
- known bans 0%; unlocated bans 100%; unknown exact legality 100%; missing region 9.08%
- lane/early-farm coverage about 68.70% of canonical players; all rank provenance is available player ranks
- accepted matches contributing 0/1/2 last-pick decisions: 55,941 / 28,959 / 7,685
- primary chronological rows: train 30,894, validation 6,592, test 6,843; evaluation pool 37 training-supported heroes
- primary cutoffs 1787468454 and 1787517986; match IDs and equal timestamps never cross partitions
- history uses training 4v4 match IDs only, completed strictly before decision time; validation/test freeze at the training cutoff
- raw/decision digests and config/source hashes are in `reports/modeling-v2/benchmark.json`
- local runs: `experiments/modeling-v2/full-seed42` at source `d9d1a7c` and `june-to-august-seed42` at `051da45`
- 20 primary models: 4 priors, 10 LR variants, 2 CatBoost, 2 LightGBM, 2 small MLP variants
- 3 additional CatBoost static/profile/pair ablations at `tree-ablations-seed42`, source `53b10a7`
- priors shrink with 100 pseudo-games; hero/side cells require 100 training rows
- the LR searches C 0.01/0.1/1; CatBoost depth 4/6 and L2 10/20; LightGBM 7/15 leaves; trees cap at 400 iterations
- the MLP embedding is 16, hidden 64/32, Adam 0.001, weight decay 0.001, batch 512, cap 20 epochs, patience 4
- primary NN parameter counts 9,345 draft-only and 14,977 full; best epochs 3 and 2 respectively
- validation log-loss winner LightGBM full 0.687002; test loss 0.684811, AUC 0.573088, Brier 0.245864
- expanded validation minimum CatBoost profile/support 0.686674, test 0.685042; extra variants reuse an inspected test set
- extra profile/pair variants were not retrained in the June-to-August holdout; treat them as exploratory
- primary test loss minimum CatBoost raw 0.684635; test AUC maximum LR context 0.577672
- sigmoid LightGBM test ECE 0.004415, versus raw 0.007944; the global prior ECE is 0.004447 with AUC 0.5
- full MLP test loss 0.687265, AUC 0.560961; it loses to conventional trees on probability quality
- tree-to-tree and tree-to-LR loss differences are not resolved by 200-replicate paired match-bootstrap intervals
- the LR context ranks Bloodseeker first in 99.49% of test drafts and has zero common-pool ally/enemy Top-1 changes
- the CatBoost raw ranks Phantom Lancer first in 92.85% of drafts despite good classification loss
- the LightGBM full has 33 Top-1 heroes, observed-pick Recall@1/3/5 0.0815/0.1776/0.2535, MRR 0.1892
- the LightGBM full common-pool Top-1 changes are ally 17.58%, enemy 56.64%; perturbations are synthetic diagnostics
- full MLP has the largest summed sensitivity among the initial 20 models; order permutation differences are zero
- extra CatBoost pairs has the largest expanded sensitivity: ally 36.72%, enemy 57.42%, 32 Top-1 heroes
- extra CatBoost static has observed-pick Recall@1 0.095 and MRR 0.2208, without better probability loss
- ranking pool covers 96.45% of primary test picks; imitation metrics are observed-pick agreement, never causal recommendation quality
- static/profile/synergy/farming additions do not consistently improve LR validation loss
- extra CatBoost profile/support improves validation loss; adding pair deltas hurts it while increasing ranking sensitivity
- separate June-to-August rows: train 20,140, validation 5,097, test 19,092; evaluation pool 36 heroes
- the June-to-August candidate pick total variation is 0.235; LR context test loss 0.693833 is worse than global 0.693090
- holdout CatBoost full test loss 0.688446, LightGBM full 0.689191, MLP full 0.691540
- the June-fitted LightGBM sigmoid worsens August loss to 0.690113 and ECE to 0.016426
- role-confidence sensitivity evaluates fixed models on subsets, without treating confidence as ground truth
- the LightGBM test confidence >=0.90 has 4,837 rows, loss 0.683349, AUC 0.571166 versus all-row AUC 0.573088
- manual review: `reports/role_review_sample.csv`, 400 unlabeled team cases, quotas 134 high / 133 medium / 133 abstention
- independent human fields remain blank; strata cover difficult safelane cases, rank, hero and abstention reasons
- the canonical byte digest matches the build; verification is in `reports/modeling-v2/source_verification.json`
- new audit/benchmark calls verify canonical bytes against the build before historical feature construction

## modeling commands and next milestone

```sh
python -m pip install -e '.[research,dev,web]'
python -m lastpick audit-dataset
python -m lastpick benchmark --seed 42 --output experiments/modeling-v2/new-run
python -m lastpick benchmark --models global hero lr_context catboost_full lightgbm_full nn_full --holdout-start 1785542400 --output experiments/modeling-v2/new-holdout
python -m lastpick evaluate --run experiments/modeling-v2/new-run --output experiments/modeling-v2/reloaded.json
python -m lastpick role-review-sample --size 400
python -m lastpick benchmark-report --run experiments/modeling-v2/new-run --audit reports/modeling-v2/generated/audit.json
```

research packages remain optional; macOS LightGBM can use PyTorch's bundled OpenMP
all experimental models, predictions and large run artifacts are gitignored and separate
from the protected ISP files; no production model, frontend or hosting changes were made

next prioritize temporal coverage/modeling and rolling-origin calibration checks
the strongest signal is performance/calibration degradation across the collection gap
complete independent role review alongside that work before supervised role claims
pair components may improve ranking behavior, but need evidence of probability benefit
larger NNs, richer static hero knowledge and production promotion are not supported yet
keep counterfactual outcomes unknown and do not generate negatives from unpicked heroes
