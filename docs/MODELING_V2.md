# modeling & evaluation v2

this milestone benchmarks observed chosen-carry win probabilities on canonical 4v4 decisions and audits how the resulting scores behave as rankings

the benchmark selected `lightgbm_full` by validation log loss; the test minima differ by metric and are reported descriptively below

three exploratory CatBoost ablations share the same frozen primary split; `catboost_profile` has the lowest expanded validation loss, while the original validation selection remains `lightgbm_full`

those extra ablations reused an already inspected test set and are not an independent confirmatory benchmark

the ISP checkpoint, result JSON, web demo, site/public and hosting configuration remain unchanged

## data and full-corpus audit

the full local cache contains 96,524 raw matches, with 96,520 canonical matches, 92,585 accepted matches and 3,935 dataset-level rejections

normalization rejected 4 source records, with reasons `{'invalid_duration': 2, 'invalid_event_team': 2}`

the build has 109,822 general decisions and 44,329 true 4v4 decisions from 36,644 matches, spanning 2026-06-04T14:16:24+00:00 to 2026-08-24T13:34:08+00:00

patch distribution `{'60': 44329}`, rank distribution `{'divine': 22048, 'immortal': 22281}`, side distribution `{'dire': 22039, 'radiant': 22290}`

all canonical rank provenance is `player_ranks`; available player rank tiers were averaged, without reconstructing exact MMR or assuming tier 75 is Immortal

the corpus includes 111 observed carry heroes, including 71 below 100 total examples; all remain in classification and the audit

accepted matches contributing 0/1/2 last-pick decisions: `{'0': 55941, '1': 28959, '2': 7685}`, percentages `{'0': 60.42123454123238, '1': 31.278284819355186, '2': 8.300480639412433}`

match rejection reasons: `{'ambiguous_final_picks': 428, 'below_min_rank': 111, 'insufficient_rank_coverage': 3127, 'missing_or_short_duration': 269}`

decision rejection reasons: `{'ambiguous_pos1': 1526, 'below_role_confidence': 2743, 'candidate_phase_extra_attempt': 7970, 'insufficient_lane_evidence': 56223, 'pos1_pick_1': 2799, 'pos1_pick_2': 4087}`

missing metadata rates: `{'avg_rank_tier': 0.0, 'exact_legality': 1.0, 'patch': 0.0, 'rank_bracket': 0.0, 'region': 0.09077579011482326, 'start_time': 0.0}`

visible-ban coverage is zero and every row has unlocated bans; those bans are audit information and never become inputs or evaluation-pool exclusions

lane and early-farm telemetry cover approximately 68.70% of canonical players, while GPM/XPM cover all players; coverage denominators and individual measurements are in the audit JSON

role method distribution `{'farm-v2': 44329}`; confidence summary `{'count': 44329, 'max': 0.95, 'mean': 0.8889630884973718, 'min': 0.65, 'quantiles': {'0.05': 0.7999999999999999, '0.25': 0.8466666666666666, '0.5': 0.91, '0.75': 0.9233333333333333, '0.95': 0.9400000000000001}}`

there are no duplicated canonical IDs or experiment decisions; canonical files are normalized forms of the same raw matches and are never concatenated as extra observations

## collection windows and split

the cache contains June and August windows only: `{'2026-06': 25237, '2026-08': 19092}`

the largest missing interval is 76.55 days; monthly rank mix is `{'2026-06': {'divine': 0.5282719816142964, 'immortal': 0.47172801838570355}, '2026-08': {'divine': 0.45652629373559606, 'immortal': 0.5434737062644039}}`

this concentration and telemetry-dependent role coverage make the dataset a selected cache rather than a representative sample of all ranked drafts

| partition | rows | matches | first UTC start | last UTC start |
| --- | ---: | ---: | --- | --- |
| test | 6843 | 5677 | 2026-08-23T20:46:26+00:00 | 2026-08-24T13:34:08+00:00 |
| train | 30894 | 25580 | 2026-06-04T14:16:24+00:00 | 2026-08-23T07:00:25+00:00 |
| validation | 6592 | 5387 | 2026-08-23T07:00:54+00:00 | 2026-08-23T20:46:25+00:00 |

chronological match-level partitions keep equal start timestamps together; candidate support, vocabulary and preprocessing use training only

primary cutoffs are 1787468454 and 1787517986, corresponding to 2026-08-23T07:00:54+00:00 and 2026-08-23T20:46:26+00:00

the tables below record the original runs. The benchmark now excludes training and validation outcomes that were still pending at the next cutoff. See [the corrected reruns](#completion-cutoff-reruns).

every accepted row, including rare/unseen chosen heroes, remains in classification; ranking eligibility is reported separately

## inputs and leakage prevention

the contract exposes match ID, side, chosen carry, four allies, four enemies, reliably visible bans, patch, rank, average tier, region, start time and outcome; role method/confidence are label and audit fields

predictors use candidate identity, unordered ally/enemy indicators, optional known side/patch/rank/region and historical features; role evidence, current-match farm, GPM, final items and outcome never enter the input representation

the AsOfHistory engine only accumulates training-match IDs whose start plus duration is strictly earlier than the decision time; validation/test history freezes at the validation cutoff, excluding unfinished training matches and all evaluation outcomes

history comes from training matches that contribute accepted 4v4 decisions, rather than every cached match; all ten players in those completed matches contribute historical hero and pair counters across positions

feature groups are profile win rate, ally synergy deltas, enemy matchup deltas, farming/game-stage summaries and separate support/reliability counts; missing numeric means use neutral defaults with explicit independent supports

historical duration-conditioned rates describe past outcomes by game length and are not causal measures of early/late strength; no item/late-lane predictors were added

static hero attributes/roles use the tracked constants snapshot pinned by the source commit, rather than a patch-indexed registry; this static ablation cannot establish historical patch-specific correctness

rank/region context assumes availability at draft time; source publication latency and contemporaneous player-rank validity are not established by this cache

## models and search

global predicts the training win rate, hero priors shrink with 100 pseudo-observations toward that rate, and hero-plus-side cells require at least 100 training rows per cell before shrinking toward their hero prior

popularity is a training pick-count ranking comparator with the global probability forecast; it does not train unpicked heroes as losing outcomes

the LR uses deterministic candidate/context categories and separate ally/enemy membership indicators, training-only scaling and C in {0.01, 0.1, 1}; the cumulative ablations below share the same split

the CatBoost models treat candidate/context identities as categorical and hero membership as binary; depth 4/6 with L2 10/20, learning rate 0.04, at most 400 trees and validation early stopping after 40 rounds

the LightGBM models use training-fitted one-hot categorical indicators; 7/15 leaves, learning rate 0.03, minimum leaf support 100, L2 10, feature fraction 0.8, at most 400 trees and validation early stopping after 40 rounds

the existing CarryModel keeps 16-dimensional candidate/ally/enemy embeddings, masked means, candidate-times-ally/enemy interactions and hidden widths 64/32; optional static features and scaled historical/context inputs are concatenated

the MLP uses CPU deterministic seeds, Adam at 0.001, weight decay 0.001, batch size 512, at most 20 epochs and patience 4; accepted v2 labels have weight one and heuristic confidence is not used as a target or weight

| model | selected configuration | train seconds |
| --- | --- | ---: |
| catboost_full | `{"depth": 4, "l2_leaf_reg": 10, "max_iterations": 400, "seed": 42, "tree_count": 255}` | 13.14 |
| catboost_raw | `{"depth": 4, "l2_leaf_reg": 10, "max_iterations": 400, "seed": 42, "tree_count": 400}` | 11.34 |
| global | `{"hero_side_minimum": 100, "kind": "global", "shrinkage": 100}` | 0.00 |
| hero | `{"hero_side_minimum": 100, "kind": "hero", "shrinkage": 100}` | 0.00 |
| hero_side | `{"hero_side_minimum": 100, "kind": "hero_side", "shrinkage": 100}` | 0.00 |
| lightgbm_full | `{"best_iteration": 386, "max_iterations": 400, "num_leaves": 7, "seed": 42}` | 6.62 |
| lightgbm_raw | `{"best_iteration": 399, "max_iterations": 400, "num_leaves": 7, "seed": 42}` | 2.48 |
| lr_candidate | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 0.87 |
| lr_context | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 0.70 |
| lr_draft | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 0.44 |
| lr_farming | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 3.68 |
| lr_full | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 4.38 |
| lr_history | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 1.57 |
| lr_matchup | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 3.19 |
| lr_profile | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 3.05 |
| lr_static | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 2.48 |
| lr_synergy | `{"C": 0.01, "max_iterations": 400, "seed": 42}` | 3.28 |
| nn_draft | `{"batch_size": 512, "best_epoch": 3, "embedding_dim": 16, "hidden_sizes": [64, 32], "learning_rate": 0.001, "max_epochs": 20, "parameters": 9345, "patience": 4, "seed": 42, "weight_decay": 0.001}` | 1.77 |
| nn_full | `{"batch_size": 512, "best_epoch": 2, "embedding_dim": 16, "hidden_sizes": [64, 32], "learning_rate": 0.001, "max_epochs": 20, "parameters": 14977, "patience": 4, "seed": 42, "weight_decay": 0.001}` | 2.09 |
| popularity | `{"hero_side_minimum": 100, "kind": "popularity", "shrinkage": 100}` | 0.00 |
| catboost_pairs | `{"depth": 6, "l2_leaf_reg": 20, "max_iterations": 400, "seed": 42, "tree_count": 312}` | 15.61 |
| catboost_profile | `{"depth": 6, "l2_leaf_reg": 20, "max_iterations": 400, "seed": 42, "tree_count": 346}` | 18.26 |
| catboost_static | `{"depth": 4, "l2_leaf_reg": 10, "max_iterations": 400, "seed": 42, "tree_count": 395}` | 17.15 |

## primary probability comparison

| model | split | accuracy | auc | log loss | brier | ece |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| catboost_full | test | 0.54596 | 0.57495 | 0.68465 | 0.24579 | 0.00834 |
| catboost_full | validation | 0.53580 | 0.56057 | 0.68724 | 0.24707 | 0.00899 |
| catboost_raw | test | 0.55122 | 0.57422 | 0.68463 | 0.24578 | 0.00971 |
| catboost_raw | validation | 0.53792 | 0.55920 | 0.68776 | 0.24732 | 0.00646 |
| global | test | 0.50870 | 0.50000 | 0.69304 | 0.24994 | 0.00445 |
| global | validation | 0.50652 | 0.50000 | 0.69315 | 0.25000 | 0.00662 |
| hero | test | 0.52155 | 0.52754 | 0.69233 | 0.24959 | 0.00735 |
| hero | validation | 0.51790 | 0.53242 | 0.69152 | 0.24919 | 0.00999 |
| hero_side | test | 0.52521 | 0.53653 | 0.69116 | 0.24901 | 0.01094 |
| hero_side | validation | 0.52806 | 0.54066 | 0.69073 | 0.24879 | 0.01056 |
| lightgbm_full | test | 0.54961 | 0.57309 | 0.68481 | 0.24586 | 0.00794 |
| lightgbm_full | validation | 0.53368 | 0.56032 | 0.68700 | 0.24696 | 0.01633 |
| lightgbm_raw | test | 0.54698 | 0.57001 | 0.68503 | 0.24598 | 0.00670 |
| lightgbm_raw | validation | 0.53686 | 0.55833 | 0.68775 | 0.24732 | 0.00876 |
| lr_candidate | test | 0.52477 | 0.52717 | 0.69393 | 0.25007 | 0.00726 |
| lr_candidate | validation | 0.52200 | 0.53273 | 0.69210 | 0.24931 | 0.01006 |
| lr_context | test | 0.55458 | 0.57767 | 0.68586 | 0.24600 | 0.02438 |
| lr_context | validation | 0.53944 | 0.56149 | 0.69027 | 0.24838 | 0.03445 |
| lr_draft | test | 0.55487 | 0.57001 | 0.68727 | 0.24670 | 0.01651 |
| lr_draft | validation | 0.53944 | 0.55804 | 0.69056 | 0.24848 | 0.02908 |
| lr_farming | test | 0.55341 | 0.56932 | 0.68728 | 0.24673 | 0.01733 |
| lr_farming | validation | 0.54536 | 0.55774 | 0.69077 | 0.24858 | 0.02299 |
| lr_full | test | 0.55356 | 0.57710 | 0.68611 | 0.24601 | 0.02420 |
| lr_full | validation | 0.54232 | 0.56154 | 0.69049 | 0.24842 | 0.03106 |
| lr_history | test | 0.55254 | 0.56993 | 0.68713 | 0.24667 | 0.01892 |
| lr_history | validation | 0.54612 | 0.55801 | 0.69070 | 0.24855 | 0.02258 |
| lr_matchup | test | 0.55180 | 0.56994 | 0.68723 | 0.24669 | 0.01966 |
| lr_matchup | validation | 0.54414 | 0.55753 | 0.69083 | 0.24860 | 0.02468 |
| lr_profile | test | 0.55093 | 0.56932 | 0.68740 | 0.24678 | 0.01945 |
| lr_profile | validation | 0.54414 | 0.55744 | 0.69060 | 0.24850 | 0.02352 |
| lr_static | test | 0.55444 | 0.57005 | 0.68727 | 0.24670 | 0.01703 |
| lr_static | validation | 0.53944 | 0.55808 | 0.69056 | 0.24848 | 0.02916 |
| lr_synergy | test | 0.55166 | 0.56915 | 0.68745 | 0.24680 | 0.01869 |
| lr_synergy | validation | 0.54308 | 0.55743 | 0.69061 | 0.24850 | 0.02453 |
| nn_draft | test | 0.52565 | 0.53791 | 0.69112 | 0.24899 | 0.01354 |
| nn_draft | validation | 0.52139 | 0.53419 | 0.69173 | 0.24929 | 0.01603 |
| nn_full | test | 0.53982 | 0.56096 | 0.68726 | 0.24708 | 0.01067 |
| nn_full | validation | 0.53125 | 0.54414 | 0.69050 | 0.24867 | 0.01224 |
| popularity | test | 0.50870 | 0.50000 | 0.69304 | 0.24994 | 0.00445 |
| popularity | validation | 0.50652 | 0.50000 | 0.69315 | 0.25000 | 0.00662 |
| catboost_pairs | test | 0.54786 | 0.57174 | 0.68521 | 0.24606 | 0.00549 |
| catboost_pairs | validation | 0.54581 | 0.55998 | 0.68712 | 0.24701 | 0.00734 |
| catboost_profile | test | 0.55107 | 0.57046 | 0.68504 | 0.24598 | 0.00739 |
| catboost_profile | validation | 0.54596 | 0.56341 | 0.68667 | 0.24679 | 0.00468 |
| catboost_static | test | 0.55137 | 0.57409 | 0.68484 | 0.24588 | 0.01058 |
| catboost_static | validation | 0.54126 | 0.56044 | 0.68735 | 0.24712 | 0.00646 |

accuracy is secondary; comparisons prioritize held-out log loss, Brier, discrimination and calibration

the validation-selected winner is `lightgbm_full`, test log-loss minimum `catboost_raw`, test AUC maximum `lr_context` and raw ECE minimum `global`

a global forecast can be well calibrated while having AUC 0.5, so ECE alone cannot select a useful recommender

## match-level uncertainty

200 seeded bootstrap replicates resample whole matches, keeping both side decisions together; intervals condition on these trained models and do not account for repeated model selection or establish causality

paired test log-loss differences below are `lightgbm_full` minus each comparator, so negative values favor the validation-selected model

| comparator | difference | 95% match-bootstrap interval |
| --- | ---: | --- |
| global | -0.008224 | [-0.011181900948205332, -0.005335907053626478] |
| hero | -0.007517 | [-0.010193313453295643, -0.004875401028648096] |
| lr_context | -0.001044 | [-0.0044004022646349115, 0.0028053684562701583] |
| catboost_raw | 0.000176 | [-0.001143154356844353, 0.0017886289648773406] |
| catboost_full | 0.000157 | [-0.0007552672152552009, 0.0012287193740001508] |
| nn_full | -0.002453 | [-0.004382842402876822, -0.0005244500279388304] |

the trees beat trivial priors on this primary split; tiny differences among the trees and versus LR are not resolved by these intervals

## calibration

the ECE uses ten fixed equal-width probability bins, including empty bins; calibration JSON contains counts, mean predictions, observed win rates and Brier scores

sigmoid calibration fits the validation-selected model's validation logits only; its validation fit score is intentionally omitted and test outcomes do not fit the transformation

the sigmoid slope is 0.98060 and intercept -0.03502, with test metrics `{'accuracy': 0.5494666082127722, 'brier': 0.2458240562584795, 'count': 6843, 'ece': 0.004415130010286234, 'log_loss': 0.6847298834713401, 'roc_auc': 0.5730877623936588}`

calibration improves the probability scale without establishing counterfactual recommendation value

## observed-pick agreement and ranking behavior

the evaluation candidate universe has 37 heroes with at least 100 training decisions; visible allies, enemies and proven earlier bans are excluded

the hidden simultaneous enemy carry and unlocated bans are not known at recommendation time; these are evaluation pools, with exact legality unknown for every row

single observed chosen heroes are the only relevance labels; Recall@1/3/5, MRR and NDCG@5 measure observed-pick agreement, not actual recommendation quality or counterfactual regret

| model | pool coverage | recall@1 | recall@3 | recall@5 | mrr | top1 heroes | largest top1 share | ally changes | enemy changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| catboost_full | 0.9645 | 0.0805 | 0.1938 | 0.2733 | 0.1955 | 28 | 21.99% | 0.1875 | 0.5664 |
| catboost_raw | 0.9645 | 0.0647 | 0.2261 | 0.3112 | 0.2104 | 8 | 92.85% | 0.0234 | 0.0156 |
| global | 0.9645 | 0.0259 | 0.0788 | 0.1676 | 0.1243 | 2 | 98.55% | 0.0000 | 0.0000 |
| hero | 0.9645 | 0.0059 | 0.1226 | 0.2400 | 0.1310 | 3 | 99.49% | 0.0000 | 0.0000 |
| hero_side | 0.9645 | 0.0045 | 0.0497 | 0.1479 | 0.1036 | 4 | 49.88% | 0.0000 | 0.0000 |
| lightgbm_full | 0.9645 | 0.0815 | 0.1776 | 0.2535 | 0.1892 | 33 | 21.74% | 0.1758 | 0.5664 |
| lightgbm_raw | 0.9645 | 0.0330 | 0.1759 | 0.2508 | 0.1601 | 4 | 61.65% | 0.0195 | 0.0703 |
| lr_candidate | 0.9645 | 0.0059 | 0.0756 | 0.0927 | 0.1109 | 3 | 99.49% | 0.0000 | 0.0000 |
| lr_context | 0.9645 | 0.0059 | 0.0815 | 0.0942 | 0.1140 | 3 | 99.49% | 0.0000 | 0.0000 |
| lr_draft | 0.9645 | 0.0056 | 0.0756 | 0.0942 | 0.1031 | 2 | 99.49% | 0.0000 | 0.0000 |
| lr_farming | 0.9645 | 0.0056 | 0.0524 | 0.1347 | 0.1029 | 5 | 99.34% | 0.0078 | 0.0039 |
| lr_full | 0.9645 | 0.0056 | 0.0567 | 0.1386 | 0.1055 | 6 | 99.36% | 0.0039 | 0.0078 |
| lr_history | 0.9645 | 0.0056 | 0.0480 | 0.1321 | 0.1026 | 5 | 98.82% | 0.0078 | 0.0195 |
| lr_matchup | 0.9645 | 0.0056 | 0.0494 | 0.1303 | 0.1020 | 5 | 99.33% | 0.0039 | 0.0039 |
| lr_profile | 0.9645 | 0.0056 | 0.0232 | 0.0959 | 0.0957 | 3 | 99.49% | 0.0000 | 0.0000 |
| lr_static | 0.9645 | 0.0056 | 0.0756 | 0.0942 | 0.1031 | 2 | 99.49% | 0.0000 | 0.0000 |
| lr_synergy | 0.9645 | 0.0056 | 0.0245 | 0.0970 | 0.0954 | 3 | 99.49% | 0.0000 | 0.0000 |
| nn_draft | 0.9645 | 0.0321 | 0.1083 | 0.1859 | 0.1365 | 17 | 49.72% | 0.2461 | 0.2266 |
| nn_full | 0.9645 | 0.0585 | 0.1491 | 0.2311 | 0.1688 | 29 | 22.15% | 0.2422 | 0.5195 |
| popularity | 0.9645 | 0.0458 | 0.1477 | 0.3448 | 0.1853 | 3 | 97.94% | 0.0000 | 0.0000 |
| catboost_pairs | 0.9645 | 0.0600 | 0.1480 | 0.2224 | 0.1674 | 32 | 16.60% | 0.3672 | 0.5742 |
| catboost_profile | 0.9645 | 0.0523 | 0.1385 | 0.2250 | 0.1606 | 24 | 28.36% | 0.4766 | 0.4609 |
| catboost_static | 0.9645 | 0.0950 | 0.2148 | 0.3129 | 0.2208 | 16 | 49.01% | 0.1641 | 0.2305 |

agreement table denominators are rows whose chosen hero is in the training-supported pool; JSON also reports coverage and all-row recalls with out-of-pool picks counting as misses

sensitivity uses 256 seeded test drafts with one visible ally/enemy replaced by an unused training-observed hero; common-pool Top-1 changes separate score changes from candidate removal

the perturbations can be unusual drafts and are model-behavior diagnostics, not estimates of real intervention effects

| model | dominant top1 hero | share | different-draft rank correlation | mean score range |
| --- | --- | ---: | ---: | ---: |
| lr_context | Bloodseeker | 99.49% | 1.0000 | 0.18513 |
| catboost_raw | Phantom Lancer | 92.85% | 0.9287 | 0.06109 |
| catboost_full | Phantom Lancer | 21.99% | 0.6613 | 0.10985 |
| lightgbm_full | Lifestealer | 21.74% | 0.5478 | 0.13961 |
| nn_full | Phantom Lancer | 22.15% | 0.5832 | 0.14403 |

additive LR can classify draft strength while leaving candidate order unchanged, because the same draft term shifts every candidate's logit; its good AUC does not imply a useful context-sensitive ranking

the full tree variants vary candidate-specific scores much more than raw trees; `catboost_pairs` is most sensitive by summed ally/enemy common-pool changes, which does not establish recommendation superiority

winning and losing observed picks remain observational and cannot validate alternatives

| model | winning recall@1 | winning mrr | losing recall@1 | losing mrr |
| --- | ---: | ---: | ---: | ---: |
| catboost_full | 0.0845 | 0.2035 | 0.0762 | 0.1871 |
| lightgbm_full | 0.0842 | 0.1970 | 0.0787 | 0.1810 |
| nn_full | 0.0649 | 0.1767 | 0.0517 | 0.1604 |

## ablations

| model | groups | validation log loss | test log loss | validation change |
| --- | --- | ---: | ---: | ---: |
| lr_candidate | candidate | 0.692100 | 0.693933 |  |
| lr_draft | candidate,draft | 0.690559 | 0.687269 | -0.001541 |
| lr_static | candidate,draft,static | 0.690556 | 0.687273 | -0.000002 |
| lr_profile | candidate,draft,static,profile,support | 0.690605 | 0.687401 | 0.000048 |
| lr_synergy | candidate,draft,static,profile,synergy,support | 0.690606 | 0.687451 | 0.000001 |
| lr_matchup | candidate,draft,static,profile,synergy,matchup,support | 0.690829 | 0.687227 | 0.000223 |
| lr_farming | candidate,draft,static,profile,synergy,matchup,farming_stage,support | 0.690768 | 0.687278 | -0.000060 |
| lr_full | candidate,draft,static,profile,synergy,matchup,farming_stage,support,context | 0.690489 | 0.686106 | -0.000279 |
| catboost_raw | candidate,draft,context | 0.687762 | 0.684635 |  |
| catboost_static | candidate,draft,static,context | 0.687353 | 0.684845 | -0.000410 |
| catboost_profile | candidate,draft,static,profile,support,context | 0.686674 | 0.685042 | -0.000678 |
| catboost_pairs | candidate,draft,static,profile,synergy,matchup,support,context | 0.687121 | 0.685213 | 0.000447 |
| catboost_full | candidate,draft,static,profile,synergy,matchup,farming_stage,support,context | 0.687245 | 0.684654 | 0.000124 |

visible draft is the main LR gain; static features add effectively nothing, and profile/support, synergy, matchup and farming-stage additions do not consistently improve validation probability quality

the profile step adds support/reliability inputs together, so it does not isolate profile from support; subsequent synergy/matchup steps isolate their deltas

context improves LR validation loss and test AUC; the full static/history bundle modestly helps tree validation and substantially improves ranking sensitivity, while raw CatBoost has essentially the same test loss as the full tree models

the full MLP improves over draft-only MLP but does not beat the strongest conventional models

the focused CatBoost sequence adds static features, then profile/support, then synergy/matchup deltas and finally farming/game-stage summaries; profile/support has the best validation loss, while pair deltas hurt that loss despite increasing ranking diversity

static CatBoost improves observed-pick agreement without improving test probability loss; these two objectives differ

the extra CatBoost variants were not separately retrained in the June-to-August holdout, so their temporal robustness remains unestablished

## patch, rank, side and time

one source patch means cross-patch stability cannot be measured; subgroup metrics retain low-support counts with unreliable flags rather than hiding them

subgroups require at least 100 rows and 20 examples of each outcome; 30-day time buckets and match-grouped chronological quartiles are descriptive diagnostics

primary train-to-test candidate pick-rate total variation is 0.2057; hero win-rate counts and historical feature shifts are stored in benchmark.json

| model | rank | rows | auc | log loss | ece |
| --- | --- | ---: | ---: | ---: | ---: |
| lr_context | divine | 3709 | 0.58460 | 0.68324 | 0.02569 |
| lr_context | immortal | 3134 | 0.56932 | 0.68895 | 0.02891 |
| catboost_full | divine | 3709 | 0.58135 | 0.68403 | 0.01401 |
| catboost_full | immortal | 3134 | 0.56738 | 0.68539 | 0.00506 |
| lightgbm_full | divine | 3709 | 0.58286 | 0.68313 | 0.01272 |
| lightgbm_full | immortal | 3134 | 0.56132 | 0.68680 | 0.00935 |
| nn_full | divine | 3709 | 0.56383 | 0.68724 | 0.01773 |
| nn_full | immortal | 3134 | 0.55848 | 0.68729 | 0.00241 |

### primary test time quartiles

| model | quartile | rows | auc | log loss | ece |
| --- | --- | ---: | ---: | ---: | ---: |
| lr_context | 0 | 1736 | 0.57273 | 0.68774 | 0.03589 |
| lr_context | 1 | 1725 | 0.54410 | 0.69502 | 0.05125 |
| lr_context | 2 | 1656 | 0.59087 | 0.68303 | 0.01658 |
| lr_context | 3 | 1726 | 0.60436 | 0.67751 | 0.02363 |
| catboost_full | 0 | 1736 | 0.56976 | 0.68617 | 0.01493 |
| catboost_full | 1 | 1725 | 0.55770 | 0.68684 | 0.01475 |
| catboost_full | 2 | 1656 | 0.57941 | 0.68362 | 0.01174 |
| catboost_full | 3 | 1726 | 0.59370 | 0.68194 | 0.01819 |
| lightgbm_full | 0 | 1736 | 0.56604 | 0.68682 | 0.01366 |
| lightgbm_full | 1 | 1725 | 0.55975 | 0.68690 | 0.01702 |
| lightgbm_full | 2 | 1656 | 0.57714 | 0.68391 | 0.01215 |
| lightgbm_full | 3 | 1726 | 0.59063 | 0.68157 | 0.01622 |
| nn_full | 0 | 1736 | 0.55984 | 0.68808 | 0.01771 |
| nn_full | 1 | 1725 | 0.54293 | 0.68968 | 0.02470 |
| nn_full | 2 | 1656 | 0.55926 | 0.68732 | 0.01065 |
| nn_full | 3 | 1726 | 0.58320 | 0.68398 | 0.02270 |

### june-to-august holdout

the separate holdout uses June for training/validation and all August rows for test, with partition sizes `{'test': 19092, 'train': 20140, 'validation': 5097}` and 36 supported candidates

holdout cutoffs: 2026-06-06T16:11:09+00:00 and 2026-08-01T00:00:00+00:00; the test boundary is inside the empty collection interval

| model | June validation auc | August test auc | June validation log loss | August test log loss | August ece |
| --- | ---: | ---: | ---: | ---: | ---: |
| catboost_full | 0.57967 | 0.55376 | 0.68330 | 0.68845 | 0.00602 |
| global | 0.50000 | 0.50000 | 0.69241 | 0.69309 | 0.00532 |
| hero | 0.51635 | 0.52326 | 0.69160 | 0.69236 | 0.00986 |
| hero_side | 0.53686 | 0.53351 | 0.69020 | 0.69156 | 0.00913 |
| lightgbm_full | 0.57831 | 0.55095 | 0.68298 | 0.68919 | 0.00858 |
| lr_context | 0.58236 | 0.55674 | 0.68441 | 0.69383 | 0.03726 |
| lr_full | 0.58153 | 0.55733 | 0.68510 | 0.69385 | 0.03416 |
| nn_full | 0.56783 | 0.54081 | 0.68531 | 0.69154 | 0.01794 |

holdout sigmoid calibration for `lightgbm_full` uses June validation only, with August test metrics `{'accuracy': 0.5373978629792583, 'brier': 0.24846638852785588, 'count': 19092, 'ece': 0.0164260190126937, 'log_loss': 0.6901127253525026, 'roc_auc': 0.5509464534366686}`

june-to-august candidate pick-rate total variation is 0.2350

the LR loses probability calibration across the collection gap despite retaining discrimination; differences in support, rank mix and collection can confound any attribution to meta drift
the June-fitted sigmoid worsens August LightGBM log loss and ECE; calibration fitted in an earlier window is not automatically portable across time

## symmetry and side effects

automated tests enforce ally/enemy order invariance, duplicate rejection and side invariance when side is absent; measured permutation differences are zero in all primary models

side is not a full-draft swap symmetry for this chosen-carry task; swapping every hero across teams would change candidate and label semantics

| model | mean Radiant-minus-Dire prediction with fixed heroes |
| --- | ---: |
| lr_context | 0.06162 |
| catboost_full | 0.04992 |
| lightgbm_full | 0.05197 |
| nn_full | 0.05979 |

empirical test selected-carry Radiant-minus-Dire win-rate difference is 0.06268; this is composition-dependent and not a causal side estimate

## role-label sensitivity and review

these comparisons evaluate the fixed trained model on confidence subsets, rather than retraining on each subset; higher heuristic confidence is evidence strength, not independent ground truth

| threshold | rows | auc | log loss | brier |
| --- | ---: | ---: | ---: | ---: |
| 0.65 | 6843 | 0.57309 | 0.68481 | 0.24586 |
| 0.75 | 6736 | 0.57322 | 0.68468 | 0.24580 |
| 0.85 | 4981 | 0.57139 | 0.68366 | 0.24529 |
| 0.9 | 4837 | 0.57117 | 0.68335 | 0.24514 |

hero distributions for every threshold/model are preserved in benchmark.json; improved subset loss cannot establish improved role-label accuracy

the manual sample is [reports/role_review_sample.csv](../reports/role_review_sample.csv), with 134 high-confidence, 133 medium-confidence and 133 abstention cases stratified by difficult safelane status, patch, rank and candidate hero

the review pack includes all team heroes, lane roles, position estimates, LH/gold/XP snapshots, GPM/XPM, heuristic evidence and blank reviewed_pos1_slot/reviewer_confidence/notes columns

the sample oversamples rare strata and abstentions, and would need sampling weights for population role-precision estimates; reviewers should use independent replay/role evidence and retain ambiguous cases

## reproducibility and commands

primary source commit `d9d1a7cd88f2ef9391894d16ff06ee0e83e70f6b`, holdout source commit `051da45f0b360f8b2e970becf6c01361fba5993a`, seed 42

decision digest `22143f4c96e855e129c9c7e7909ab2ed11484a85aa1c9958d878d93a1d5ce4be`, canonical digest `85532fb81b2717d1b975cdac9197fae0fff096b9c3bb23f36c57b613b8d0e58f`, raw digest `fb7fb863dc98c7743329bf868baf532ec5e55a33ca775013129795280e16c8a3`

runtime versions `{'catboost': '1.2.10', 'lightgbm': '4.7.0', 'numpy': '2.5.2', 'scikit-learn': '1.9.0', 'torch': '2.9.1'}`

each run stores source hashes, dataset/manifest digests, config, training allowlist digest, split IDs/cutoffs, fitted model/preprocessing, search results, train duration, probability/ranking CSVs, calibration, grouped uncertainty and artifact checksums

local runs are `experiments/modeling-v2/full-seed42` and `experiments/modeling-v2/june-to-august-seed42`; generated models and predictions are gitignored, while compact tables/summary JSON and the review pack are tracked

```sh
python -m pip install -e '.[research,dev,web]'
python -m lastpick normalize
python -m lastpick dataset
python -m lastpick audit-dataset
python -m lastpick benchmark --seed 42 --bootstrap 200 --output experiments/modeling-v2/new-primary
python -m lastpick benchmark --models global hero hero_side lr_context lr_full catboost_full lightgbm_full nn_full --holdout-start 1785542400 --seed 42 --output experiments/modeling-v2/new-holdout
python -m lastpick benchmark --models catboost_static catboost_profile catboost_pairs --seed 42 --output experiments/modeling-v2/new-tree-ablations
python -m lastpick evaluate --run experiments/modeling-v2/new-primary --output experiments/modeling-v2/reloaded.json
python -m lastpick rolling-benchmark --output experiments/modeling-v2/new-weekly
python -m lastpick role-review-sample --size 400
python -m lastpick role-review-sample --pilot
python -m lastpick benchmark-report --run experiments/modeling-v2/new-primary --holdout experiments/modeling-v2/new-holdout --ablations experiments/modeling-v2/new-tree-ablations --reevaluation experiments/modeling-v2/reloaded.json --audit reports/modeling-v2/generated/audit.json --output reports/modeling-v2 --document docs/MODELING_V2.md
```

use --build to select a retained dataset build, --models to select the suite, --seed for deterministic sampling and --output for a fresh run directory; research dependencies are optional and imported by research commands

local Apple Silicon runs used arch -arm64 with the existing venv and optional packages under /private/tmp/lastpick-research; LightGBM used the OpenMP library already bundled with PyTorch

saved-model reevaluation reproduces every primary metric exactly; artifact hashes detect dataset/run corruption before reloading

## completion cutoff reruns

The original benchmark split by match start, which allowed some outcomes into
fitting or tuning before those matches ended. The corrected benchmark uses
`start_time + duration < cutoff` for training labels before validation and
for validation labels before test. Both sides of a match are excluded together.
The test windows are the same as in the original runs.

| Run | Train rows removed | Validation rows removed | Train / validation / test rows |
| --- | ---: | ---: | --- |
| Main | 165 from 137 matches | 637 from 508 matches | 30,729 / 5,955 / 6,843 |
| June to August | 368 from 314 matches | 0 | 19,772 / 5,097 / 19,092 |

The main validation winner remained LightGBM with historical features. Its
test log loss rose from 0.684811 to 0.686115. Sigmoid calibration moved the
corrected result to 0.686065. CatBoost with raw draft features had the lowest descriptive test
log loss at 0.684635 and still put one hero first in 70.6% of test drafts.

CatBoost with historical features became the June validation winner and
scored 0.688677 test log loss in August. The June-fitted sigmoid moved that
to 0.689660. Context logistic regression still scored worse than the global
prior in August (0.693696 versus 0.693093).

The 20 main and 8 holdout models used seed 42, the original candidate support
and hyperparameter settings, and 200 match-bootstrap replicates. NumPy was
2.5.5 for the corrected runs and 2.5.2 for the original runs. Reloading all 28
saved models reproduced the test metrics exactly. The complete
local outputs are in `experiments/modeling-v2/completion-main-seed42` and
`experiments/modeling-v2/completion-june-seed42`.
[Model comparison](../reports/modeling-v2/completion-cutoff-comparison.csv)

The weekly benchmark checks Monday–Sunday UTC windows. Each test and
validation week needs observations on all seven days and at least 1,000
decisions. The recent training window needs at least 3,000 decisions before
the completion exclusion. Each fold compares all available training history
against the 28 days before validation. The latest eligible week is reserved by default.
The saved 4v4 corpus has no eligible folds out of 13 UTC weeks. See the coverage
report before using `--include-final` on a future build.

The blinded role pilot selects 30 distinct matches from the original review
pack. Give two reviewers separate copies and have each record a position-1
slot and confidence. Leave the slot blank and record the reason if the role
cannot be established independently. Compare answers by match ID and side only
after both reviewers have finished.

## checks and engineering

the experiments package adds a validated contract, audit, training-only features, priors/LR/trees/MLP, metrics/calibration, ranking behavior, drift, role review and report generation

strict typing covers the contract, audit, review and report APIs; numerical research modules follow the existing dynamic ML convention and are excluded from strict mypy

synthetic tests cover match/time splits, train-only preprocessing/pools/history, order/side invariants, ECE, match bootstrap, ranking calculations, low-support groups, digests, baseline seeds and saved-run reload; optional tree smoke tests skip when research packages are absent

the CI installs the normal dev/web extras and runs no real corpus or full model training; the local numerical smoke uses two tiny MLP epochs

local verification passed 91 tests with 3 warnings in 5.65 seconds, or 89 tests with 2 optional skips and 1 warning in 5.25 seconds without research packages

the Ruff lint check passed, all 97 files were formatted and strict mypy passed for 32 source files; check output is summarized in reports/modeling-v2/checks.json and remote CI has not run

## conclusions and next research question

the LightGBM with history is the validation-selected benchmark and a good probability/context-sensitivity compromise; raw CatBoost has the lowest primary test loss, while LR with context has the highest primary test AUC and is the simplest competitive classifier

sigmoid-calibrated LightGBM has the lowest measured primary ECE, narrowly below a trivial global forecast; its calibration does not prove stronger rankings

the current small NN does not justify an architecture expansion from these results; raw tree and LR concentration also rule out automatic production promotion

prioritize temporal modeling with continuously covered independent time blocks and rolling-origin evaluation, including calibration and support drift; the observed June-to-August failure is stronger evidence for this direction than the static-feature or small-pair ablation gains

complete the independent role review alongside that work before claiming a supervised role classifier; separate synergy/matchup components remain promising for ranking behavior but need better validation of their probability and recommendation value

offline data can measure chosen-carry prediction, calibration and observed-pick agreement, while unchosen outcomes, counterfactual regret and actual recommendation benefit remain unobserved
