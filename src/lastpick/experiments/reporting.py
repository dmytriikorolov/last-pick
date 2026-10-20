# Turn completed benchmark runs into tables and research notes
# Write the summaries tracked with the project

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from lastpick.data.io import atomic_json
from lastpick.experiments.data import digest


def read_json(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    return value


def export_report(
    run: Path,
    audit_path: Path,
    output: Path,
    document: Path,
    holdout: Path | None = None,
    reevaluation: Path | None = None,
    ablation_run: Path | None = None,
) -> None:
    results = read_json(run / "metrics.json")
    config = read_json(run / "config.json")
    winners = read_json(run / "winners.json")
    audit = read_json(audit_path)
    paired = read_json(run / "paired_comparisons.json")
    sigmoid = read_json(run / "sigmoid_calibration.json")
    drift = read_json(run / "drift.json")
    if audit["dataset"]["decisions_sha256"] != config["dataset"]["decisions_sha256"]:
        raise ValueError("audit and experiment dataset differ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.csv").write_text((run / "comparison.csv").read_text())
    # detailed predictions and models remain local
    # retain small, reviewable summaries
    compact_audit = dict(audit)
    compact_audit["distributions"] = {
        k: v for k, v in audit["distributions"].items() if k != "start_time"
    }
    atomic_json(output / "audit.json", compact_audit)
    summary: dict[str, Any] = {
        "primary_config": config,
        "primary_results": results,
        "winners": winners,
        "paired_comparisons": paired,
        "sigmoid_calibration": sigmoid,
        "drift": drift,
        "source_run_artifacts_sha256": digest(run / "artifacts.json"),
    }
    if ablation_run is not None:
        ablation_config = read_json(ablation_run / "config.json")
        if any(
            ablation_config[key] != config[key]
            for key in ("dataset", "split", "seed", "evaluation_candidate_pool")
        ):
            raise ValueError("tree ablation run does not share the primary data contract")
        extra = read_json(ablation_run / "metrics.json")
        if set(extra) & set(results):
            raise ValueError("duplicate model names in tree ablation run")
        summary["tree_ablations"] = {
            "config": ablation_config,
            "results": extra,
            "sigmoid_calibration": read_json(ablation_run / "sigmoid_calibration.json"),
            "source_run_artifacts_sha256": digest(ablation_run / "artifacts.json"),
        }
        results = {**results, **extra}
        with (
            (output / "comparison.csv").open("a") as destination,
            (ablation_run / "comparison.csv").open() as source,
        ):
            next(source)
            shutil.copyfileobj(source, destination)
    if reevaluation is not None:
        repeated = read_json(reevaluation)
        for name, value in summary["primary_results"].items():
            if (
                value["status"] == "complete"
                and repeated[name]["metrics"] != value["partitions"]["test"]["metrics"]
            ):
                raise ValueError(f"reevaluation differs from saved metrics: {name}")
        summary["reevaluated_time_quartiles"] = {
            name: value["subgroups"]["time_quartile"] for name, value in repeated.items()
        }
        summary["reevaluation_sha256"] = digest(reevaluation)
    secondary: dict[str, Any] = {}
    if holdout is not None:
        secondary = read_json(holdout / "metrics.json")
        summary["holdout"] = {
            "config": read_json(holdout / "config.json"),
            "results": secondary,
            "winners": read_json(holdout / "winners.json"),
            "sigmoid_calibration": read_json(holdout / "sigmoid_calibration.json"),
            "drift": read_json(holdout / "drift.json"),
            "source_run_artifacts_sha256": digest(holdout / "artifacts.json"),
        }
        (output / "holdout_comparison.csv").write_text((holdout / "comparison.csv").read_text())
    atomic_json(output / "benchmark.json", summary)
    ranking_rows = []
    for name, result in results.items():
        if result["status"] != "complete":
            continue
        test = result["partitions"]["test"]
        ranking_rows.append(
            {
                "model": name,
                **test["ranking"]["eligible_metrics"],
                "coverage": test["ranking"]["coverage"],
                "top1_unique_heroes": test["ranking"]["top1_unique_heroes"],
                "top1_largest_share": max(test["ranking"]["top1_percent"].values()),
                "ally_top1_change_common_pool": test["sensitivity"]["ally"][
                    "top1_change_common_pool"
                ],
                "enemy_top1_change_common_pool": test["sensitivity"]["enemy"][
                    "top1_change_common_pool"
                ],
            }
        )
    with (output / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ranking_rows)
    sequence = [
        "lr_candidate",
        "lr_draft",
        "lr_static",
        "lr_profile",
        "lr_synergy",
        "lr_matchup",
        "lr_farming",
        "lr_full",
    ]
    if ablation_run is not None:
        sequence += [
            "catboost_raw",
            "catboost_static",
            "catboost_profile",
            "catboost_pairs",
            "catboost_full",
        ]
    ablations = []
    previous: dict[str, Any] | None = None
    for name in sequence:
        if name not in results:
            continue
        if name == "catboost_raw":
            previous = None
        validation = results[name]["partitions"]["validation"]["metrics"]
        test = results[name]["partitions"]["test"]["metrics"]
        ablations.append(
            {
                "model": name,
                "groups": ",".join(results[name]["feature_groups"]),
                "validation_log_loss": validation["log_loss"],
                "test_log_loss": test["log_loss"],
                "validation_delta_from_previous": validation["log_loss"] - previous["log_loss"]
                if previous
                else None,
            }
        )
        previous = validation
    with (output / "ablations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ablations[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ablations)
    complete = {name: v for name, v in results.items() if v["status"] == "complete"}
    expanded_validation_best = min(
        complete, key=lambda name: complete[name]["partitions"]["validation"]["metrics"]["log_loss"]
    )
    most_sensitive = max(
        complete,
        key=lambda name: (
            complete[name]["partitions"]["test"]["sensitivity"]["ally"]["top1_change_common_pool"]
            + complete[name]["partitions"]["test"]["sensitivity"]["enemy"][
                "top1_change_common_pool"
            ]
        ),
    )
    names_path = document.parent.parent / "data/constants/heroes.csv"
    names: dict[str, str] = {}
    if names_path.exists():
        with names_path.open() as stream:
            names = {str(r["hero_id"]): r["hero_name"] for r in csv.DictReader(stream)}
    winner = winners["validation_selected"]
    partition = config["split"]["partitions"]
    verification_lines: list[str] = []
    if (output / "checks.json").exists():
        checks = read_json(output / "checks.json")
        research = checks["pytest_research"]
        normal = checks["pytest_without_research"]
        verification_lines = [
            f"local verification passed {research['passed']} tests with {research['warnings']} warnings in {research['seconds']} seconds, or {normal['passed']} tests with {normal['skipped']} optional skips and {normal['warnings']} warning in {normal['seconds']} seconds without research packages",
            "",
            f"the Ruff lint check passed, all {checks['ruff_format']['files']} files were formatted and strict mypy passed for {checks['mypy']['files']} source files; check output is summarized in reports/modeling-v2/checks.json and remote CI has not run",
            "",
        ]
    lines = [
        "# modeling & evaluation v2",
        "",
        "this milestone benchmarks observed chosen-carry win probabilities on canonical 4v4 decisions and audits how the resulting scores behave as rankings",
        "",
        f"the benchmark selected `{winner}` by validation log loss; the test minima differ by metric and are reported descriptively below",
        "",
        "the ISP checkpoint, result JSON, web demo, site/public and hosting configuration remain unchanged",
        "",
        "## data and full-corpus audit",
        "",
        f"the full local cache contains {audit['raw_matches']:,} raw matches, with {audit['normalized_matches']:,} canonical matches, {audit['accepted_matches']:,} accepted matches and {audit['rejected_matches']:,} dataset-level rejections",
        "",
        f"normalization rejected {audit['normalization']['rejected']} source records, with reasons `{audit['normalization']['rejection_reasons']}`",
        "",
        f"the build has {audit['general_decisions']:,} general decisions and {audit['last_pick_decisions']:,} true 4v4 decisions from {audit['last_pick_unique_matches']:,} matches, spanning {audit['time_range'][0]} to {audit['time_range'][1]}",
        "",
        f"patch distribution `{audit['distributions']['patch']}`, rank distribution `{audit['distributions']['rank_bracket']}`, side distribution `{audit['distributions']['side']}`",
        "",
        "all canonical rank provenance is `player_ranks`; available player rank tiers were averaged, without reconstructing exact MMR or assuming tier 75 is Immortal",
        "",
        f"the corpus includes {audit['unique_carry_heroes']} observed carry heroes, including {len(audit['low_support_heroes_under_100'])} below 100 total examples; all remain in classification and the audit",
        "",
        f"accepted matches contributing 0/1/2 last-pick decisions: `{audit['accepted_match_contribution']}`, percentages `{audit['accepted_match_contribution_percent']}`",
        "",
        f"match rejection reasons: `{audit['rejection_reasons']}`",
        "",
        f"decision rejection reasons: `{audit['decision_rejection_reasons']}`",
        "",
        f"missing metadata rates: `{audit['missing_metadata_rates']}`",
        "",
        "visible-ban coverage is zero and every row has unlocated bans; those bans are audit information and never become inputs or evaluation-pool exclusions",
        "",
        f"lane and early-farm telemetry cover approximately {100 * audit['telemetry_coverage']['lane_role']:.2f}% of canonical players, while GPM/XPM cover all players; coverage denominators and individual measurements are in the audit JSON",
        "",
        f"role method distribution `{audit['distributions']['role_method']}`; confidence summary `{audit['role_confidence_summary']}`",
        "",
        "there are no duplicated canonical IDs or experiment decisions; canonical files are normalized forms of the same raw matches and are never concatenated as extra observations",
        "",
        "## collection windows and split",
        "",
        f"the cache contains June and August windows only: `{audit['distributions']['month']}`",
        "",
        f"the largest missing interval is {audit['collection_discontinuities']['largest_time_gaps'][0]['gap_seconds'] / 86400:.2f} days; monthly rank mix is `{audit['collection_discontinuities']['month_rank_mix']}`",
        "",
        "this concentration and telemetry-dependent role coverage make the dataset a selected cache rather than a representative sample of all ranked drafts",
        "",
        "| partition | rows | matches | first UTC start | last UTC start |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    if ablation_run is not None:
        lines[6:6] = [
            f"three exploratory CatBoost ablations share the same frozen primary split; `{expanded_validation_best}` has the lowest expanded validation loss, while the original validation selection remains `{winner}`",
            "",
            "those extra ablations reused an already inspected test set and are not an independent confirmatory benchmark",
            "",
        ]
    for key, value in partition.items():
        lines.append(
            f"| {key} | {value['rows']} | {value['matches']} | {value['start']} | {value['end']} |"
        )
    lines += [
        "",
        "chronological match-level partitions keep equal start timestamps together; candidate support, vocabulary and preprocessing use training only",
        "",
        f"primary cutoffs are {config['split']['validation_cutoff']} and {config['split']['test_cutoff']}, corresponding to {config['split']['validation_cutoff_utc']} and {config['split']['test_cutoff_utc']}",
        "",
        "every accepted row, including rare/unseen chosen heroes, remains in classification; ranking eligibility is reported separately",
        "",
        "## inputs and leakage prevention",
        "",
        "the contract exposes match ID, side, chosen carry, four allies, four enemies, reliably visible bans, patch, rank, average tier, region, start time and outcome; role method/confidence are label and audit fields",
        "",
        "predictors use candidate identity, unordered ally/enemy indicators, optional known side/patch/rank/region and historical features; role evidence, current-match farm, GPM, final items and outcome never enter the input representation",
        "",
        "the AsOfHistory engine only accumulates training-match IDs whose start plus duration is strictly earlier than the decision time; validation/test history freezes at the validation cutoff, excluding unfinished training matches and all evaluation outcomes",
        "",
        "history comes from training matches that contribute accepted 4v4 decisions, rather than every cached match; all ten players in those completed matches contribute historical hero and pair counters across positions",
        "",
        "feature groups are profile win rate, ally synergy deltas, enemy matchup deltas, farming/game-stage summaries and separate support/reliability counts; missing numeric means use neutral defaults with explicit independent supports",
        "",
        "historical duration-conditioned rates describe past outcomes by game length and are not causal measures of early/late strength; no item/late-lane predictors were added",
        "",
        "static hero attributes/roles use the tracked constants snapshot pinned by the source commit, rather than a patch-indexed registry; this static ablation cannot establish historical patch-specific correctness",
        "",
        "rank/region context assumes availability at draft time; source publication latency and contemporaneous player-rank validity are not established by this cache",
        "",
        "## models and search",
        "",
        "global predicts the training win rate, hero priors shrink with 100 pseudo-observations toward that rate, and hero-plus-side cells require at least 100 training rows per cell before shrinking toward their hero prior",
        "",
        "popularity is a training pick-count ranking comparator with the global probability forecast; it does not train unpicked heroes as losing outcomes",
        "",
        "the LR uses deterministic candidate/context categories and separate ally/enemy membership indicators, training-only scaling and C in {0.01, 0.1, 1}; the cumulative ablations below share the same split",
        "",
        "the CatBoost models treat candidate/context identities as categorical and hero membership as binary; depth 4/6 with L2 10/20, learning rate 0.04, at most 400 trees and validation early stopping after 40 rounds",
        "",
        "the LightGBM models use training-fitted one-hot categorical indicators; 7/15 leaves, learning rate 0.03, minimum leaf support 100, L2 10, feature fraction 0.8, at most 400 trees and validation early stopping after 40 rounds",
        "",
        "the existing CarryModel keeps 16-dimensional candidate/ally/enemy embeddings, masked means, candidate-times-ally/enemy interactions and hidden widths 64/32; optional static features and scaled historical/context inputs are concatenated",
        "",
        "the MLP uses CPU deterministic seeds, Adam at 0.001, weight decay 0.001, batch size 512, at most 20 epochs and patience 4; accepted v2 labels have weight one and heuristic confidence is not used as a target or weight",
        "",
        "| model | selected configuration | train seconds |",
        "| --- | --- | ---: |",
    ]
    for name, value in complete.items():
        p = {
            k: v
            for k, v in value["parameters"].items()
            if k not in ("search", "epochs", "estimator_parameters")
        }
        lines.append(
            f"| {name} | `{json.dumps(p, sort_keys=True)}` | {value['train_seconds']:.2f} |"
        )
    lines += [
        "",
        "## primary probability comparison",
        "",
        "| model | split | accuracy | auc | log loss | brier | ece |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in complete.items():
        for split, entry in value["partitions"].items():
            m = entry["metrics"]
            lines.append(
                f"| {name} | {split} | {m['accuracy']:.5f} | {m['roc_auc']:.5f} | {m['log_loss']:.5f} | {m['brier']:.5f} | {m['ece']:.5f} |"
            )
    lines += [
        "",
        "accuracy is secondary; comparisons prioritize held-out log loss, Brier, discrimination and calibration",
        "",
        f"the validation-selected winner is `{winner}`, test log-loss minimum `{winners['test_log_loss']}`, test AUC maximum `{winners['test_auc']}` and raw ECE minimum `{winners['test_ece']}`",
        "",
        "a global forecast can be well calibrated while having AUC 0.5, so ECE alone cannot select a useful recommender",
        "",
        "## match-level uncertainty",
        "",
        "200 seeded bootstrap replicates resample whole matches, keeping both side decisions together; intervals condition on these trained models and do not account for repeated model selection or establish causality",
        "",
        f"paired test log-loss differences below are `{winner}` minus each comparator, so negative values favor the validation-selected model",
        "",
        "| comparator | difference | 95% match-bootstrap interval |",
        "| --- | ---: | --- |",
    ]
    for name in ("global", "hero", "lr_context", "catboost_raw", "catboost_full", "nn_full"):
        if name not in paired["test_differences"]:
            continue
        value = paired["test_differences"][name]["log_loss"]
        lines.append(f"| {name} | {value['difference']:.6f} | {value['interval_95']} |")
    lines += [
        "",
        "the trees beat trivial priors on this primary split; tiny differences among the trees and versus LR are not resolved by these intervals",
        "",
        "## calibration",
        "",
        "the ECE uses ten fixed equal-width probability bins, including empty bins; calibration JSON contains counts, mean predictions, observed win rates and Brier scores",
        "",
        "sigmoid calibration fits the validation-selected model's validation logits only; its validation fit score is intentionally omitted and test outcomes do not fit the transformation",
        "",
        f"the sigmoid slope is {sigmoid['slope']:.5f} and intercept {sigmoid['intercept']:.5f}, with test metrics `{sigmoid['test']['metrics']}`",
        "",
        "calibration improves the probability scale without establishing counterfactual recommendation value",
        "",
        "## observed-pick agreement and ranking behavior",
        "",
        f"the evaluation candidate universe has {len(config['evaluation_candidate_pool'])} heroes with at least 100 training decisions; visible allies, enemies and proven earlier bans are excluded",
        "",
        "the hidden simultaneous enemy carry and unlocated bans are not known at recommendation time; these are evaluation pools, with exact legality unknown for every row",
        "",
        "single observed chosen heroes are the only relevance labels; Recall@1/3/5, MRR and NDCG@5 measure observed-pick agreement, not actual recommendation quality or counterfactual regret",
        "",
        "| model | pool coverage | recall@1 | recall@3 | recall@5 | mrr | top1 heroes | largest top1 share | ally changes | enemy changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranking_rows:
        lines.append(
            f"| {row['model']} | {row['coverage']:.4f} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | {row['recall_at_5']:.4f} | {row['mrr']:.4f} | {row['top1_unique_heroes']} | {row['top1_largest_share']:.2f}% | {row['ally_top1_change_common_pool']:.4f} | {row['enemy_top1_change_common_pool']:.4f} |"
        )
    lines += [
        "",
        "agreement table denominators are rows whose chosen hero is in the training-supported pool; JSON also reports coverage and all-row recalls with out-of-pool picks counting as misses",
        "",
        "sensitivity uses 256 seeded test drafts with one visible ally/enemy replaced by an unused training-observed hero; common-pool Top-1 changes separate score changes from candidate removal",
        "",
        "the perturbations can be unusual drafts and are model-behavior diagnostics, not estimates of real intervention effects",
        "",
        "| model | dominant top1 hero | share | different-draft rank correlation | mean score range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("lr_context", "catboost_raw", "catboost_full", "lightgbm_full", "nn_full"):
        if name not in complete:
            continue
        value = complete[name]["partitions"]["test"]
        r, s = value["ranking"], value["sensitivity"]
        hero = max(r["top1_percent"], key=lambda h: r["top1_percent"][h])
        lines.append(
            f"| {name} | {names.get(hero, hero)} | {r['top1_percent'][hero]:.2f}% | {s['different_drafts_mean_rank_correlation']:.4f} | {r['mean_score_range']:.5f} |"
        )
    lines += [
        "",
        "additive LR can classify draft strength while leaving candidate order unchanged, because the same draft term shifts every candidate's logit; its good AUC does not imply a useful context-sensitive ranking",
        "",
        f"the full tree variants vary candidate-specific scores much more than raw trees; `{most_sensitive}` is most sensitive by summed ally/enemy common-pool changes, which does not establish recommendation superiority",
        "",
        "winning and losing observed picks remain observational and cannot validate alternatives",
        "",
        "| model | winning recall@1 | winning mrr | losing recall@1 | losing mrr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("catboost_full", "lightgbm_full", "nn_full"):
        if name in complete:
            r = complete[name]["partitions"]["test"]["ranking"]
            winning_metrics, losing_metrics = (
                r["winning_observed_picks"]["eligible_metrics"],
                r["losing_observed_picks"]["eligible_metrics"],
            )
            lines.append(
                f"| {name} | {winning_metrics['recall_at_1']:.4f} | {winning_metrics['mrr']:.4f} | {losing_metrics['recall_at_1']:.4f} | {losing_metrics['mrr']:.4f} |"
            )
    lines += [
        "",
        "## ablations",
        "",
        "| model | groups | validation log loss | test log loss | validation change |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in ablations:
        delta = (
            f"{row['validation_delta_from_previous']:.6f}"
            if row["validation_delta_from_previous"] is not None
            else ""
        )
        lines.append(
            f"| {row['model']} | {row['groups']} | {row['validation_log_loss']:.6f} | {row['test_log_loss']:.6f} | {delta} |"
        )
    lines += [
        "",
        "visible draft is the main LR gain; static features add effectively nothing, and profile/support, synergy, matchup and farming-stage additions do not consistently improve validation probability quality",
        "",
        "the profile step adds support/reliability inputs together, so it does not isolate profile from support; subsequent synergy/matchup steps isolate their deltas",
        "",
        "context improves LR validation loss and test AUC; the full static/history bundle modestly helps tree validation and substantially improves ranking sensitivity, while raw CatBoost has essentially the same test loss as the full tree models",
        "",
        "the full MLP improves over draft-only MLP but does not beat the strongest conventional models",
        "",
    ]
    if ablation_run is not None:
        lines += [
            "the focused CatBoost sequence adds static features, then profile/support, then synergy/matchup deltas and finally farming/game-stage summaries; profile/support has the best validation loss, while pair deltas hurt that loss despite increasing ranking diversity",
            "",
            "static CatBoost improves observed-pick agreement without improving test probability loss; these two objectives differ",
            "",
            "the extra CatBoost variants were not separately retrained in the June-to-August holdout, so their temporal robustness remains unestablished",
            "",
        ]
    lines += [
        "## patch, rank, side and time",
        "",
        "one source patch means cross-patch stability cannot be measured; subgroup metrics retain low-support counts with unreliable flags rather than hiding them",
        "",
        "subgroups require at least 100 rows and 20 examples of each outcome; 30-day time buckets and match-grouped chronological quartiles are descriptive diagnostics",
        "",
        f"primary train-to-test candidate pick-rate total variation is {drift['train_to_test_pick_total_variation']:.4f}; hero win-rate counts and historical feature shifts are stored in benchmark.json",
        "",
        "| model | rank | rows | auc | log loss | ece |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("lr_context", "catboost_full", "lightgbm_full", "nn_full"):
        if name in complete:
            for group, m in complete[name]["partitions"]["test"]["subgroups"][
                "rank_bracket"
            ].items():
                if m["reliable"]:
                    lines.append(
                        f"| {name} | {group} | {m['count']} | {m['roc_auc']:.5f} | {m['log_loss']:.5f} | {m['ece']:.5f} |"
                    )
    if reevaluation is not None:
        lines += [
            "",
            "### primary test time quartiles",
            "",
            "| model | quartile | rows | auc | log loss | ece |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for name in ("lr_context", "catboost_full", "lightgbm_full", "nn_full"):
            if name not in summary["reevaluated_time_quartiles"]:
                continue
            for group, m in summary["reevaluated_time_quartiles"][name].items():
                if m["reliable"]:
                    lines.append(
                        f"| {name} | {group} | {m['count']} | {m['roc_auc']:.5f} | {m['log_loss']:.5f} | {m['ece']:.5f} |"
                    )
    if secondary:
        hconfig = summary["holdout"]["config"]
        hcal = summary["holdout"]["sigmoid_calibration"]
        lines += [
            "",
            "### june-to-august holdout",
            "",
            f"the separate holdout uses June for training/validation and all August rows for test, with partition sizes `{ {k: v['rows'] for k, v in hconfig['split']['partitions'].items()} }` and {len(hconfig['evaluation_candidate_pool'])} supported candidates",
            "",
            f"holdout cutoffs: {hconfig['split']['validation_cutoff_utc']} and {hconfig['split']['test_cutoff_utc']}; the test boundary is inside the empty collection interval",
            "",
            "| model | June validation auc | August test auc | June validation log loss | August test log loss | August ece |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, v in secondary.items():
            if v["status"] == "complete":
                a, b = v["partitions"]["validation"]["metrics"], v["partitions"]["test"]["metrics"]
                lines.append(
                    f"| {name} | {a['roc_auc']:.5f} | {b['roc_auc']:.5f} | {a['log_loss']:.5f} | {b['log_loss']:.5f} | {b['ece']:.5f} |"
                )
        lines += [
            "",
            f"holdout sigmoid calibration for `{hcal['base_model']}` uses June validation only, with August test metrics `{hcal['test']['metrics']}`",
            "",
            f"june-to-august candidate pick-rate total variation is {summary['holdout']['drift']['train_to_test_pick_total_variation']:.4f}",
            "",
            "the LR loses probability calibration across the collection gap despite retaining discrimination; differences in support, rank mix and collection can confound any attribution to meta drift",
            "the June-fitted sigmoid worsens August LightGBM log loss and ECE; calibration fitted in an earlier window is not automatically portable across time",
            "",
        ]
    lines += [
        "## symmetry and side effects",
        "",
        "automated tests enforce ally/enemy order invariance, duplicate rejection and side invariance when side is absent; measured permutation differences are zero in all primary models",
        "",
        "side is not a full-draft swap symmetry for this chosen-carry task; swapping every hero across teams would change candidate and label semantics",
        "",
        "| model | mean Radiant-minus-Dire prediction with fixed heroes |",
        "| --- | ---: |",
    ]
    for name in ("lr_context", "catboost_full", "lightgbm_full", "nn_full"):
        if name in complete:
            effect = complete[name]["partitions"]["test"]["sensitivity"]["side_effect"][
                "mean_radiant_minus_dire"
            ]
            lines.append(f"| {name} | {effect:.5f} |")
    side_stats = complete[winner]["partitions"]["test"]["subgroups"]["side"]
    empirical = side_stats["0"]["observed_win_rate"] - side_stats["1"]["observed_win_rate"]
    lines += [
        "",
        f"empirical test selected-carry Radiant-minus-Dire win-rate difference is {empirical:.5f}; this is composition-dependent and not a causal side estimate",
        "",
        "## role-label sensitivity and review",
        "",
        "these comparisons evaluate the fixed trained model on confidence subsets, rather than retraining on each subset; higher heuristic confidence is evidence strength, not independent ground truth",
        "",
        "| threshold | rows | auc | log loss | brier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for threshold, value in complete[winner]["partitions"]["test"]["role_sensitivity"].items():
        m = value["metrics"]
        if value["count"] and m.get("roc_auc") is not None:
            lines.append(
                f"| {threshold} | {value['count']} | {m['roc_auc']:.5f} | {m['log_loss']:.5f} | {m['brier']:.5f} |"
            )
        else:
            lines.append(
                f"| {threshold} | {value['count']} | unreliable | unreliable | unreliable |"
            )
    lines += [
        "",
        "hero distributions for every threshold/model are preserved in benchmark.json; improved subset loss cannot establish improved role-label accuracy",
        "",
        "the manual sample is [reports/role_review_sample.csv](../reports/role_review_sample.csv), with 134 high-confidence, 133 medium-confidence and 133 abstention cases stratified by difficult safelane status, patch, rank and candidate hero",
        "",
        "the review pack includes all team heroes, lane roles, position estimates, LH/gold/XP snapshots, GPM/XPM, heuristic evidence and blank reviewed_pos1_slot/reviewer_confidence/notes columns",
        "",
        "the sample oversamples rare strata and abstentions, and would need sampling weights for population role-precision estimates; reviewers should use independent replay/role evidence and retain ambiguous cases",
        "",
        "## reproducibility and commands",
        "",
        f"primary source commit `{config['git_commit']}`, holdout source commit `{summary.get('holdout', {}).get('config', {}).get('git_commit', 'not run')}`, seed {config['seed']}",
        "",
        f"decision digest `{config['dataset']['decisions_sha256']}`, canonical digest `{config['dataset']['canonical_input_digest']}`, raw digest `{config['dataset']['raw_digest']}`",
        "",
        f"runtime versions `{config['versions']}`",
        "",
        "each run stores source hashes, dataset/manifest digests, config, training allowlist digest, split IDs/cutoffs, fitted model/preprocessing, search results, train duration, probability/ranking CSVs, calibration, grouped uncertainty and artifact checksums",
        "",
        f"local runs are `{run}` and `{holdout}`; generated models and predictions are gitignored, while compact tables/summary JSON and the review pack are tracked",
        "",
        "```sh",
        "python -m pip install -e '.[research,dev,web]'",
        "python -m lastpick normalize",
        "python -m lastpick dataset",
        "python -m lastpick audit-dataset",
        "python -m lastpick benchmark --seed 42 --bootstrap 200 --output experiments/modeling-v2/new-primary",
        "python -m lastpick benchmark --models global hero hero_side lr_context lr_full catboost_full lightgbm_full nn_full --holdout-start 1785542400 --seed 42 --output experiments/modeling-v2/new-holdout",
        "python -m lastpick benchmark --models catboost_static catboost_profile catboost_pairs --seed 42 --output experiments/modeling-v2/new-tree-ablations",
        "python -m lastpick evaluate --run experiments/modeling-v2/new-primary --output experiments/modeling-v2/reloaded.json",
        "python -m lastpick role-review-sample --size 400",
        "python -m lastpick benchmark-report --run experiments/modeling-v2/new-primary --holdout experiments/modeling-v2/new-holdout --ablations experiments/modeling-v2/new-tree-ablations --reevaluation experiments/modeling-v2/reloaded.json --audit reports/modeling-v2/generated/audit.json --output reports/modeling-v2 --document docs/MODELING_V2.md",
        "```",
        "",
        "use --build to select a retained dataset build, --models to select the suite, --seed for deterministic sampling and --output for a fresh run directory; research dependencies are optional and imported by research commands",
        "",
        "local Apple Silicon runs used arch -arm64 with the existing venv and optional packages under /private/tmp/lastpick-research; LightGBM used the OpenMP library already bundled with PyTorch",
        "",
        "saved-model reevaluation reproduces every primary metric exactly; artifact hashes detect dataset/run corruption before reloading",
        "",
        "## checks and engineering",
        "",
        "the experiments package adds a validated contract, audit, training-only features, priors/LR/trees/MLP, metrics/calibration, ranking behavior, drift, role review and report generation",
        "",
        "strict typing covers the contract, audit, review and report APIs; numerical research modules follow the existing dynamic ML convention and are excluded from strict mypy",
        "",
        "synthetic tests cover match/time splits, train-only preprocessing/pools/history, order/side invariants, ECE, match bootstrap, ranking calculations, low-support groups, digests, baseline seeds and saved-run reload; optional tree smoke tests skip when research packages are absent",
        "",
        "the CI installs the normal dev/web extras and runs no real corpus or full model training; the local numerical smoke uses two tiny MLP epochs",
        "",
        *verification_lines,
        "## conclusions and next research question",
        "",
        "the LightGBM with history is the validation-selected benchmark and a good probability/context-sensitivity compromise; raw CatBoost has the lowest primary test loss, while LR with context has the highest primary test AUC and is the simplest competitive classifier",
        "",
        "sigmoid-calibrated LightGBM has the lowest measured primary ECE, narrowly below a trivial global forecast; its calibration does not prove stronger rankings",
        "",
        "the current small NN does not justify an architecture expansion from these results; raw tree and LR concentration also rule out automatic production promotion",
        "",
        "prioritize temporal modeling with continuously covered independent time blocks and rolling-origin evaluation, including calibration and support drift; the observed June-to-August failure is stronger evidence for this direction than the static-feature or small-pair ablation gains",
        "",
        "complete the independent role review alongside that work before claiming a supervised role classifier; separate synergy/matchup components remain promising for ranking behavior but need better validation of their probability and recommendation value",
        "",
        "offline data can measure chosen-carry prediction, calibration and observed-pick agreement, while unchosen outcomes, counterfactual regret and actual recommendation benefit remain unobserved",
        "",
    ]
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("\n".join(lines))
