# builds carry model inputs from earlier training matches

import csv
import math
from itertools import combinations
from typing import Any

PROFILE_FIELDS = [
    "hero_win_rate",
    "hero_log_games",
    "early_win_rate",
    "mid_win_rate",
    "late_win_rate",
    "early_reliability",
    "mid_reliability",
    "late_reliability",
    "avg_gpm",
    "avg_gold_10",
    "avg_lh_10",
    "farm_reliability",
]
SYNERGY_FIELDS = [
    "synergy_mean_delta",
    "synergy_max_delta",
    "synergy_mean_log_games",
    "synergy_mean_reliability",
]
MATCHUP_FIELDS = [
    "matchup_mean_delta",
    "matchup_min_delta",
    "matchup_mean_log_games",
    "matchup_mean_reliability",
]
FEATURE_FIELDS = PROFILE_FIELDS + SYNERGY_FIELDS + MATCHUP_FIELDS
# uses 100 imaginary games at a 50 percent win rate to soften small sample results
PRIOR_GAMES = 100


# reads csv rows one at a time
def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


# creates empty counters for heroes, ally pairs and enemy matchups
def empty_stats(
    prior_games: float = PRIOR_GAMES, early_seconds: int = 1800, late_seconds: int = 2700
) -> dict[str, Any]:
    return {
        "heroes": {},
        "synergy": {},
        "matchups": {},
        "matches": 0,
        "latest_time": 0,
        "prior_games": prior_games,
        "early_seconds": early_seconds,
        "late_seconds": late_seconds,
    }


# creates a text key for two heroes
def pair_key(a, b, allied=False):
    # ally order does not matter but enemy matchups keep the candidate first
    if allied:
        a, b = sorted((a, b))
    return f"{a}:{b}"


# adds one match to the historical hero and pair counters
def add_match(stats: dict[str, Any], match: dict[str, Any]) -> None:
    won = match["radiant_win"]
    duration = match["duration"]
    # groups matches by duration below 30 minutes, below 45 minutes or longer
    stage = (
        None
        if duration is None
        else 0
        if duration < stats.get("early_seconds", 1800)
        else 1
        if duration < stats.get("late_seconds", 2700)
        else 2
    )
    for team, result in ((match["radiant"], won), (match["dire"], 1 - won)):
        for hero in team:
            # creates the counters if this hero has not been seen yet
            # positions 0 and 1 store total games and wins
            # positions 2 to 7 store games and wins for each duration group
            # positions 8 to 13 store totals and counts for gpm, gold_10 and lh_10
            profile = stats["heroes"].setdefault(str(hero), [0.0] * 14)
            profile[0] += 1
            profile[1] += result
            if stage is not None:
                profile[2 + 2 * stage] += 1
                profile[3 + 2 * stage] += result
            for i, value in enumerate(match["farm"].get(hero, [None] * 3)):
                if value is not None:
                    profile[8 + 2 * i] += value
                    profile[9 + 2 * i] += 1
        # visits every unique pair of allies once
        for a, b in combinations(team, 2):
            pair = stats["synergy"].setdefault(pair_key(a, b, True), [0, 0])
            pair[0] += 1
            pair[1] += result
    # records enemy matchups from both heroes' perspectives
    for a in match["radiant"]:
        for b in match["dire"]:
            for first, second, result in ((a, b, won), (b, a, 1 - won)):
                pair = stats["matchups"].setdefault(pair_key(first, second), [0, 0])
                pair[0] += 1
                pair[1] += result
    stats["matches"] += 1
    stats["latest_time"] = max(stats["latest_time"], match["start_time"])


# pulls win rates with few games towards 50 percent
def win_rate(wins, games, prior_games=PRIOR_GAMES):
    return (wins + prior_games * 0.5) / (games + prior_games)


# gives more weight to statistics based on more games
def reliability(games, prior_games=PRIOR_GAMES):
    return games / (games + prior_games)


# summarizes the candidate's history with visible allies or enemies
def pair_values(stats, candidate, others, allied):
    prior = stats.get("prior_games", PRIOR_GAMES)
    table = stats["synergy" if allied else "matchups"]
    candidate_profile = stats["heroes"].get(str(candidate), [0.0] * 14)
    candidate_rate = win_rate(candidate_profile[1], candidate_profile[0], prior)
    deltas, log_games, support = [], [], []
    for hero in others:
        games, wins = table.get(pair_key(candidate, hero, allied), [0, 0])
        # compares enemy results with the candidate's usual win rate
        reference = candidate_rate
        if allied:
            profile = stats["heroes"].get(str(hero), [0.0] * 14)
            # ally results use the average of both heroes' usual win rates
            reference = (reference + win_rate(profile[1], profile[0], prior)) / 2
        # shrinks pair effects with little history and uses zero for unseen pairs
        delta = (wins / games - reference) * reliability(games, prior) if games else 0.0
        deltas.append(delta)
        # compresses large game counts while keeping zero games as zero
        log_games.append(math.log1p(games))
        support.append(reliability(games, prior))
    if not others:
        return [0.0] * 4
    # includes the strongest ally effect or the weakest enemy matchup effect
    return [
        sum(deltas) / len(deltas),
        max(deltas) if allied else min(deltas),
        sum(log_games) / len(log_games),
        sum(support) / len(support),
    ]


# creates one ordered list of historical inputs for a carry candidate
def feature_vector(
    stats: dict[str, Any], candidate: int, allies: list[int], enemies: list[int]
) -> list[float]:
    prior = stats.get("prior_games", PRIOR_GAMES)
    profile = stats["heroes"].get(str(candidate), [0.0] * 14)
    # keeps values in the same order as feature_fields
    values = [win_rate(profile[1], profile[0], prior), math.log1p(profile[0])]
    values += [win_rate(profile[3 + 2 * i], profile[2 + 2 * i], prior) for i in range(3)]
    values += [reliability(profile[2 + 2 * i], prior) for i in range(3)]
    # divides each farm total by its own count of available values
    values += [
        profile[8 + 2 * i] / profile[9 + 2 * i] if profile[9 + 2 * i] else 0.0 for i in range(3)
    ]
    values += [sum(reliability(profile[9 + 2 * i], prior) for i in range(3)) / 3]
    return (
        values
        + pair_values(stats, candidate, allies, True)
        + pair_values(stats, candidate, enemies, False)
    )


# converts training rows into a numeric table of historical inputs
def features_for_rows(rows, stats):
    import numpy as np

    output = []
    for row in rows:
        allies = [int(row[f"ally_{i}"]) for i in range(1, 5) if row[f"ally_{i}"]]
        enemies = [int(row[f"enemy_{i}"]) for i in range(1, 5) if row[f"enemy_{i}"]]
        output.append(feature_vector(stats, int(row["candidate_hero_id"]), allies, enemies))
    return np.asarray(output, dtype=np.float32)


# loads complete drafts and stats for training matches only
def load_training_matches(data_dir, train_rows, validation_cutoff):
    allowed = {row["match_id"]: int(row["start_time"]) for row in train_rows}
    # prevents validation or test results from entering the historical source
    if any(time >= validation_cutoff for time in allowed.values()):
        raise ValueError("historical source contains a validation/test match")
    matches = {}
    for row in csv_rows(data_dir / "full_draft_examples.csv"):
        match_id = row["match_id"]
        if match_id not in allowed:
            continue
        if int(row["start_time"]) != allowed[match_id]:
            raise ValueError("historical draft timestamp mismatch")
        if match_id in matches:
            raise ValueError("duplicate historical match")
        matches[match_id] = {
            "match_id": match_id,
            "start_time": allowed[match_id],
            "radiant_win": int(row["radiant_win"]),
            "radiant": [int(row[f"radiant_{i}"]) for i in range(1, 6)],
            "dire": [int(row[f"dire_{i}"]) for i in range(1, 6)],
            "duration": None,
            "farm": {},
        }
    # checks that every allowed training match has a complete draft
    if set(matches) != set(allowed):
        raise ValueError("missing training match in complete drafts")
    seen = set()
    for row in csv_rows(data_dir / "match_features.csv"):
        match_id = row["match_id"]
        if match_id not in matches:
            continue
        # checks that the separate csv files describe the same match
        if (
            int(row["start_time"]) != allowed[match_id]
            or int(row["radiant_win"]) != matches[match_id]["radiant_win"]
        ):
            raise ValueError("historical match metadata mismatch")
        matches[match_id]["duration"] = float(row["duration"]) if row["duration"] else None
        seen.add(match_id)
    if seen != set(allowed):
        raise ValueError("missing training match metadata")
    # attaches available farm stats to each hero in each match
    for row in csv_rows(data_dir / "player_match_features.csv"):
        if row["match_id"] in matches:
            matches[row["match_id"]]["farm"][int(row["hero_id"])] = [
                float(row[field]) if row[field] else None for field in ("gpm", "gold_10", "lh_10")
            ]
    return sorted(matches.values(), key=lambda match: (match["start_time"], match["match_id"]))


# builds training inputs using only matches before each time block
def training_features(train_rows, matches, blocks=5, freeze_before=None, settings=None):
    import numpy as np

    from lastpick.features.asof import AsOfHistory
    from lastpick.legacy.history_adapter import canonical_history

    if blocks <= 0 or not train_rows:
        raise ValueError("positive blocks and nonempty training rows required")
    times = sorted({int(row["start_time"]) for row in train_rows})
    boundaries = sorted({times[int(len(times) * i / blocks)] for i in range(blocks)})
    output = np.empty((len(train_rows), len(FEATURE_FIELDS)), dtype=np.float32)
    history = AsOfHistory(canonical_history(matches), settings=settings)
    report = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else float("inf")
        history.advance(start)
        indices = [j for j, row in enumerate(train_rows) if start <= int(row["start_time"]) < end]
        output[indices] = features_for_rows([train_rows[j] for j in indices], history.stats)
        latest = history.matches[history.position - 1][0] if history.position else None
        report.append(
            {
                "start_time": start,
                "rows": len(indices),
                "source_matches": history.stats["matches"],
                "latest_source_time": history.stats["latest_time"],
                "latest_available_time": latest,
            }
        )
    if freeze_before is None:
        freeze_before = int(max((time for time, _ in history.matches), default=times[-1])) + 1
    history.advance(freeze_before)
    return output, history.stats, report


# selects inputs while optionally leaving out one group for comparison
def selected_fields(remove=None):
    groups = {"synergy": SYNERGY_FIELDS, "matchup": MATCHUP_FIELDS, "profile": PROFILE_FIELDS}
    omitted = groups.get(remove, [])
    return [field for field in FEATURE_FIELDS if field not in omitted]


# saves historical stats and training averages needed to scale the selected inputs
def feature_bundle(stats, train_features, fields):
    indices = [FEATURE_FIELDS.index(field) for field in fields]
    # selects all rows but only the requested columns
    values = train_features[:, indices]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    # avoids division by zero for columns with almost no variation
    scale[scale < 1e-6] = 1.0
    return {"stats": stats, "fields": fields, "mean": mean.tolist(), "scale": scale.tolist()}


# scales inputs using the averages and standard deviations saved during training
def normalize(values, bundle):
    import numpy as np

    indices = [FEATURE_FIELDS.index(field) for field in bundle["fields"]]
    return ((values[:, indices] - np.asarray(bundle["mean"])) / np.asarray(bundle["scale"])).astype(
        np.float32
    )
