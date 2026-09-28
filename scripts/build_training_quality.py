#!/usr/bin/env python3
"""checks training example quality and assigns training weights"""

import csv
import math
from collections import Counter
from pathlib import Path

if __package__:
    from .hero_lookup import hero_id_to_name
else:
    from hero_lookup import hero_id_to_name


ROOT = Path(__file__).resolve().parent.parent
# clean examples get full weight, questionable ones get half and excluded ones get zero
WEIGHTS = {"clean": 1.0, "questionable": 0.5, "exclude": 0.0}
EXCLUDE_BELOW_SECONDS = 720
QUESTIONABLE_BELOW_SECONDS = 1200
HIGH_DEATHS = 15
LOW_GPM = 338
LOW_NET_WORTH = 7286
# these opendota status codes indicate serious leaving or abandoning
SEVERE_LEAVER_CODES = {2, 3, 4, 5, 6}
FINAL_FIELDS = ("kills", "deaths", "assists", "last_hits", "denies", "gpm",
                "xpm", "final_net_worth", "hero_damage", "tower_damage", "hero_healing")
FLAG_COLUMNS = (
    "candidate_abandon", "other_abandon", "missing_telemetry",
    "incomplete_match_data", "invalid_final_stats", "extremely_short_match",
    "short_match", "high_deaths", "low_gpm", "low_net_worth",
)
EXCLUDE_FLAGS = ("candidate_abandon", "missing_telemetry", "incomplete_match_data",
                 "invalid_final_stats", "extremely_short_match")
QUESTIONABLE_FLAGS = ("other_abandon", "short_match", "high_deaths", "low_gpm",
                      "low_net_worth")
COLUMNS = ("match_id", "side", "candidate_hero_id", "candidate_hero_name",
           "candidate_player_slot", "quality_status", "training_weight") + FLAG_COLUMNS


# reads csv rows one at a time
def rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


# converts a value to a number or returns none if it is missing or invalid
def value(raw):
    if raw in (None, ""):
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        return None
    # rejects infinity and nan because they are not usable stats
    return result if math.isfinite(result) else None


# checks the abandon count and leaver status of a player
def abandoned(player):
    count = value(player.get("abandons"))
    status = value(player.get("leaver_status"))
    return (count is not None and count > 0) or status in SEVERE_LEAVER_CODES


# checks one training example and assigns its quality status and weight
def quality_row(example, match, candidate, other_abandon=False, player_count=10):
    candidate_id = int(example["candidate_hero_id"])
    result = {
        "match_id": example["match_id"], "side": example["side"],
        "candidate_hero_id": candidate_id,
        "candidate_hero_name": hero_id_to_name(candidate_id),
        "candidate_player_slot": example["candidate_player_slot"],
    }
    duration = value(match.get("duration")) if match else None
    # checks that the player record belongs to the example's hero and team
    matched = (candidate is not None and
               candidate.get("hero_id") == str(candidate_id) and
               candidate.get("side") == example["side"])
    # stores each problem as 1 when present and 0 when absent
    # not matched stops the player checks before accessing a missing record
    result.update({
        "candidate_abandon": int(matched and abandoned(candidate)),
        "other_abandon": int(other_abandon),
        "missing_telemetry": int(not matched or any(
            value(candidate.get(field)) is None or value(candidate.get(field)) < 0
            for field in ("lh_10", "gold_10"))),
        "incomplete_match_data": int(match is None or player_count != 10 or
                                     value(match.get("player_data_complete_count")) != 10),
        "invalid_final_stats": int(not matched or any(
            value(candidate.get(field)) is None or value(candidate.get(field)) < 0
            for field in FINAL_FIELDS) or duration is None or duration <= 0),
        "extremely_short_match": int(duration is not None and 0 < duration < EXCLUDE_BELOW_SECONDS),
        "short_match": int(duration is not None and
                           EXCLUDE_BELOW_SECONDS <= duration < QUESTIONABLE_BELOW_SECONDS),
        "high_deaths": int(matched and value(candidate.get("deaths")) is not None and
                           value(candidate["deaths"]) >= HIGH_DEATHS),
        "low_gpm": int(matched and value(candidate.get("gpm")) is not None and
                       value(candidate["gpm"]) <= LOW_GPM),
        "low_net_worth": int(matched and value(candidate.get("final_net_worth")) is not None and
                             value(candidate["final_net_worth"]) <= LOW_NET_WORTH),
    })
    # exclusion takes priority over questionable flags
    # match wins and losses do not affect the quality status
    status = ("exclude" if any(result[key] for key in EXCLUDE_FLAGS) else
              "questionable" if any(result[key] for key in QUESTIONABLE_FLAGS) else "clean")
    result["quality_status"] = status
    result["training_weight"] = WEIGHTS[status]
    return result


# joins training examples with match and player stats and writes their quality rows
def build(input_dir, output_path):
    examples = list(rows(input_dir / "training_examples.csv"))
    # a set keeps unique match and side pairs so repeated examples can be detected
    decisions = {(row["match_id"], row["side"]) for row in examples}
    if len(decisions) != len(examples):
        raise ValueError("duplicate (match_id, side) in training_examples.csv")
    # uses match id and player slot together to find each example's carry
    wanted = {(row["match_id"], row["candidate_player_slot"]) for row in examples}
    match_ids = {row["match_id"] for row in examples}
    # keeps only matches used by the training examples
    matches = {row["match_id"]: row for row in rows(input_dir / "match_features.csv")
               if row["match_id"] in match_ids}
    candidates = {}
    severe_players = Counter()
    player_counts = Counter()
    # counts all players and abandons but only stores the candidate player records
    for row in rows(input_dir / "player_match_features.csv"):
        match_id = row["match_id"]
        if match_id not in match_ids:
            continue
        player_counts[match_id] += 1
        if abandoned(row):
            severe_players[match_id] += 1
        key = (match_id, row["player_slot"])
        if key in wanted:
            if key in candidates:
                raise ValueError(f"duplicate candidate player: {key}")
            candidates[key] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    statuses = Counter()
    reasons = Counter()
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        writer.writeheader()
        for example in examples:
            match_id = example["match_id"]
            candidate = candidates.get((match_id, example["candidate_player_slot"]))
            # removes the candidate's abandon from the total to check other players separately
            others = severe_players[match_id] - int(candidate is not None and abandoned(candidate))
            row = quality_row(example, matches.get(match_id), candidate,
                              others > 0, player_counts[match_id])
            writer.writerow(row)
            # counts statuses and individual problems for the console summary
            statuses[row["quality_status"]] += 1
            for flag in FLAG_COLUMNS:
                reasons[flag] += row[flag]
    return statuses, reasons


# builds the quality csv and prints the status and problem counts
def main():
    statuses, reasons = build(ROOT / "data" / "processed", ROOT / "data" / "processed" / "training_quality.csv")
    print("status counts:", dict(statuses))
    print("reason counts:", dict(reasons))


if __name__ == "__main__":
    main()
