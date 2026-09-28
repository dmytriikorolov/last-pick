#!/usr/bin/env python3
# builds hero statistics from saved match data

import csv
import statistics
from array import array
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

if __package__:
    from .hero_lookup import names_by_id
else:
    from hero_lookup import names_by_id


ROOT = Path(__file__).resolve().parent.parent
METRICS = [
    "gpm", "xpm", "last_hits", "final_net_worth", "gold_10", "gold_20",
    "gold_30", "lh_10", "lh_20", "lh_30", "kills", "deaths", "assists",
    "hero_damage", "hero_healing", "tower_damage",
]
METRIC_NAMES = {"final_net_worth": "net_worth", "hero_healing": "healing"}
STAGES = (("early", 0, 1800), ("mid", 1800, 2700), ("late", 2700, float("inf")))
AGHS_KEYS = {"ultimate_scepter", "ultimate_scepter_2"}


# reads csv rows one at a time
def rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)


# converts a value to a number and leaves missing values as none
def number(value):
    return float(value) if value not in (None, "") else None


# calculates an average or returns a blank when there is no data
def mean(total, count):
    return round(total / count, 4) if count else ""


# writes rows and column names to a csv file
def write(path, columns, data):
    with path.open("w", newline="", encoding="utf-8") as target:
        if data and isinstance(data[0], dict):
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(target)
            writer.writerow(columns)
            writer.writerows(data)


# counts games and wins for an ally pair
def add_pair(pairs, a, b, won):
    key = tuple(sorted((a, b)))
    pairs[key][0] += 1
    pairs[key][1] += won


# calculates win rates for heroes playing together and against each other
def pair_rows(drafts):
    hero_results = defaultdict(lambda: [0, 0])
    allies = defaultdict(lambda: [0, 0])
    enemies = defaultdict(lambda: [0, 0])
    for draft in drafts:
        radiant = [int(draft[f"radiant_{i}"]) for i in range(1, 6)]
        dire = [int(draft[f"dire_{i}"]) for i in range(1, 6)]
        radiant_win = int(draft["radiant_win"])
        for team, won in ((radiant, radiant_win), (dire, 1 - radiant_win)):
            for hero in team:
                hero_results[hero][0] += 1
                hero_results[hero][1] += won
            for a, b in combinations(team, 2):
                add_pair(allies, a, b, won)
        for a in radiant:
            for b in dire:
                enemies[a, b][0] += 1
                enemies[a, b][1] += radiant_win
                enemies[b, a][0] += 1
                enemies[b, a][1] += 1 - radiant_win
    ally_rows = []
    for (a, b), (games, wins) in sorted(allies.items()):
        reference = (hero_results[a][1] / hero_results[a][0] +
                     hero_results[b][1] / hero_results[b][0]) / 2
        ally_rows.append((a, names_by_id[a], b, names_by_id[b], games, wins,
                          mean(wins, games), round(wins / games - reference, 4)))
    enemy_rows = []
    for (a, b), (games, wins) in sorted(enemies.items()):
        reference = hero_results[a][1] / hero_results[a][0]
        enemy_rows.append((a, names_by_id[a], b, names_by_id[b], games, wins,
                           mean(wins, games), round(wins / games - reference, 4)))
    return ally_rows, enemy_rows


# combines saved match data into hero statistics tables
def build(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # looks up match results and duration using the match id
    matches = {}
    for row in rows(input_dir / "match_features.csv"):
        matches[row["match_id"]] = row

    # gives each new hero its own counters for totals and available values
    profiles = defaultdict(lambda: {"games": 0, "wins": 0,
                                    "totals": Counter(), "counts": Counter(),
                                    "stages": defaultdict(lambda: [0, 0])})
    log_coverage = Counter()
    lane_groups = defaultdict(list)
    # stores games, wins, combined gold, xp and last hits for each lane pair
    lane_pairs = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])

    # saves the lane pairs and their farm for one match
    def flush_lanes(match_id):
        for (side, _), team in lane_groups.items():
            if len(team) != 2 or team[0]["hero_id"] == team[1]["hero_id"]:
                continue
            # sorts hero ids so the same pair always uses the same key
            a, b = sorted(int(player["hero_id"]) for player in team)
            pair = lane_pairs[a, b]
            pair[0] += 1
            pair[1] += int(matches[match_id]["radiant_win"]) == (side == "radiant")
            for index, field in enumerate(("gold", "xp", "lh"), 2):
                pair[index] += sum(float(player[f"{field}_10"]) for player in team)

    current_match_id = None
    # expects player rows from the same match to be grouped together
    for row in rows(input_dir / "player_match_features.csv"):
        if row["match_id"] != current_match_id:
            if current_match_id is not None:
                # finishes the previous match before collecting the next one
                flush_lanes(current_match_id)
                lane_groups.clear()
            current_match_id = row["match_id"]
        hero = int(row["hero_id"])
        if hero not in names_by_id:
            raise ValueError(f"unknown hero ID: {hero}")
        match = matches[row["match_id"]]
        # checks whether this player's side matches the winning side
        won = int(match["radiant_win"]) == (row["side"] == "radiant")
        data = profiles[hero]
        data["games"] += 1
        data["wins"] += won
        if match["has_parsed_time_series"] == "1":
            log_coverage[hero] += 1
        duration = number(match["duration"])
        if duration is not None:
            for stage, start, end in STAGES:
                if start <= duration < end:
                    data["stages"][stage][0] += 1
                    data["stages"][stage][1] += won
                    break



        # counts each available value separately so missing values do not become zeros
        for field in METRICS + ["aghanims_scepter", "aghanims_shard"]:
            value = number(row[field])
            if value is not None:
                data["totals"][field] += value
                data["counts"][field] += 1
        # only collects safelane and offlane players with complete 10 minute farm data
        lane = row["lane_role"]
        if lane in ("1", "3") and all(row[f"{field}_10"] != "" for field in ("gold", "xp", "lh")):
            lane_groups[row["side"], lane].append(row)
    # saves the last match because no next match will trigger it
    if current_match_id is not None:
        flush_lanes(current_match_id)

    profile_columns = ["hero_id", "hero_name", "games", "wins", "win_rate"]
    for field in METRICS:
        label = METRIC_NAMES.get(field, field)
        profile_columns += [f"avg_{label}", f"{label}_count"]
    profile_columns += ["early_strength", "early_games", "mid_strength", "mid_games",
                        "late_strength", "late_games", "aghs_rate", "aghs_count",
                        "shard_rate", "shard_count", "aghs_median_purchase_time",
                        "aghs_avg_purchase_time", "aghs_purchase_games", "shard_median_purchase_time",
                        "shard_avg_purchase_time", "shard_purchase_games", "purchase_log_games"]

    # stores purchase counts, total times and all first purchase times for the median
    item_stats = defaultdict(lambda: [0, 0.0, array("i")])
    aghs_times = defaultdict(lambda: array("i"))
    shard_times = defaultdict(lambda: array("i"))
    current_player = None
    first_times = {}

    # saves the first item purchase times for one player
    def flush_purchases():
        if current_player is None:
            return
        match_id, _, hero = current_player
        patch = matches[match_id]["patch"]
        first_aghs = []
        for key, (name, time) in first_times.items():
            if time is None:
                continue
            item = item_stats[hero, key, patch, name]
            item[0] += 1
            item[1] += time
            item[2].append(int(time))
            if key in AGHS_KEYS:
                first_aghs.append(int(time))
            elif key == "aghanims_shard":
                shard_times[hero].append(int(time))
        # counts the earliest scepter purchase
        if first_aghs:
            aghs_times[hero].append(min(first_aghs))

    # expects purchases from the same player and match to be grouped together
    for row in rows(input_dir / "item_purchase_events.csv"):
        player = (row["match_id"], row["player_slot"], int(row["hero_id"]))
        if player != current_player:
            flush_purchases()
            first_times = {}
            current_player = player
        key = row["item_key"]
        time = number(row["purchase_time"])

        # keeps only the earliest known purchase of each item for this player
        if key not in first_times or (time is not None and
                                      (first_times[key][1] is None or time < first_times[key][1])):
            first_times[key] = (row["item"], time)
    # saves the last player's purchases
    flush_purchases()

    # turns collected totals into averages and win rates for each hero
    profile_rows = []
    for hero in sorted(names_by_id):
        data = profiles[hero]
        output = {"hero_id": hero, "hero_name": names_by_id[hero],
                  "games": data["games"], "wins": data["wins"],
                  "win_rate": mean(data["wins"], data["games"])}
        for field in METRICS:
            count = data["counts"][field]
            label = METRIC_NAMES.get(field, field)
            output[f"avg_{label}"] = mean(data["totals"][field], count)
            output[f"{label}_count"] = count
        for stage, _, _ in STAGES:
            games, wins = data["stages"][stage]
            output[f"{stage}_strength"] = mean(wins, games)
            output[f"{stage}_games"] = games
        for flag, label in (("aghanims_scepter", "aghs"), ("aghanims_shard", "shard")):
            count = data["counts"][flag]
            output[f"{label}_rate"] = mean(data["totals"][flag], count)
            output[f"{label}_count"] = count
            times = aghs_times[hero] if label == "aghs" else shard_times[hero]
            output[f"{label}_median_purchase_time"] = statistics.median(times) if times else ""
            output[f"{label}_avg_purchase_time"] = mean(sum(times), len(times))
            output[f"{label}_purchase_games"] = len(times)
        output["purchase_log_games"] = log_coverage[hero]
        profile_rows.append(output)
    write(output_dir / "hero_profiles.csv", profile_columns, profile_rows)

    # uses complete drafts to count ally pairs and enemy matchups
    ally_rows, enemy_rows = pair_rows(rows(input_dir / "full_draft_examples.csv"))
    write(output_dir / "hero_pair_synergy.csv", ["hero_a_id", "hero_a_name", "hero_b_id",
          "hero_b_name", "games_together", "wins_together", "win_rate_together",
          "synergy_delta"], ally_rows)
    write(output_dir / "hero_matchups.csv", ["hero_id", "hero_name", "enemy_id",
          "enemy_name", "games_against", "wins_against", "win_rate_against",
          "matchup_delta"], enemy_rows)

    item_rows = []
    for (hero, key, patch, name), (games, total, times) in sorted(item_stats.items()):
        item_rows.append((hero, names_by_id[hero], key, name, patch, games,
                          mean(total, games), statistics.median(times)))
    # includes parsed games without purchases when calculating purchase rates
    patch_coverage = Counter()
    for row in rows(input_dir / "player_match_features.csv"):
        match = matches[row["match_id"]]
        if match["has_parsed_time_series"] == "1":
            patch_coverage[int(row["hero_id"]), match["patch"]] += 1
    item_rows = [(hero, name, key, item_name, patch, patch_coverage[hero, patch], games,
                  mean(games, patch_coverage[hero, patch]), avg, median)
                 for hero, name, key, item_name, patch, games, avg, median in item_rows]
    write(output_dir / "hero_item_profiles.csv", ["hero_id", "hero_name", "item",
          "item_name", "patch", "games_with_purchase_log", "games_item_bought",
          "purchase_rate", "average_purchase_time", "median_purchase_time"], item_rows)

    # averages the combined farm of the two lane partners at 10 minutes
    lane_rows = [(a, names_by_id[a], b, names_by_id[b], games, wins, mean(wins, games),
                  mean(gold, games), mean(xp, games), mean(lh, games))
                 for (a, b), (games, wins, gold, xp, lh) in sorted(lane_pairs.items())]
    write(output_dir / "lane_pair_profiles.csv", ["hero_a_id", "hero_a_name", "hero_b_id",
          "hero_b_name", "games_laned_together", "wins", "win_rate",
          "avg_combined_gold_10", "avg_combined_xp_10", "avg_combined_lh_10"], lane_rows)

    # stores upgrade count, total upgrade order and number of player games
    abilities = defaultdict(lambda: [0, 0, 0])
    last_ability_game = {}
    for row in rows(input_dir / "ability_upgrade_events.csv"):
        hero, ability = int(row["hero_id"]), int(row["ability_id"])
        entry = abilities[hero, ability]
        entry[0] += 1
        entry[1] += int(row["upgrade_order"])
        game = (row["match_id"], row["player_slot"])
        # counts a player game once even when the same ability is upgraded several times
        if last_ability_game.get((hero, ability)) != game:
            entry[2] += 1
            last_ability_game[hero, ability] = game
    ability_rows = [(hero, names_by_id[hero], ability, count, games, mean(total, count))
                    for (hero, ability), (count, total, games) in sorted(abilities.items())]
    write(output_dir / "hero_ability_profiles.csv", ["hero_id", "hero_name", "ability_id",
          "upgrade_count", "games_with_upgrade", "average_upgrade_order"], ability_rows)

    # counts objective events without building a separate statistics table
    objective_count = sum(1 for _ in rows(input_dir / "objective_events.csv"))
    return {"heroes": len(profile_rows), "allies": len(ally_rows), "matchups": len(enemy_rows),
            "items": len(item_rows), "lanes": len(lane_rows), "abilities": len(ability_rows),
            "objectives_inspected": objective_count}


# builds the tables in the processed data folder and prints their sizes
def main():
    print(build(ROOT / "data" / "processed", ROOT / "data" / "processed"))


if __name__ == "__main__":
    main()
