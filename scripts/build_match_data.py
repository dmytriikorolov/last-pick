#!/usr/bin/env python3
"""extracts match stats and events from saved opendota files"""

import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

if __package__:
    from .hero_lookup import names_by_id
else:
    from hero_lookup import names_by_id


root = Path(__file__).resolve().parent.parent
default_input_dir = root / "data" / "raw" / "opendota" / "matches"
default_output_dir = root / "data" / "processed"
item_source = root / "data" / "constants" / "dotaconstants_items.json"

match_columns = [
    "match_id", "start_time", "patch", "duration", "radiant_win", "game_mode",
    "lobby_type", "has_leaver", "player_data_complete_count",
    "has_parsed_time_series", "objective_event_count",
]
player_columns = [
    "match_id", "player_slot", "side", "hero_id", "hero_name", "lane_role",
    "position_est", "kills", "deaths", "assists", "last_hits", "denies",
    "gpm", "xpm", "final_net_worth", "hero_damage", "tower_damage",
    "hero_healing",
]
for minute in (10, 20, 30):
    player_columns += [f"lh_{minute}", f"gold_{minute}", f"xp_{minute}"]
player_columns += [f"item_{i}" for i in range(6)]
player_columns += [f"backpack_{i}" for i in range(3)]
player_columns += ["aghanims_scepter", "aghanims_shard", "leaver_status", "abandons"]
purchase_columns = [
    "match_id", "player_slot", "hero_id", "hero_name", "item", "item_key",
    "purchase_time",
]
ability_columns = ["match_id", "player_slot", "hero_id", "hero_name",
                   "ability_id", "upgrade_order"]
objective_columns = ["match_id", "event_time", "event_type", "team",
                     "player_slot", "key", "value"]

tracked_player_fields = [
    "lane_role", "position_est", "kills", "deaths", "assists", "last_hits",
    "denies", "gold_per_min", "xp_per_min", "net_worth", "hero_damage",
    "tower_damage", "hero_healing", "lh_t", "gold_t", "xp_t",
    "purchase_log", "ability_upgrades_arr", "item_0", "item_5", "backpack_0",
    "aghanims_scepter", "aghanims_shard", "leaver_status", "abandons",
]
complete_fields = [
    "hero_id", "kills", "deaths", "assists", "last_hits", "denies",
    "gold_per_min", "xp_per_min", "net_worth", "hero_damage", "tower_damage",
    "hero_healing",
]


# gets a player's stat at a given minute or returns a blank if unavailable
def snapshot(player, field, minute, duration):
    timeline = player.get(field)
    # skips snapshots from after the match ended
    if duration is not None and duration < minute * 60:
        return ""
    if not isinstance(timeline, list) or len(timeline) <= minute:
        return ""
    value = timeline[minute]
    # python treats booleans as numbers so they need a separate check
    # very nice language
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else ""


# creates match, player, purchase, ability and objective rows for one match
def build_match_rows(match, item_names):
    match_id = match["match_id"]
    duration = match.get("duration")
    players = match.get("players")
    if not isinstance(players, list):
        players = []
    # opendota uses slots 0 to 4 for radiant and 128 to 132 for dire
    valid_slots = set(range(5)) | set(range(128, 133))
    selected = []
    seen_slots = set()
    # keeps only valid players and ignores repeated slots
    for player in players:
        if not isinstance(player, dict):
            continue
        slot = player.get("player_slot")
        if not isinstance(slot, int) or isinstance(slot, bool) or \
                slot not in valid_slots or slot in seen_slots:
            continue
        selected.append(player)
        seen_slots.add(slot)

    player_rows, purchase_rows, ability_rows = [], [], []
    for player in selected:
        slot = player["player_slot"]
        hero_id = player.get("hero_id")
        hero_name = names_by_id.get(hero_id, "")
        # missing stats stay blank in the csv instead of becoming zeros
        row = {
            "match_id": match_id, "player_slot": slot,
            "side": "radiant" if slot < 128 else "dire",
            "hero_id": hero_id, "hero_name": hero_name,
            "lane_role": player.get("lane_role"),
            "position_est": player.get("position_est"),
            "kills": player.get("kills"), "deaths": player.get("deaths"),
            "assists": player.get("assists"), "last_hits": player.get("last_hits"),
            "denies": player.get("denies"), "gpm": player.get("gold_per_min"),
            "xpm": player.get("xp_per_min"),
            "final_net_worth": player.get("net_worth"),
            "hero_damage": player.get("hero_damage"),
            "tower_damage": player.get("tower_damage"),
            "hero_healing": player.get("hero_healing"),
            "aghanims_scepter": player.get("aghanims_scepter"),
            "aghanims_shard": player.get("aghanims_shard"),
            "leaver_status": player.get("leaver_status"),
            "abandons": player.get("abandons"),
        }
        # copies farm stats at 10, 20 and 30 minutes when available
        for minute in (10, 20, 30):
            for source, name in (("lh_t", "lh"), ("gold_t", "gold"), ("xp_t", "xp")):
                row[f"{name}_{minute}"] = snapshot(player, source, minute, duration)
        # saves final item and backpack ids without guessing purchase times
        for i in range(6):
            row[f"item_{i}"] = player.get(f"item_{i}")
        for i in range(3):
            row[f"backpack_{i}"] = player.get(f"backpack_{i}")
        player_rows.append(row)

        # keeps item keys and raw times including purchases before the match starts
        for purchase in player.get("purchase_log") or []:
            if not isinstance(purchase, dict):
                continue
            key = purchase.get("key")
            if not isinstance(key, str) or not key:
                continue
            purchase_rows.append({
                "match_id": match_id, "player_slot": slot, "hero_id": hero_id,
                "hero_name": hero_name, "item": item_names.get(key, key),
                "item_key": key, "purchase_time": purchase.get("time"),
            })

        # upgrade order shows the sequence of learned abilities rather than the time
        upgrades = player.get("ability_upgrades_arr")
        if isinstance(upgrades, list):
            for order, ability_id in enumerate(upgrades, 1):
                if isinstance(ability_id, int) and not isinstance(ability_id, bool):
                    ability_rows.append({
                        "match_id": match_id, "player_slot": slot,
                        "hero_id": hero_id, "hero_name": hero_name,
                        "ability_id": ability_id, "upgrade_order": order,
                    })

    # copies objective events such as building kills with their available fields
    objective_rows = []
    objectives = match.get("objectives")
    if isinstance(objectives, list):
        for event in objectives:
            if isinstance(event, dict):
                objective_rows.append({
                    "match_id": match_id, "event_time": event.get("time"),
                    "event_type": event.get("type"), "team": event.get("team"),
                    "player_slot": event.get("player_slot"),
                    "key": event.get("key"), "value": event.get("value"),
                })

    # leaves the leaver flag blank when all leaver data is missing
    leaver_fields = [player.get(field) for player in selected
                     for field in ("leaver_status", "abandons")]
    has_leaver = "" if all(value is None for value in leaver_fields) else int(
        any(isinstance(value, (int, float)) and value > 0 for value in leaver_fields))
    # counts complete player records and checks for any player with all three timelines
    match_row = {
        "match_id": match_id, "start_time": match.get("start_time"),
        "patch": match.get("patch"), "duration": duration,
        "radiant_win": int(match["radiant_win"]) if isinstance(match.get("radiant_win"), bool) else "",
        "game_mode": match.get("game_mode"), "lobby_type": match.get("lobby_type"),
        "has_leaver": has_leaver,
        "player_data_complete_count": sum(all(player.get(field) is not None
                                               for field in complete_fields)
                                          for player in selected),
        "has_parsed_time_series": int(any(all(isinstance(player.get(field), list)
                                              and player[field] for field in
                                              ("lh_t", "gold_t", "xp_t"))
                                          for player in selected)),
        "objective_event_count": len(objective_rows) if isinstance(objectives, list) else "",
    }
    return match_row, player_rows, purchase_rows, ability_rows, objective_rows


# formats a percentage or returns n/a when there is no data
def percentage(count, total):
    return f"{100 * count / total:.1f}%" if total else "n/a"


# reads saved matches and writes the five csv tables
def main():
    # uses the raw item key when a display name is missing
    with item_source.open(encoding="utf-8") as source:
        item_names = {key: value.get("dname") or key
                      for key, value in json.load(source).items()}

    outputs = {
        "match_features.csv": match_columns,
        "player_match_features.csv": player_columns,
        "item_purchase_events.csv": purchase_columns,
        "ability_upgrade_events.csv": ability_columns,
        "objective_events.csv": objective_columns,
    }
    default_output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    available = Counter()
    seen_matches = set()
    # closes all five output files automatically even if processing fails
    with ExitStack() as stack:
        writers = {}
        for filename, columns in outputs.items():
            file = stack.enter_context((default_output_dir / filename).open(
                "w", newline="", encoding="utf-8"))
            writers[filename] = csv.DictWriter(file, fieldnames=columns)
            writers[filename].writeheader()

        # skips unreadable files and missing or repeated match ids
        for path in sorted(default_input_dir.glob("*.json")):
            counts["raw_files"] += 1
            try:
                with path.open(encoding="utf-8") as source:
                    match = json.load(source)
            except (OSError, ValueError):
                counts["invalid_json"] += 1
                continue
            if not isinstance(match, dict) or not isinstance(match.get("match_id"), int):
                counts["missing_match_id"] += 1
                continue
            if match["match_id"] in seen_matches:
                counts["duplicate_match_id"] += 1
                continue
            seen_matches.add(match["match_id"])
            match_row, player_rows, purchase_rows, ability_rows, objective_rows = build_match_rows(
                match, item_names)
            # sends each type of row to its matching csv file
            for filename, rows in (
                ("match_features.csv", [match_row]),
                ("player_match_features.csv", player_rows),
                ("item_purchase_events.csv", purchase_rows),
                ("ability_upgrade_events.csv", ability_rows),
                ("objective_events.csv", objective_rows),
            ):
                writers[filename].writerows(rows)
                counts[filename] += len(rows)
            # counts available player fields for the console summary
            for player in match.get("players") or []:
                if not isinstance(player, dict):
                    continue
                counts["players_audited"] += 1
                for field in tracked_player_fields:
                    value = player.get(field)
                    if field in ("lh_t", "gold_t", "xp_t", "purchase_log",
                                 "ability_upgrades_arr"):
                        present = isinstance(value, list)
                    else:
                        present = value is not None
                    if present:
                        available[field] += 1
            if counts["raw_files"] % 20000 == 0:
                print(f"processed {counts['raw_files']} raw matches", flush=True)

    print(f"raw matches processed: {counts['raw_files']}")
    for filename in outputs:
        print(f"{filename}: {counts[filename]}")
    print(f"players audited: {counts['players_audited']}")
    for field in ("lane_role", "lh_t", "gold_t", "xp_t", "purchase_log",
                  "ability_upgrades_arr", "hero_damage", "tower_damage", "net_worth"):
        print(f"{field}: {percentage(available[field], counts['players_audited'])}")


if __name__ == "__main__":
    main()
