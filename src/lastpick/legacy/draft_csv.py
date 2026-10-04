#!/usr/bin/env python3
# convert raw opendota match json files into a flat csv of drafts
import csv
import json
from collections import Counter

from lastpick.config import project_root

# default directories
data_dir = project_root() / "data"
default_input_dir = data_dir / "raw" / "opendota" / "matches"
default_output_path = data_dir / "processed" / "drafts.csv"

# build csv header: match metadata + 10 successful picks and extra attempts
csv_columns = ["match_id", "start_time", "patch", "radiant_win", "extra_pick_events"]
for i in range(10):
    csv_columns.append(f"pick_{i}_order")
    csv_columns.append(f"pick_{i}_team")
    csv_columns.append(f"pick_{i}_hero_id")


def validate_match(match: dict) -> tuple[bool, str, dict | None]:
    if not isinstance(match, dict):
        return False, "match_not_object", None
    # basic match metadata checks
    if not isinstance(match.get("match_id"), int):
        return False, "missing match_id", None
    if not isinstance(match.get("radiant_win"), bool):
        return False, "missing radiant_win", None

    players = match.get("players")
    expected_slots = set(range(5)) | set(range(128, 133))
    if not isinstance(players, list) or len(players) != 10:
        return False, "invalid_final_players", None
    if any(
        not isinstance(p, dict)
        or p.get("player_slot") not in expected_slots
        or not isinstance(p.get("hero_id"), int)
        or isinstance(p.get("hero_id"), bool)
        for p in players
    ):
        return False, "invalid_final_players", None
    if {p["player_slot"] for p in players} != expected_slots:
        return False, "invalid_final_players", None
    final_teams = {p["hero_id"]: 0 if p["player_slot"] < 128 else 1 for p in players}
    if len(final_teams) != 10:
        return False, "duplicate_final_hero", None

    picks_bans = match.get("picks_bans")
    if not isinstance(picks_bans, list):
        return False, "missing picks_bans", None

    # include pick attempts that may not appear in the final played lineup
    picks = [p for p in picks_bans if isinstance(p, dict) and p.get("is_pick") is True]
    if len(picks) < 10:
        return False, "incomplete_final_picks", None

    # take care of order cause we need to locate extra attempts during draft phase
    orders = [p.get("order") for p in picks]
    if any(not isinstance(o, int) or isinstance(o, bool) for o in orders):
        return False, "invalid_pick_order", None
    if set(orders) != set(range(len(picks))):
        return False, "ambiguous_pick_orders", None

    picks = sorted(picks, key=lambda p: p["order"])
    if any(
        not isinstance(p.get("hero_id"), int)
        or isinstance(p.get("hero_id"), bool)
        or p.get("team") not in (0, 1)
        for p in picks
    ):
        return False, "invalid_pick_event", None

    successful = [p for p in picks if p["hero_id"] in final_teams]
    if len(successful) != 10 or {p["hero_id"] for p in successful} != set(final_teams):
        return False, "ambiguous_final_picks", None
    if any(p["team"] != final_teams[p["hero_id"]] for p in successful):
        return False, "final_pick_team_mismatch", None

    teams = [p["team"] for p in successful]

    # each team must pick exactly 5 heroes
    if Counter(teams) != Counter({0: 5, 1: 5}):
        return False, "team_pick_counts_not_5_each", None

    # phase 1: two picks per team before phase 2 begins
    if Counter(teams[:4]) != Counter({0: 2, 1: 2}):
        return False, "first_phase_pick_counts_not_2_each", None

    # rounds 1 and 2 (first 8 picks): exactly 4 picks per team
    if Counter(teams[:8]) != Counter({0: 4, 1: 4}):
        return False, "known_pick_counts_not_4_each", None

    # round 3 (last 2 picks): exactly 1 pick per team
    if Counter(teams[8:]) != Counter({0: 1, 1: 1}):
        return False, "final_pick_counts_not_1_each", None

    # construct flat row for csv
    extra_events = []
    successful_seen = 0
    for pick in picks:
        if pick["hero_id"] in final_teams:
            successful_seen += 1
        else:
            if successful_seen == 10:
                return False, "extra_after_final_pick", None
            phase = 1 if successful_seen < 4 else 2 if successful_seen < 8 else 3
            extra_events.append(
                {
                    "order": pick["order"],
                    "team": pick["team"],
                    "hero_id": pick["hero_id"],
                    "phase": phase,
                }
            )

    row = {
        "match_id": match["match_id"],
        "start_time": match.get("start_time", ""),
        "patch": match.get("patch", ""),
        "radiant_win": match["radiant_win"],
        "extra_pick_events": json.dumps(extra_events),
    }
    for i, p in enumerate(successful):
        row[f"pick_{i}_order"] = p["order"]
        row[f"pick_{i}_team"] = p["team"]
        row[f"pick_{i}_hero_id"] = p["hero_id"]

    return True, "valid", row


def main():
    if not default_input_dir.is_dir():
        print(f"input directory does not exist: {default_input_dir}")
        return

    rows = []
    rejected = Counter()
    normal_accepted = 0
    overfull_accepted = Counter()
    overfull_rejected = 0
    incomplete_rejected = 0

    # read all match files in sorted order
    for path in sorted(default_input_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                match = json.load(f)
            if not isinstance(match, dict):
                rejected["match_not_object"] += 1
                continue

            valid, reason, row = validate_match(match)
            pick_count = sum(
                p.get("is_pick") is True
                for p in match.get("picks_bans") or []
                if isinstance(p, dict)
            )
            if valid and row is not None:
                rows.append(row)
                if pick_count == 10:
                    normal_accepted += 1
                else:
                    overfull_accepted[pick_count] += 1
            else:
                rejected[reason] += 1
                if pick_count > 10:
                    overfull_rejected += 1
                if pick_count < 10:
                    incomplete_rejected += 1

        except Exception:
            rejected["invalid_json"] += 1

    # write parsed drafts to csv
    default_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(default_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"input files: {len(rows) + sum(rejected.values())}")
    print(f"accepted drafts: {len(rows)}")
    print(f"normal exactly-10 accepted: {normal_accepted}")
    print(f"overfull recovered: {sum(overfull_accepted.values())}")
    print(f"overfull still rejected: {overfull_rejected}")
    print(f"incomplete rejected: {incomplete_rejected}")
    print(f"recovered by event count: {dict(sorted(overfull_accepted.items()))}")
    print(f"rejected drafts: {sum(rejected.values())}")
    for reason, count in sorted(rejected.items()):
        print(f"  {reason}: {count}")
    print(f"output: {default_output_path}")


if __name__ == "__main__":
    main()
