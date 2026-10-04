#!/usr/bin/env python3
# builds pos1 draft decisions from locally saved OpenDota matches
import csv
import json
from collections import Counter

from lastpick.config import project_root

if __package__:
    from lastpick.legacy.draft_csv import validate_match
else:
    from lastpick.legacy.draft_csv import validate_match


data_dir = project_root() / "data"
default_input_dir = data_dir / "raw" / "opendota" / "matches"
default_output_path = data_dir / "processed" / "training_examples.csv"

csv_columns = [
    "match_id",
    "side",
    "candidate_hero_id",
    "candidate_player_slot",
    "candidate_team_pick_number",
    "candidate_global_pick_order",
    "ally_1",
    "ally_2",
    "ally_3",
    "ally_4",
    "enemy_1",
    "enemy_2",
    "enemy_3",
    "enemy_4",
    "known_ally_count",
    "known_enemy_count",
    "pos1_confidence",
    "patch",
    "start_time",
    "radiant_win",
    "win",
]


# finds the carry by comparing safelane farm at 10 minutes
def infer_pos1(players: list, team: int) -> tuple[dict | None, str]:
    slots = range(5) if team == 0 else range(128, 133)
    team_players = [p for p in players if isinstance(p, dict) and p.get("player_slot") in slots]
    if len(team_players) != 5 or len({p["player_slot"] for p in team_players}) != 5:
        return None, "missing_required_data"
    if any(
        p.get("lane_role") is None or not isinstance(p.get("hero_id"), int) for p in team_players
    ):
        return None, "missing_required_data"

    # OpenDota lane_role 1 is safelane; its two players are usually pos1 and pos5
    safelane = [p for p in team_players if p["lane_role"] == 1]
    if len(safelane) != 2:
        return None, "ambiguous_pos1"

    for player in safelane:
        for field in ("lh_t", "gold_t"):
            timeline = player.get(field)
            if not isinstance(timeline, list) or len(timeline) <= 10:
                return None, "missing_required_data"
            if not isinstance(timeline[10], (int, float)):
                return None, "missing_required_data"

    safelane.sort(key=lambda p: p["lh_t"][10], reverse=True)
    farmed, other = safelane
    if farmed["lh_t"][10] - other["lh_t"][10] < 15:
        return None, "ambiguous_pos1"
    if farmed["gold_t"][10] - other["gold_t"][10] < 400:
        return None, "ambiguous_pos1"
    return farmed, "high"


# creates training examples from the visible picks in one match
def build_match_examples(match: dict) -> tuple[list[dict], Counter, bool]:
    skipped = Counter()
    if not isinstance(match, dict) or match.get("game_mode") != 22 or match.get("lobby_type") != 7:
        skipped["invalid_draft"] += 1
        return [], skipped, False

    valid, _, draft = validate_match(match)
    if not valid:
        skipped["invalid_draft"] += 1
        return [], skipped, False

    picks = [(int(draft[f"pick_{i}_team"]), int(draft[f"pick_{i}_hero_id"])) for i in range(10)]
    extra_pick_events = json.loads(draft["extra_pick_events"])
    players = match.get("players")
    if not isinstance(players, list):
        players = []

    examples = []
    for team in (0, 1):
        player, confidence = infer_pos1(players, team)
        if player is None:
            skipped[confidence] += 1
            continue

        candidate = player["hero_id"]
        candidate_orders = [
            i
            for i, (pick_team, hero_id) in enumerate(picks)
            if pick_team == team and hero_id == candidate
        ]
        if len(candidate_orders) != 1:
            skipped["missing_required_data"] += 1
            continue
        candidate_order = candidate_orders[0]
        known_allies = [
            hero_id for pick_team, hero_id in picks[:candidate_order] if pick_team == team
        ]
        team_pick_number = len(known_allies) + 1
        if team_pick_number in (1, 2):
            skipped[f"pos1_pick_{team_pick_number}"] += 1
            continue
        candidate_phase = 2 if team_pick_number in (3, 4) else 3
        if any(event["phase"] == candidate_phase for event in extra_pick_events):
            skipped["candidate_phase_extra_attempt"] += 1
            continue

        # enemy picks in the candidates current phase are hidden even when OpenDota places them earlier in the global pick order
        visible_enemy_count = 2 if team_pick_number in (3, 4) else 4
        known_enemies = [hero_id for pick_team, hero_id in picks if pick_team != team]
        known_enemies = known_enemies[:visible_enemy_count]

        example = {
            "match_id": draft["match_id"],
            "side": "radiant" if team == 0 else "dire",
            "candidate_hero_id": candidate,
            "candidate_player_slot": player["player_slot"],
            "candidate_team_pick_number": team_pick_number,
            "candidate_global_pick_order": draft[f"pick_{candidate_order}_order"],
            "known_ally_count": len(known_allies),
            "known_enemy_count": len(known_enemies),
            "pos1_confidence": confidence,
            "patch": draft["patch"],
            "start_time": draft["start_time"],
            "radiant_win": int(draft["radiant_win"]),
            "win": int(draft["radiant_win"] if team == 0 else not draft["radiant_win"]),
        }
        for i in range(4):
            example[f"ally_{i + 1}"] = known_allies[i] if i < len(known_allies) else ""
            example[f"enemy_{i + 1}"] = known_enemies[i] if i < len(known_enemies) else ""
        examples.append(example)

    return examples, skipped, True


# reads saved matches and writes the training dataset to csv
def main():
    default_output_path.parent.mkdir(parents=True, exist_ok=True)
    inspected = 0
    valid_matches = 0
    match_rows = Counter()
    skipped = Counter()
    team_picks = Counter()
    sides = Counter()
    shapes = Counter()

    with default_output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_columns)
        writer.writeheader()
        for path in sorted(default_input_dir.glob("*.json")):
            inspected += 1
            try:
                with path.open(encoding="utf-8") as source:
                    match = json.load(source)
                examples, reasons, valid = build_match_examples(match)
            except (OSError, ValueError, TypeError):
                examples, reasons, valid = [], Counter({"invalid_draft": 1}), False
            valid_matches += int(valid)
            skipped.update(reasons)
            match_rows[len(examples)] += 1
            for example in examples:
                writer.writerow(example)
                team_picks[example["candidate_team_pick_number"]] += 1
                sides[example["side"]] += 1
                shapes[(example["known_ally_count"], example["known_enemy_count"])] += 1
            if inspected % 20000 == 0:
                print(f"inspected {inspected} raw matches", flush=True)

    print(f"raw matches inspected: {inspected}")
    print(f"valid matches: {valid_matches}")
    print(f"total training examples: {sum(match_rows[i] * i for i in match_rows)}")
    print(f"matches with 2/1/0 examples: {match_rows[2]}/{match_rows[1]}/{match_rows[0]}")
    print(f"pos1 team picks 3/4/5: {team_picks[3]}/{team_picks[4]}/{team_picks[5]}")
    print(f"skipped teams, ambiguous pos1: {skipped['ambiguous_pos1']}")
    print(f"skipped teams, pos1 pick 1/2: {skipped['pos1_pick_1'] + skipped['pos1_pick_2']}")
    print(f"skipped matches, invalid draft: {skipped['invalid_draft']}")
    print(f"skipped teams, missing required data: {skipped['missing_required_data']}")
    print(
        f"skipped teams, extra attempt in candidate phase: {skipped['candidate_phase_extra_attempt']}"
    )
    print(f"radiant/dire examples: {sides['radiant']}/{sides['dire']}")
    print(f"visibility shapes: {dict(sorted(shapes.items()))}")
    print(f"output: {default_output_path}")


if __name__ == "__main__":
    main()
