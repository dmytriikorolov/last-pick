#!/usr/bin/env python3
# builds one complete 5v5 example

import csv
from pathlib import Path


data_dir = Path(__file__).resolve().parent.parent / "data"
default_input_path = data_dir / "processed" / "drafts.csv"
default_output_path = data_dir / "processed" / "full_draft_examples.csv"
csv_columns = (["match_id", "patch", "start_time"]
               + [f"radiant_{i}" for i in range(1, 6)]
               + [f"dire_{i}" for i in range(1, 6)]
               + ["radiant_win"])


# checks the final picks and creates one row with both teams and the result
def build_full_draft_row(draft: dict) -> dict:
    teams = {0: [], 1: []}
    heroes = []
    for i in range(10):
        try:
            team = int(draft[f"pick_{i}_team"])
            hero = int(draft[f"pick_{i}_hero_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing final pick {i}") from error
        if team not in teams or hero <= 0:
            raise ValueError(f"invalid final pick {i}")
        teams[team].append(hero)
        heroes.append(hero)

    if len(teams[0]) != 5 or len(teams[1]) != 5 or len(set(heroes)) != 10:
        raise ValueError("expected five heroes per team and ten unique final heroes")
    if draft["radiant_win"] not in {"True", "False"}:
        raise ValueError("invalid radiant_win")

    row = {"match_id": draft["match_id"], "patch": draft["patch"],
           "start_time": draft["start_time"],
           "radiant_win": int(draft["radiant_win"] == "True")}
    for team, name in ((0, "radiant"), (1, "dire")):
        for i, hero in enumerate(teams[team], 1):
            row[f"{name}_{i}"] = hero
    return row


# reads parsed drafts and writes the full draft dataset to csv
def main():

    seen_matches = set()
    distinct_heroes = set()
    count = 0
    default_output_path.parent.mkdir(parents=True, exist_ok=True)
    with default_input_path.open(newline="", encoding="utf-8") as source, default_output_path.open("w", newline="", encoding="utf-8") as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=csv_columns)
        writer.writeheader()
        for draft in reader:
            match_id = draft["match_id"]
            if not match_id or match_id in seen_matches:
                raise ValueError(f"missing or duplicate match_id: {match_id}")
            row = build_full_draft_row(draft)
            writer.writerow(row)
            seen_matches.add(match_id)
            distinct_heroes.update(row[f"{team}_{i}"] for team in ("radiant", "dire")
                                   for i in range(1, 6))
            count += 1

    print(f"full draft rows: {count}")
    print(f"distinct heroes: {len(distinct_heroes)}")
    print(f"output: {default_output_path}")


if __name__ == "__main__":
    main()
