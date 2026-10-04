#!/usr/bin/env python3
# builds local hero features from the opendota constants
import csv
import json

from lastpick.config import project_root

constants_dir = project_root() / "data" / "constants"
default_source = constants_dir / "dotaconstants_heroes.json"
default_output = constants_dir / "heroes.csv"
roles = ("Carry", "Support", "Nuker", "Disabler", "Durable", "Escape", "Pusher", "Initiator")
csv_columns = ["hero_id", "hero_name", "primary_attr", "attack_type"] + [
    f"is_{role.lower()}" for role in roles
]


# checks hero details and creates a row for each hero
def build_rows(source):
    rows = []
    seen_names = set()
    for key, hero in sorted(source.items(), key=lambda item: int(item[0])):
        hero_id = int(key)
        name = hero["localized_name"]
        hero_roles = hero["roles"]
        if hero["id"] != hero_id or not name or name.casefold() in seen_names:
            raise ValueError(f"invalid or duplicate hero: {key}")
        if hero["primary_attr"] not in {"str", "agi", "int", "all"}:
            raise ValueError(f"invalid primary attribute for hero {key}")
        if hero["attack_type"] not in {"Melee", "Ranged"}:
            raise ValueError(f"invalid attack type for hero {key}")
        if not isinstance(hero_roles, list) or set(hero_roles) - set(roles):
            raise ValueError(f"unknown role for hero {key}")
        row = {
            "hero_id": hero_id,
            "hero_name": name,
            "primary_attr": hero["primary_attr"],
            "attack_type": hero["attack_type"],
        }
        row.update({f"is_{role.lower()}": int(role in hero_roles) for role in roles})
        rows.append(row)
        seen_names.add(name.casefold())
    return rows


# reads saved hero data and writes it to csv
def main():
    with default_source.open(encoding="utf-8") as source:
        rows = build_rows(json.load(source))
    default_output.parent.mkdir(parents=True, exist_ok=True)
    with default_output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"hero metadata rows: {len(rows)}")
    print(f"output: {default_output}")


if __name__ == "__main__":
    main()
