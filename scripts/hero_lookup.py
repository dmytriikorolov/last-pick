# looks up hero names and ids from the saved hero metadata

import csv
from pathlib import Path


metadata_path = Path(__file__).resolve().parent.parent / "data" / "constants" / "heroes.csv"

# loads the hero table when this file is imported
with metadata_path.open(newline="", encoding="utf-8") as source:
    heroes = list(csv.DictReader(source))

# creates dictionaries for looking up names by id and ids by name
names_by_id = {int(row["hero_id"]): row["hero_name"] for row in heroes}
# casefold makes name lookup ignore capital letters
ids_by_name = {name.casefold(): hero_id for hero_id, name in names_by_id.items()}


# returns the name for a valid hero id
def hero_id_to_name(hero_id):
    return names_by_id[int(hero_id)]


# returns the hero id while ignoring spaces around the name and capital letters
def hero_name_to_id(name):
    return ids_by_name[name.strip().casefold()]


# formats a hero name with its id when requested
def format_hero(hero_id, include_id=False):
    name = hero_id_to_name(hero_id)
    return f"{name} ({int(hero_id)})" if include_id else name
