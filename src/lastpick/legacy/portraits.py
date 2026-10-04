# very easy file just to download the hero icons
import json
from urllib.request import urlopen

from lastpick.config import project_root

ROOT = project_root()
DESTINATION = ROOT / "web/static/heroes"


def download(hero):
    target = DESTINATION / f"{hero['id']}.png"
    if target.exists():
        return
    url = "https://cdn.cloudflare.steamstatic.com" + hero["img"]
    with urlopen(url, timeout=30) as response:
        target.write_bytes(response.read())


def main():
    heroes = json.loads((ROOT / "data/constants/dotaconstants_heroes.json").read_text())
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for hero in heroes.values():
        download(hero)
    print(f"Portraits saved in {DESTINATION}")


if __name__ == "__main__":
    main()
