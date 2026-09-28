#!/usr/bin/env python3
# collect public high-rank matches using free opendota api (3000 requests/day limit)

import json
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

if __package__:
    from .parse_drafts import validate_match as validate_final_draft
else:
    from parse_drafts import validate_match as validate_final_draft

# macos ssl certificate fallback
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    try:
        ssl_context = ssl.create_default_context()
    except Exception:
        ssl_context = ssl._create_unverified_context()

# configuration
api_url = "https://api.opendota.com/api"
ranked_all_pick_game_mode = 22  # ranked all pick
ranked_lobby_type = 7 # ranked matchmaking
min_avg_rank_tier = 70 # divine rank or higher (approx 4600+ mmr)
min_num_rank_tier = 5 # at least 5 players with rank
free_daily_budget = 3000 # free daily api limit
target_matches = 150000 # total raw match files, including existing files
request_delay_seconds = 1.2 # rate limit for free tier
output_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "opendota"
timeout_seconds = 20.0


@dataclass
class CrawlerState:
    path: Path
    maximum_requests: int
    used_requests: int = 0
    cursor: int | None = None

    @classmethod
    def load(cls, path: Path, maximum_requests: int) -> "CrawlerState":
        today = date.today().isoformat()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cursor = data.get("cursor")
                if data.get("date") == today:
                    return cls(path, maximum_requests, int(data.get("requests", 0)), cursor)
                return cls(path, maximum_requests, 0, cursor)
            except Exception:
                pass
        return cls(path, maximum_requests)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "date": date.today().isoformat(),
            "requests": self.used_requests,
            "cursor": self.cursor,
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def consume_request(self) -> None:
        if self.used_requests >= self.maximum_requests:
            raise RuntimeError("daily request budget exhausted")
        self.used_requests += 1
        self.save()

    def set_cursor(self, cursor: int) -> None:
        self.cursor = cursor
        self.save()


def fetch_json(url: str, state: CrawlerState) -> Any:
    for attempt in range(3):
        state.consume_request()
        request = Request(url, headers={"User-Agent": "isp-dota-recommender/0.1"})
        try:
            with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429 and attempt < 2:
                time.sleep(5.0)
                continue
            raise


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_cached_or_fetch(path: Path, url: str, state: CrawlerState) -> tuple[Any, bool]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), False
    value = fetch_json(url, state)
    save_json(path, value)
    time.sleep(request_delay_seconds)
    return value, True


def validate_draft(match: dict[str, Any]) -> tuple[bool, str]:
    valid, reason, _ = validate_final_draft(match)
    return valid, reason


def validate_match(match: dict[str, Any], index_entry: dict[str, Any]) -> tuple[bool, str]:
    game_mode = match.get("game_mode", index_entry.get("game_mode"))
    lobby_type = match.get("lobby_type", index_entry.get("lobby_type"))
    avg_rank_tier = match.get("avg_rank_tier") or index_entry.get("avg_rank_tier")
    num_rank_tier = match.get("num_rank_tier") or index_entry.get("num_rank_tier")

    if game_mode != ranked_all_pick_game_mode:
        return False, "unsupported game mode"
    if lobby_type != ranked_lobby_type:
        return False, "unsupported lobby type"
    if not isinstance(avg_rank_tier, (int, float)) or avg_rank_tier < min_avg_rank_tier:
        return False, "below min avg rank tier"
    if not isinstance(num_rank_tier, int) or num_rank_tier < min_num_rank_tier:
        return False, "insufficient rank coverage"
    return validate_draft(match)


def main():
    state = CrawlerState.load(output_dir / "crawler_state.json", free_daily_budget)
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    existing_matches = sum(1 for _ in matches_dir.glob("*.json"))

    counts: dict[str, int] = {}
    accepted = 0 # newly accepted matches
    new_downloaded = 0
    seen_ids: set[int] = set()

    if state.cursor:
        print(f"matches on disk: {existing_matches}")

    # paging
    while existing_matches + new_downloaded < target_matches and state.used_requests < state.maximum_requests:
        url = f"{api_url}/publicMatches" + (f"?less_than_match_id={state.cursor}" if state.cursor else "")
        index_path = output_dir / "index" / f"cursor_{state.cursor or 'latest'}.json"

        try:
            index, _ = load_cached_or_fetch(index_path, url, state)
        except Exception as error:
            print(f"failed to fetch public matches: {error}")
            break

        if not index:
            break

        # update and persist cursor to the oldest match id on this page
        cursor = min(entry["match_id"] for entry in index if "match_id" in entry)
        state.set_cursor(cursor)

        # process each match on the page
        for entry in index:
            if existing_matches + new_downloaded >= target_matches or state.used_requests >= state.maximum_requests:
                break

            match_id = entry.get("match_id")
            if not match_id or match_id in seen_ids:
                continue
            seen_ids.add(match_id)

            # skip non-ranked, low rank, or uncalibrated lobbies
            if (
                entry.get("game_mode") != ranked_all_pick_game_mode
                or entry.get("lobby_type") != ranked_lobby_type
                or (entry.get("avg_rank_tier") or 0) < min_avg_rank_tier
                or (entry.get("num_rank_tier") or 0) < min_num_rank_tier
            ):
                continue

            # fetch detailed match
            detail_path = output_dir / "matches" / f"{match_id}.json"
            detail_url = f"{api_url}/matches/{match_id}"
            try:
                match, is_network = load_cached_or_fetch(detail_path, detail_url, state)
            except Exception as error:
                print(f"match {match_id} failed: {error}")
                continue
            if is_network:
                new_downloaded += 1

            # validate draft and record
            valid, reason = validate_match(match, entry)
            counts[reason] = counts.get(reason, 0) + 1
            if valid and is_network:
                accepted += 1
                print(f"downloaded match {match_id} (raw files: {existing_matches + new_downloaded})")

    print("\ndone.")
    print(f"matches on disk: {existing_matches + new_downloaded}/{target_matches}")
    print(f"new matches accepted: {accepted}")
    print(f"new downloaded: {new_downloaded}")
    print(f"requests used today: {state.used_requests}/{state.maximum_requests}")
    if counts:
        print("filters summary:")
        for reason, count in sorted(counts.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
