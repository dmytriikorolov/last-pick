#!/usr/bin/env python3
# collects high-mmr matches using the opendota api

import json
import os
import ssl
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import urllib.parse

if __package__:
    from .parse_drafts import validate_match as validate_final_draft
else:
    from parse_drafts import validate_match as validate_final_draft

# fix ssl errors on macos
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    try:
        ssl_context = ssl.create_default_context()
    except Exception:
        ssl_context = ssl._create_unverified_context()

api_url = "https://api.opendota.com/api"

# read api key from .env
env_file = Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENDOTA_API_KEY="):
            os.environ.setdefault("OPENDOTA_API_KEY", line.split("=", 1)[1].strip())

api_key = os.getenv("OPENDOTA_API_KEY", "")

# total limits across existing and newly collected data
max_total_requests = 150000
target_matches = 150000

# use 5 workers to download faster if we have an api key
max_workers = 5 if api_key else 1
request_delay_seconds = 0.05 if api_key else 1.2

# match filters
ranked_all_pick_game_mode = 22
ranked_lobby_type = 7
min_avg_rank_tier = 75 # 75 is the highest value saved in public_matches
min_num_rank_tier = 8 # need at least 8 players with public ranks
min_patch_id = 59 # patch 7.40
output_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "opendota"
timeout_seconds = 60

state_lock = threading.Lock()
rate_limit_lock = threading.Lock()

@dataclass
class CrawlerState:
    path: Path
    maximum_requests: int
    total_requests: int = 0
    cursor: int | None = None

    @classmethod
    def load(cls, path: Path, maximum_requests: int) -> "CrawlerState":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    path=path,
                    maximum_requests=maximum_requests,
                    total_requests=int(data.get("total_requests", 0)),
                    cursor=data.get("cursor"),
                )
            except Exception:
                pass
        return cls(path, maximum_requests)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_requests": self.total_requests,
            "cursor": self.cursor,
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def consume_request(self) -> None:
        with state_lock:
            if self.total_requests >= self.maximum_requests:
                print(f"Limit of {self.maximum_requests} API calls reached. Stopping")
                os._exit(0)
            self.total_requests += 1
            self.save()

    def set_cursor(self, cursor: int) -> None:
        with state_lock:
            self.cursor = cursor
            self.save()


def fetch_json(url: str, state: CrawlerState) -> Any:
    for attempt in range(5):
        # sleep so we don't hit the API rate limits (max 5 requests per second)
        if api_key:
            with rate_limit_lock:
                time.sleep(0.2)
                
        state.consume_request()
        request = Request(url, headers={"User-Agent": "isp-dota-recommender/0.1"})
        try:
            with urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code == 429 and attempt < 4:
                print(f"API rate limit hit. Sleeping 5s... (retry {attempt+1})")
                time.sleep(5.0)
                continue
            raise


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # save to a temp file first so threads don't corrupt the file
    tmp_path = path.with_suffix(f'.{threading.get_ident()}.tmp')
    tmp_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


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
    patch = match.get("patch", index_entry.get("patch", 0))
    duration = match.get("duration", 0) or 0

    if game_mode != ranked_all_pick_game_mode:
        return False, "wrong game mode"
    if lobby_type != ranked_lobby_type:
        return False, "wrong lobby type"
    if not isinstance(avg_rank_tier, (int, float)) or avg_rank_tier < min_avg_rank_tier:
        return False, "rank too low"
    if not isinstance(num_rank_tier, int) or num_rank_tier < min_num_rank_tier:
        return False, "not enough ranked players"
    if patch and patch < min_patch_id:
        return False, "old patch"
    if duration < 900:
        return False, "game is too short"

    return validate_draft(match)

# returns (match_metadata, was_downloaded, is_valid, reason)
# match_metadata: {match_id, rank, mode} | reason: "ok", "rank too low", "game is too short", or error msg
def fetch_and_validate_worker(entry: dict[str, Any], state: CrawlerState, output_dir: Path) -> tuple[dict[str, Any], bool, bool, str]:
    match_id = entry.get("match_id")
    detail_path = output_dir / "matches" / f"{match_id}.json"
    detail_url = f"{api_url}/matches/{match_id}"
    if api_key:
        detail_url += f"?api_key={api_key}"
        
    is_network = False
    try:
        match, is_network = load_cached_or_fetch(detail_path, detail_url, state)
        valid, reason = validate_match(match, entry)
        return entry, is_network, valid, reason
    except Exception as error:
        return entry, is_network, False, f"error: {error}"


def main():
    state = CrawlerState.load(output_dir / "api_crawler_state.json", max_total_requests)
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    existing_matches = sum(1 for _ in matches_dir.glob("*.json"))

    counts: dict[str, int] = {}
    accepted = 0      # newly accepted matches
    new_downloaded = 0
    seen_ids: set[int] = set()

    print("--- Collecting Immortal matches (Patch 7.40+) ---")
    print(f"Using API key: {'yes' if api_key else 'no'}")
    print(f"Max budget: {max_total_requests} requests")
    print(f"Requests used: {state.total_requests}/{max_total_requests}")
    print(f"Filters: rank >= {min_avg_rank_tier}, patch >= {min_patch_id}")
    print(f"Workers: {max_workers}")
    print(f"Matches on disk: {existing_matches}\n")

    while existing_matches + new_downloaded < target_matches and state.total_requests < state.maximum_requests:
        cursor_clause = f"AND match_id < {state.cursor}" if state.cursor else ""
        sql = f"""
            SELECT match_id, avg_rank_tier, game_mode, lobby_type
            FROM public_matches
            WHERE avg_rank_tier >= {min_avg_rank_tier}
              AND game_mode = {ranked_all_pick_game_mode}
              AND lobby_type = {ranked_lobby_type}
              {cursor_clause}
            ORDER BY match_id DESC
            LIMIT 1000
        """
        
        params = urllib.parse.urlencode({'sql': sql})
        url = f"{api_url}/explorer?{params}"
        if api_key:
            url += f"&api_key={api_key}"

        try:
            print(f"Fetching match IDs with SQL... (cursor={state.cursor})")
            explorer_result = fetch_json(url, state)
            time.sleep(request_delay_seconds)
            
            rows = explorer_result.get("rows", [])
            if not rows:
                print("No more matches found.")
                break
                
        except Exception as error:
            print(f"SQL query failed: {error}")
            if state.cursor:
                print("Query timed out. Skipping 200,000 matches backwards")
                state.set_cursor(state.cursor - 200000)
                time.sleep(2.0)
                continue
            else:
                break

        cursor = min(row["match_id"] for row in rows if "match_id" in row)
        state.set_cursor(cursor)

        # start workers
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            remaining = target_matches - existing_matches - new_downloaded
            for entry in rows:
                if state.total_requests >= state.maximum_requests or len(futures) >= remaining:
                    break
                match_id = entry.get("match_id")
                if not match_id or match_id in seen_ids:
                    continue
                seen_ids.add(match_id)
                futures.append(executor.submit(fetch_and_validate_worker, entry, state, output_dir))
            
            for future in as_completed(futures):
                entry, is_network, valid, reason = future.result()
                counts[reason] = counts.get(reason, 0) + 1
                if is_network:
                    new_downloaded += 1
                
                if valid and is_network:
                    accepted += 1
                    if new_downloaded % 50 == 0 or existing_matches + new_downloaded >= target_matches:
                        match_id = entry.get("match_id")
                        print(f"Downloaded match {match_id} (raw files: {existing_matches + new_downloaded}, requests used: {state.total_requests})")

    print("\nDone collecting.")
    print(f"Raw matches on disk: {existing_matches + new_downloaded}/{target_matches}")
    print(f"New matches accepted: {accepted}")
    print(f"Newly downloaded: {new_downloaded}")
    print(f"Total API requests used: {state.total_requests}/{state.maximum_requests}")
    if counts:
        print("Filter results:")
        for reason, count in sorted(counts.items()):
            print(f"  {reason}: {count}")

if __name__ == "__main__":
    main()
