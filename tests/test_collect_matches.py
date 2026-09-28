import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

from scripts import collect_matches_api as collector
from test_draft_parsing import make_match


# checks that a saved match is reused instead of downloaded again
def test_download_is_saved_and_not_repeated(tmp_path):
    match, _, _ = make_match(8)
    match.update(avg_rank_tier=80, num_rank_tier=10, duration=2400)
    state = collector.CrawlerState(tmp_path / "state.json", 10)
    path = tmp_path / "123.json"
    with patch.object(collector, "api_key", ""), \
            patch.object(collector, "urlopen", return_value=io.StringIO(json.dumps(match))) as network, \
            patch.object(collector.time, "sleep"):
        saved, downloaded = collector.load_cached_or_fetch(path, "https://example.test", state)
        assert downloaded and collector.validate_match(saved, {})[0]
        cached, downloaded = collector.load_cached_or_fetch(path, "https://example.test", state)
    assert not downloaded
    assert cached == json.loads(path.read_text()) == match
    assert network.call_count == 1


# checks that broken or low rank matches are rejected
def test_bad_response_is_rejected(tmp_path):
    state = collector.CrawlerState(tmp_path / "state.json", 10)
    with patch.object(collector, "load_cached_or_fetch", return_value=([], True)):
        _, _, valid, reason = collector.fetch_and_validate_worker({"match_id": 123}, state, tmp_path)
    assert not valid and reason.startswith("error:")
    match, _, _ = make_match(8)
    match.update(avg_rank_tier=1, num_rank_tier=10, duration=2400)
    assert not collector.validate_match(match, {})[0]


# checks that the collector retries after an api rate limit
def test_rate_limit_retries_without_real_requests(tmp_path):
    state = collector.CrawlerState(tmp_path / "state.json", 10)
    limited = HTTPError("https://example.test", 429, "limited", {}, None)
    with patch.object(collector, "api_key", ""), \
            patch.object(collector, "urlopen", side_effect=[limited, io.StringIO('{"ok": true}')]) as network, \
            patch.object(collector.time, "sleep"):
        assert collector.fetch_json("https://example.test", state) == {"ok": True}
    assert network.call_count == 2
