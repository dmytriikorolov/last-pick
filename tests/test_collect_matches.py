import io
import json
from dataclasses import replace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import pytest
from test_draft_parsing import make_match

from lastpick.config import Config, IngestionSettings
from lastpick.ingestion.collector import BudgetExhausted, Collector
from lastpick.ingestion.opendota import FetchFailure, InvalidPayload, OpenDotaSource
from lastpick.ingestion.source import DiscoveryPage


def payload(match_id=123):
    match, _, _ = make_match(8)
    match.update(match_id=match_id, avg_rank_tier=80, num_rank_tier=10, duration=2400)
    return match


def provider(entries=(123,)):
    source = Mock(name="provider")
    source.name = "opendota"
    page = DiscoveryPage(tuple({"match_id": i} for i in entries), min(entries))
    source.discover_matches.side_effect = [page, DiscoveryPage((), None)]
    source.fetch_match.side_effect = lambda match_id: payload(match_id)
    return source


def test_duplicate_handling_and_cache_reuse(tmp_path):
    source = provider((123, 123))
    collector = Collector(tmp_path, Config(), source)
    assert collector.run()["downloaded"] == 1
    assert source.fetch_match.call_count == 1
    cached = Collector(tmp_path, Config(), provider())
    match, downloaded = cached.cache_or_fetch({"match_id": 123})
    assert not downloaded and match == payload()
    cached.source.fetch_match.assert_not_called()
    assert (
        json.loads((tmp_path / Config().paths.ingestion / "metadata/123.json").read_text())[
            "source"
        ]
        == "opendota"
    )


def test_restart_retains_unfinished_page_and_budget(tmp_path):
    source = provider((124, 123))
    source.fetch_match.side_effect = [payload(124), FetchFailure("offline")]
    collector = Collector(tmp_path, Config(), source)
    collector.consume_request()
    counts = collector.run()
    assert counts["accepted"] == 1 and counts["fetch_failure"] == 1
    assert collector.state.cursor is None and collector.state.pending == [{"match_id": 123}]
    restarted = Collector(tmp_path, Config(), provider())
    assert restarted.state.requests == 1
    restarted.source.discover_matches.side_effect = [DiscoveryPage((), None)]
    assert restarted.run()["accepted"] == 1
    restarted.source.fetch_match.assert_called_once_with(123)
    assert restarted.state.cursor == 123 and not restarted.state.pending


def test_validation_rejection_keeps_raw_data_and_is_distinct_from_failure(tmp_path):
    source = provider()
    source.fetch_match.side_effect = [dict(payload(), avg_rank_tier=1)]
    collector = Collector(tmp_path, Config(), source)
    counts = collector.run()
    assert counts["validation_rejection"] == 1 and "fetch_failure" not in counts
    assert (collector.raw_dir / "123.json").exists()
    report = json.loads((collector.directory / "outcomes/123.json").read_text())
    assert report["reason"] == "below_min_rank"


@pytest.mark.parametrize("invalid", [[], {}, {"match_id": 999}])
def test_invalid_detail_is_reported_without_poisoning_cache(tmp_path, invalid):
    source = provider()
    source.fetch_match.side_effect = [invalid]
    collector = Collector(tmp_path, Config(), source)
    assert collector.run()["invalid_payload"] == 1
    assert not (collector.raw_dir / "123.json").exists()
    assert (collector.directory / "invalid_payloads/123.json").exists()


def test_corrupted_existing_files_are_never_overwritten(tmp_path):
    collector = Collector(tmp_path, Config(), provider())
    collector.raw_dir.mkdir(parents=True)
    path = collector.raw_dir / "123.json"
    path.write_text("unfinished")
    assert collector.run()["invalid_payload"] == 1
    assert path.read_text() == "unfinished"
    collector.source.fetch_match.assert_not_called()
    collector.state_path.write_text("broken")
    with pytest.raises(ValueError):
        Collector(tmp_path, Config(), provider())


def test_budget_is_counted_before_request_and_persisted(tmp_path):
    config = replace(Config(), ingestion=IngestionSettings(request_budget=1, daily_budget=False))
    collector = Collector(tmp_path, config, provider())
    collector.consume_request()
    restarted = Collector(tmp_path, config, provider())
    with pytest.raises(BudgetExhausted):
        restarted.consume_request()
    assert restarted.state.requests == 1


def test_rate_limit_retries_and_honors_retry_after():
    consume, sleep = Mock(), Mock()
    source = OpenDotaSource(IngestionSettings(delay_seconds=0), Config().filters, consume, sleep)
    limited = HTTPError("https://example.test", 429, "limited", {"Retry-After": "7"}, None)
    with patch(
        "lastpick.ingestion.opendota.urlopen", side_effect=[limited, io.StringIO('{"ok": true}')]
    ) as network:
        assert source.request("test") == {"ok": True}
    assert network.call_count == consume.call_count == 2
    sleep.assert_any_call(7)


def test_permanent_http_error_never_exposes_secret_or_retries():
    source = OpenDotaSource(IngestionSettings(), Config().filters, Mock(), Mock())
    source.api_key = "secret-example"
    error = HTTPError("https://example.test?api_key=secret-example", 403, "forbidden", {}, None)
    with patch("lastpick.ingestion.opendota.urlopen", side_effect=error) as network:
        with pytest.raises(FetchFailure, match="http_403") as caught:
            source.request("test")
    assert network.call_count == 1 and "secret-example" not in str(caught.value)


@pytest.mark.parametrize("value", [{}, [{"match_id": True}], [{"match_id": 0}]])
def test_discovery_payload_shape(value):
    source = OpenDotaSource(IngestionSettings(), Config().filters, Mock(), Mock())
    with patch.object(source, "request", return_value=value):
        with pytest.raises(InvalidPayload):
            source.discover_matches(None)


def test_atomic_write_failure_preserves_previous_cache(tmp_path):
    from lastpick.data.io import atomic_json

    path = tmp_path / "match.json"
    path.write_text('{"previous": true}')
    with patch("pathlib.Path.replace", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            atomic_json(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"previous": True}


def test_cache_identity_mismatch_preserves_existing_data(tmp_path):
    from lastpick.data.io import atomic_json

    collector = Collector(tmp_path, Config(), provider())
    atomic_json(collector.raw_dir / "123.json", payload(999))
    with pytest.raises(InvalidPayload, match="cache_match_id_mismatch"):
        collector.cache_or_fetch({"match_id": 123})
    assert json.loads((collector.raw_dir / "123.json").read_text())["match_id"] == 999
    collector.source.fetch_match.assert_not_called()


def test_sparse_discovery_does_not_erase_previous_rank_metadata(tmp_path):
    collector = Collector(tmp_path, Config(), provider())
    collector.cache_or_fetch({"match_id": 123, "avg_rank_tier": 80, "num_rank_tier": 10})
    collector.cache_or_fetch({"match_id": 123, "avg_rank_tier": None})
    metadata = json.loads((collector.directory / "metadata/123.json").read_text())["discovery"]
    assert metadata["avg_rank_tier"] == 80 and metadata["num_rank_tier"] == 10


def test_daily_budget_reset_keeps_cursor_and_pending_work(tmp_path):
    collector = Collector(tmp_path, Config(), provider())
    collector.state.requests = Config().ingestion.request_budget
    collector.state.request_day = "2000-01-01"
    collector.state.cursor = 500
    collector.state.pending = [{"match_id": 123}]
    collector.save()
    restarted = Collector(tmp_path, Config(), provider())
    restarted.consume_request()
    assert restarted.state.requests == 1
    assert restarted.state.cursor == 500 and restarted.state.pending == [{"match_id": 123}]


def test_source_and_adapter_interfaces_support_another_payload_shape(tmp_path):
    from lastpick.data.opendota import OpenDotaAdapter

    class ExampleAdapter:
        name = "example"

        def identifier(self, value):
            return value.get("record", {}).get("id")

        def normalize(self, value, discovery=None):
            return replace(
                OpenDotaAdapter().normalize(value["record"]["detail"], discovery), source=self.name
            )

    source = provider()
    source.name = "example"
    source.fetch_match.side_effect = [{"record": {"id": 123, "detail": payload()}}]
    config = replace(Config(), source="example")
    collector = Collector(tmp_path, config, source, ExampleAdapter())
    assert collector.run()["accepted"] == 1
