import json

import pytest

from scripts.collect_weather_v4_dataset import Collector, collect, sha


class Response:
    def __init__(self, status=200, body=b'{"ok":true}'):
        self.status_code = status
        self.body = body
        self.headers = {"content-type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, _):
        yield self.body


def test_immutable_cache_validates_url_and_body(tmp_path, monkeypatch):
    c = Collector(tmp_path)
    calls = []
    monkeypatch.setattr(c.session, "get", lambda url, **kwargs: calls.append(url) or Response())
    assert c.get("weather/day", "https://example.test") == b'{"ok":true}'
    assert c.get("weather/day", "https://example.test") == b'{"ok":true}' and len(calls) == 1
    with pytest.raises(ValueError, match="URL mismatch"):
        c.get("weather/day", "https://changed.test")
    (tmp_path / "sources/weather/day/response.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        c.get("weather/day", "https://example.test")


def test_rate_limit_is_saved_and_not_automatically_retried(tmp_path, monkeypatch):
    c = Collector(tmp_path)
    monkeypatch.setattr(c.session, "get", lambda *args, **kwargs: Response(429, b"rate limit"))
    with pytest.raises(RuntimeError, match="no automatic retries"):
        c.get("weather/day", "https://example.test")
    assert c.get("weather/day", "https://example.test") is None and c.calls == 1
    receipt = json.loads((tmp_path / "sources/weather/day/receipt.json").read_bytes())
    assert receipt["status"] == 429 and receipt["sha256"] == sha(b"rate limit")


def test_protocol_drift_and_budget_stop_before_network(tmp_path):
    (tmp_path / "protocol.json").write_text(json.dumps({"protocol": {"changed": True}}))
    with pytest.raises(ValueError, match="protocol drift"):
        collect(tmp_path)
    c = Collector(tmp_path)
    c.calls = 300
    with pytest.raises(RuntimeError, match="budget reached"):
        c.get("next", "https://example.test")


def test_original_month_boundaries_leak_but_corrected_embargo_does_not():
    from scripts.build_weather_v4_dataset import validate_split_timing
    from scripts.collect_weather_v4_dataset import PROTOCOL

    with pytest.raises(ValueError, match="embargo"):
        validate_split_timing(PROTOCOL)
    corrected = dict(
        train=["2026-04-15", "2026-08-29"],
        calibration=["2026-09-01", "2026-09-28"],
        evaluation=["2026-10-01", "2026-11-03"],
    )
    result = validate_split_timing(corrected)
    assert [r["separation_hours"] for r in result] == [36, 36]


def test_missing_successful_cached_body_is_corruption(tmp_path, monkeypatch):
    c = Collector(tmp_path)
    monkeypatch.setattr(c.session, "get", lambda *args, **kwargs: Response())
    c.get("cached", "https://example.test")
    (tmp_path / "sources/cached/response.bin").unlink()
    with pytest.raises(ValueError, match="body missing"):
        c.get("cached", "https://example.test")
