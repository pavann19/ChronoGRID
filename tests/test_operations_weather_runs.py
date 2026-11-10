import json
from datetime import date, datetime, timedelta, timezone

import pytest

from chronogrid.operations.capture import delivery_grid
from chronogrid.operations.journal import Journal, encode
from chronogrid.operations.weather_runs import (
    SOURCE,
    RunRequest,
    capture_single_run,
    delivery_weather,
)


def fixture(day):
    initialization = datetime.combine(day - timedelta(days=1), datetime.min.time(), timezone.utc)
    req = RunRequest(initialization=initialization)
    body = dict(
        latitude=50.125,
        longitude=8.6875,
        elevation=103,
        utc_offset_seconds=0,
        hourly_units={
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "wind_speed_10m": "m/s",
            "direct_radiation": "W/m²",
        },
        hourly={
            "time": [(initialization + timedelta(hours=i)).isoformat() for i in range(72)],
            "temperature_2m": [10] * 72,
            "relative_humidity_2m": [80] * 72,
            "wind_speed_10m": [4] * 72,
            "direct_radiation": list(range(72)),
        },
    )
    return req, body


@pytest.mark.parametrize("day", [date(2027, 1, 15), date(2027, 3, 28), date(2027, 10, 31)])
def test_single_run_radiation_alignment_dst_and_lineage(tmp_path, day):
    req, body = fixture(day)
    clock = [req.initialization + timedelta(hours=7)]
    store = Journal(tmp_path / "weather.sqlite", lambda: clock[0])
    urls = []

    def fetch(url):
        urls.append(url)
        return encode(body)

    h = capture_single_run(store, req, fetch=fetch)
    clock[0] += timedelta(hours=1)
    result = delivery_weather(store, h, day, clock[0])
    assert urls == [req.url()] and "models=icon_eu" in urls[0] and "seamless" not in urls[0]
    assert len(result["features"]) == len(delivery_grid(day))
    for row, t in zip(result["features"], delivery_grid(day)):
        assert row["timestamp"] == t.isoformat()
        assert (
            row["direct_radiation_delivery_mean_wm2"]
            == int((t - req.initialization).total_seconds() / 3600) + 1
        )
        assert row["temperature_instant_c"] == 10
    assert len(result["inputs"]) == 3 and all(
        "not_independently_attested" in str(x["quality_flags"]) for x in result["inputs"]
    )
    assert store.verify()["objects"] == 3
    assert store.rows("forecast") == []
    from fastapi.testclient import TestClient

    from chronogrid.operations.api import create_app

    response = TestClient(create_app(tmp_path / "weather.sqlite")).get("/api/weather/runs")
    assert response.status_code == 200
    record = response.json()["runs"][0]
    assert record["sha256"] == h and record["valid_hours"] == 72 and record["model"] == "icon_eu"
    assert "not echoed" in record["identity_evidence"]
    # A later retrieval never establishes availability at an earlier cutoff.
    with pytest.raises(ValueError, match="unavailable at issuance"):
        delivery_weather(store, h, day, clock[0] - timedelta(hours=2))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda b: b.update(latitude=0), "grid point"),
        (lambda b: b["hourly_units"].update(wind_speed_10m="km/h"), "unit"),
        (lambda b: b["hourly"]["time"].__setitem__(3, b["hourly"]["time"][2]), "gap|duplicate"),
        (lambda b: b["hourly"]["relative_humidity_2m"].__setitem__(25, 101), "physical"),
        (lambda b: b["hourly"]["temperature_2m"].__setitem__(25, True), "numbers"),
        (lambda b: b["hourly"]["time"].__setitem__(0, "2027-01-14T00:00:00.001+00:00"), "grid|gap"),
    ],
)
def test_reject_bad_source_and_preserve_raw(tmp_path, mutation, match):
    req, body = fixture(date(2027, 1, 15))
    mutation(body)
    store = Journal(tmp_path / "weather.sqlite", lambda: req.initialization + timedelta(hours=8))
    with pytest.raises(ValueError, match=match):
        capture_single_run(store, req, fetch=lambda _: encode(body))
    assert len(store.rows("snapshot")) == 2  # rejected evidence is retained, no normalized success


def test_missing_values_preserved_then_unusable_day_rejected(tmp_path):
    req, body = fixture(date(2027, 1, 15))
    body["hourly"]["direct_radiation"][25] = None
    store = Journal(tmp_path / "weather.sqlite", lambda: req.initialization + timedelta(hours=8))
    h = capture_single_run(store, req, fetch=lambda _: encode(body))
    assert json.loads(store.snapshot_bytes(h))["rows"][25]["direct_radiation"] is None
    with pytest.raises(ValueError, match="missing delivery weather"):
        delivery_weather(store, h, date(2027, 1, 15), store.clock())


def test_future_init_nonmidnight_and_late_gate(tmp_path):
    req, body = fixture(date(2027, 1, 15))
    clock = [req.initialization - timedelta(hours=1)]
    store = Journal(tmp_path / "weather.sqlite", lambda: clock[0])
    with pytest.raises(ValueError, match="future"):
        capture_single_run(store, req, fetch=lambda _: encode(body))
    assert not store.rows("snapshot")
    with pytest.raises(ValueError, match="00 UTC"):
        RunRequest(initialization=req.initialization + timedelta(hours=6))
    with pytest.raises(ValueError, match="timezone-aware"):
        RunRequest(initialization=datetime(2027, 1, 14))
    clock[0] = req.initialization + timedelta(hours=8)
    h = capture_single_run(store, req, fetch=lambda _: encode(body))
    clock[0] += timedelta(hours=4)
    with pytest.raises(ValueError, match="issuance window"):
        delivery_weather(store, h, date(2027, 1, 15), clock[0])


def test_normalized_tampering_and_stale_run(tmp_path):
    req, body = fixture(date(2027, 1, 15))
    clock = [req.initialization + timedelta(hours=8)]
    store = Journal(tmp_path / "weather.sqlite", lambda: clock[0])
    h = capture_single_run(store, req, fetch=lambda _: encode(body))
    content = json.loads(store.snapshot_bytes(h))
    content["rows"][24]["temperature_2m"] = 60
    bad = store.snapshot(encode(content), SOURCE, clock[0])
    with pytest.raises(ValueError, match="differs from raw"):
        delivery_weather(store, bad, date(2027, 1, 15), clock[0])
    clock[0] = req.initialization + timedelta(hours=56)
    with pytest.raises(ValueError, match="stale"):
        delivery_weather(store, h, date(2027, 1, 17), clock[0])


def test_network_failure_is_evidence_not_empty_success(tmp_path):
    req, _ = fixture(date(2027, 1, 15))
    store = Journal(tmp_path / "weather.sqlite", lambda: req.initialization + timedelta(hours=8))

    def unavailable(_):
        raise TimeoutError("bounded source timeout")

    with pytest.raises(TimeoutError):
        capture_single_run(store, req, fetch=unavailable)
    assert len(store.rows("snapshot")) == 1
    assert store.rows("weather_error")[0]["payload"]["reason"] == "bounded source timeout"
    assert store.verify()["objects"] == 2
