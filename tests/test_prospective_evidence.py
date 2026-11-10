import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

import chronogrid.evaluation.prospective as evidence
from chronogrid.evaluation.prospective import ActualObservation, EvidenceStore, PredictionRecord

D = "a" * 64


def record(day="2027-01-15"):
    zone = ZoneInfo("Europe/Berlin")
    start = datetime.fromisoformat(day).replace(tzinfo=zone)
    end = (start + timedelta(days=1)).astimezone(timezone.utc)
    start = start.astimezone(timezone.utc)
    issued = (start - timedelta(days=1)).replace(hour=10)
    return dict(
        record_id="origin_001",
        protocol_sha256=D,
        issued_at=issued.isoformat(),
        issuance_timezone="UTC",
        delivery_date=day,
        model_id="frozen-primary",
        model_sha256=D,
        feature_code_sha256=D,
        route="primary",
        routing_reason="healthy",
        data_quality_flags=[],
        outage=False,
        inputs=[
            dict(
                source="snapshot",
                snapshot_sha256=D,
                event_time_start=(issued - timedelta(days=1)).isoformat(),
                event_time_end=(issued - timedelta(hours=1)).isoformat(),
                available_at=(issued - timedelta(minutes=2)).isoformat(),
                captured_at=(issued - timedelta(minutes=1)).isoformat(),
                quality_flags=[],
                outage=False,
            )
        ],
        calibration_method="pooled_marginal_mad",
        calibration_state_version="frozen-v1",
        calibration_state_sha256=D,
        nominal_coverage=0.9,
        effective_alpha=0.1,
        deliveries=[
            dict(
                timestamp=(start + timedelta(hours=i)).isoformat(),
                point_mw=100,
                lower_mw=90,
                upper_mw=110,
                interval_status="bounded",
                quality_flags=[],
            )
            for i in range(int((end - start).total_seconds() / 3600))
        ],
    )


def set_clock(monkeypatch, t):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return t.astimezone(tz or timezone.utc)

    monkeypatch.setattr(evidence, "datetime", Clock)


@pytest.mark.parametrize("day,n", [("2027-01-15", 24), ("2027-03-28", 23), ("2027-10-31", 25)])
def test_complete_local_delivery_day_including_dst(day, n):
    r = PredictionRecord.model_validate(record(day))
    assert len(r.deliveries) == n


@pytest.mark.parametrize(
    "change",
    [
        "naive",
        "timezone",
        "duplicate",
        "order",
        "future_input",
        "after_gate",
        "actual_in_prediction",
        "bad_bounds",
        "nan",
    ],
)
def test_reject_invalid_record(change):
    r = record()
    if change == "naive":
        r["issued_at"] = "2027-01-14T10:00:00"
    elif change == "timezone":
        r["issuance_timezone"] = "Europe/Berlin"
    elif change == "duplicate":
        r["deliveries"][1] = r["deliveries"][0]
    elif change == "order":
        r["deliveries"].reverse()
    elif change == "future_input":
        r["inputs"][0]["captured_at"] = "2027-01-14T11:00:00Z"
    elif change == "after_gate":
        r["issued_at"] = "2027-01-14T12:00:00Z"
    elif change == "actual_in_prediction":
        r["actual_mw"] = 100
    elif change == "bad_bounds":
        r["deliveries"][0]["lower_mw"] = 120
    else:
        r["deliveries"][0]["point_mw"] = float("nan")
    with pytest.raises((ValidationError, ValueError)):
        PredictionRecord.model_validate(r)


def test_commit_once_nested_immutability_and_tamper(tmp_path, monkeypatch):
    r = PredictionRecord.model_validate(record())
    set_clock(monkeypatch, r.issued_at + timedelta(minutes=1))
    store = EvidenceStore(tmp_path)
    key = store.commit_prediction(r)
    with pytest.raises(FileExistsError):
        store.commit_prediction(r)
    with pytest.raises(ValidationError):
        r.deliveries[0].point_mw = 0
    with pytest.raises(ValueError):
        store.commit_prediction(r.model_copy(update={"issued_at": datetime(2027, 1, 14)}))
    path = tmp_path / "prediction_origin_001.json"
    data = json.loads(path.read_text())
    data["prediction"]["deliveries"][0]["point_mw"] = 101
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        store.read_prediction("origin_001")
    assert len(key) == 64


def test_backfill_and_future_issuance_rejected(tmp_path, monkeypatch):
    r = PredictionRecord.model_validate(record())
    store = EvidenceStore(tmp_path)
    for now in (r.issued_at - timedelta(seconds=1), r.deliveries[0].timestamp):
        set_clock(monkeypatch, now)
        with pytest.raises(ValueError):
            store.commit_prediction(r)
    assert not list(tmp_path.iterdir())


def test_actual_linkage_and_revision_preservation(tmp_path, monkeypatch):
    r = PredictionRecord.model_validate(record())
    store = EvidenceStore(tmp_path)
    set_clock(monkeypatch, r.issued_at + timedelta(minutes=1))
    key = store.commit_prediction(r)
    t = r.deliveries[0].timestamp
    payload = dict(
        prediction_sha256=key,
        delivery_timestamp=t,
        observation_end=t + timedelta(hours=1),
        available_at=t + timedelta(hours=2),
        source="meter",
        source_snapshot_sha256=D,
        revision="v1",
        value_mw=101,
        status="observed",
        quality_flags=[],
    )
    a = ActualObservation.model_validate(payload)
    with pytest.raises(ValueError):
        store.link_actual("origin_001", a)  # not yet available
    set_clock(monkeypatch, t + timedelta(hours=3))
    store.link_actual("origin_001", a)
    with pytest.raises(FileExistsError):
        store.link_actual("origin_001", a)
    for updates in (
        {"prediction_sha256": "b" * 64},
        {
            "delivery_timestamp": t + timedelta(days=2),
            "observation_end": t + timedelta(days=2, hours=1),
            "available_at": t + timedelta(days=2, hours=2),
        },
    ):
        with pytest.raises(ValueError):
            store.link_actual("origin_001", ActualObservation.model_validate(payload | updates))
    for updates in (
        {"available_at": t},
        {"value_mw": None},
        {"delivery_timestamp": t.replace(tzinfo=None)},
    ):
        with pytest.raises(ValueError):
            ActualObservation.model_validate(payload | updates)
    store.link_actual(
        "origin_001",
        ActualObservation.model_validate(payload | {"revision": "v2", "value_mw": 102}),
    )
    store.link_actual(
        "origin_001",
        ActualObservation.model_validate(
            payload | {"revision": "v3", "value_mw": None, "status": "missing"}
        ),
    )
    assert len(list(tmp_path.glob("actual_*"))) == 3
    assert store.read_prediction("origin_001")[1] == key


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path).read_prediction("../escape")
