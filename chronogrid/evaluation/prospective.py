"""Local prospective evidence contract; not a production capture integration."""

import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


def utc(t: datetime) -> datetime:
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return t.astimezone(timezone.utc)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class InputSnapshot(Frozen):
    source: str = Field(min_length=1)
    snapshot_sha256: Digest
    event_time_start: datetime
    event_time_end: datetime
    available_at: datetime
    captured_at: datetime
    quality_flags: tuple[str, ...]
    outage: bool

    @model_validator(mode="after")
    def times(self):
        start, end, available, captured = map(
            utc, (self.event_time_start, self.event_time_end, self.available_at, self.captured_at)
        )
        if start > end or available > captured:
            raise ValueError("invalid input event range or capture before availability")
        return self


class Delivery(Frozen):
    timestamp: datetime
    point_mw: FiniteFloat
    lower_mw: FiniteFloat | None
    upper_mw: FiniteFloat | None
    interval_status: Literal["bounded", "unbounded", "unavailable"]
    quality_flags: tuple[str, ...]

    @model_validator(mode="after")
    def bounds(self):
        utc(self.timestamp)
        if self.interval_status == "bounded":
            if self.lower_mw is None or self.upper_mw is None or self.lower_mw > self.upper_mw:
                raise ValueError("bounded interval requires ordered finite endpoints")
        elif self.lower_mw is not None or self.upper_mw is not None:
            raise ValueError("unbounded/unavailable uses null endpoints and explicit status")
        return self


class PredictionRecord(Frozen):
    schema_version: Literal["1.0"] = "1.0"
    experiment_kind: Literal["prospective"] = "prospective"
    protocol_sha256: Digest
    record_id: Identifier
    issued_at: datetime
    issuance_timezone: str
    delivery_date: date
    delivery_timezone: Literal["Europe/Berlin"] = "Europe/Berlin"
    model_id: str = Field(min_length=1)
    model_sha256: Digest
    feature_code_sha256: Digest
    route: Literal["primary", "fallback"]
    routing_reason: str = Field(min_length=1)
    data_quality_flags: tuple[str, ...]
    outage: bool
    inputs: tuple[InputSnapshot, ...] = Field(min_length=1)
    calibration_method: Literal["pooled_marginal_mad", "daily_sup_norm"]
    calibration_state_version: str = Field(min_length=1)
    calibration_state_sha256: Digest
    nominal_coverage: float = Field(gt=0, lt=1)
    effective_alpha: float = Field(gt=0, lt=1)
    deliveries: tuple[Delivery, ...] = Field(min_length=23, max_length=25)

    @model_validator(mode="after")
    def timeline(self):
        issued = utc(self.issued_at)
        zone = ZoneInfo(self.issuance_timezone)
        if self.issued_at.utcoffset() != issued.astimezone(zone).utcoffset():
            raise ValueError("issuance timezone disagrees with timestamp offset")
        local = ZoneInfo(self.delivery_timezone)
        start = datetime.combine(self.delivery_date, datetime.min.time(), local).astimezone(
            timezone.utc
        )
        end = datetime.combine(
            self.delivery_date + timedelta(days=1), datetime.min.time(), local
        ).astimezone(timezone.utc)
        expected = tuple(
            start + timedelta(hours=h) for h in range(int((end - start).total_seconds() / 3600))
        )
        if tuple(utc(x.timestamp) for x in self.deliveries) != expected:
            raise ValueError(
                "delivery hours must be complete ordered local day in UTC, including DST"
            )
        if issued >= start:
            raise ValueError("issuance must precede delivery")
        gate = datetime.combine(
            self.delivery_date - timedelta(days=1), datetime.min.time(), local
        ).replace(hour=12)
        if issued > utc(gate):
            raise ValueError("issuance after frozen day-ahead cutoff")
        if any(utc(x.available_at) > issued or utc(x.captured_at) > issued for x in self.inputs):
            raise ValueError("input unavailable at issuance")
        return self


class ActualObservation(Frozen):
    prediction_sha256: Digest
    delivery_timestamp: datetime
    observation_end: datetime
    available_at: datetime
    source: str = Field(min_length=1)
    source_snapshot_sha256: Digest
    revision: Identifier
    value_mw: FiniteFloat | None
    status: Literal["observed", "missing"]
    quality_flags: tuple[str, ...]

    @model_validator(mode="after")
    def timeline(self):
        if utc(self.observation_end) != utc(self.delivery_timestamp) + timedelta(hours=1):
            raise ValueError("actual must identify its hourly interval end")
        if utc(self.available_at) < utc(self.observation_end):
            raise ValueError("actual availability before observation end")
        if (self.status == "observed") != (self.value_mw is not None):
            raise ValueError("observed requires value; missing requires null")
        return self


def canonical(record: BaseModel) -> bytes:
    # Round-trip validation rejects bypasses via model_copy/model_construct.
    record = type(record).model_validate_json(record.model_dump_json())
    return json.dumps(
        record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class EvidenceStore:
    """Exclusive-create local files plus hash verification, not adversarial WORM storage."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _write(self, name: str, payload: dict):
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        # No overwrite API: an origin/actual revision can only be committed once.
        with (self.root / name).open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def commit_prediction(self, record: PredictionRecord) -> str:
        body = canonical(record)
        record = PredictionRecord.model_validate_json(body)
        now = datetime.now(timezone.utc)
        if not utc(record.issued_at) <= now < utc(record.deliveries[0].timestamp):
            raise ValueError("prospective commit must occur after issuance and before delivery")
        key = digest(body)
        self._write(
            f"prediction_{record.record_id}.json",
            dict(sha256=key, recorded_at=now.isoformat(), prediction=json.loads(body)),
        )
        return key

    def read_prediction(self, record_id: str) -> tuple[PredictionRecord, str, datetime]:
        # Validate identifiers before using them in a path.
        from pydantic import TypeAdapter

        record_id = TypeAdapter(Identifier).validate_python(record_id)
        envelope = json.loads((self.root / f"prediction_{record_id}.json").read_text())
        record = PredictionRecord.model_validate(envelope["prediction"])
        if record.record_id != record_id or digest(canonical(record)) != envelope["sha256"]:
            raise ValueError("prediction content hash/identity mismatch")
        committed = utc(datetime.fromisoformat(envelope["recorded_at"]))
        if not utc(record.issued_at) <= committed < utc(record.deliveries[0].timestamp):
            raise ValueError("invalid prediction receipt time")
        return record, envelope["sha256"], committed

    def link_actual(self, record_id: str, actual: ActualObservation) -> str:
        body = canonical(actual)
        actual = ActualObservation.model_validate_json(body)
        prediction, key, committed = self.read_prediction(record_id)
        now = datetime.now(timezone.utc)
        if actual.prediction_sha256 != key:
            raise ValueError("actual points to another prediction")
        if utc(actual.delivery_timestamp) not in {utc(x.timestamp) for x in prediction.deliveries}:
            raise ValueError("actual hour outside prediction vector")
        if not committed < utc(actual.observation_end) <= utc(actual.available_at) <= now:
            raise ValueError("actual linkage violates availability timeline")
        hour = utc(actual.delivery_timestamp).strftime("%Y%m%dT%H%M%SZ")
        self._write(
            f"actual_{record_id}_{hour}_{actual.revision}.json",
            dict(sha256=digest(body), recorded_at=now.isoformat(), actual=json.loads(body)),
        )
        return digest(body)
