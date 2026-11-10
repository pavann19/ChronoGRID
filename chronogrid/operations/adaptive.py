"""Append-only delayed daily-batch adaptation; no unconditional coverage guarantee."""

import json
import math
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from chronogrid.evaluation.prospective import Delivery, Digest, Frozen, Identifier, canonical, utc
from chronogrid.operations.capture import delivery_grid
from chronogrid.operations.contracts import Forecast, Run
from chronogrid.operations.journal import Conflict, IntegrityFailure, Journal, digest, encode
from chronogrid.uncertainty.conformal import ConformalCalibrationPool
from chronogrid.uncertainty.schemas import CalibrationTable


class StaleAdaptiveState(Conflict):
    pass


class AdaptiveConfig(Frozen):
    run_id: Identifier
    arm: Identifier
    head: Literal["primary", "fallback"]
    model_sha256: Digest
    base_calibration_sha256: Digest
    feature_policy: str = Field(min_length=1)
    target_kind: Literal["marginal_hourly", "simultaneous_daily"]
    target_alpha: float = Field(default=0.1, gt=0, lt=1)
    gamma: float = Field(default=0.01, gt=0, le=1)
    alpha_min: float = Field(default=0.01, gt=0, lt=1)
    alpha_max: float = Field(default=0.5, gt=0, lt=1)
    minimum_delay_hours: int = Field(default=48, ge=1, le=720)
    label_deadline_hours: int = Field(default=168, ge=1, le=2160)
    excluded_actual_quality_flags: tuple[str, ...] = ("invalid", "missing", "unverified_source")

    @model_validator(mode="after")
    def limits(self):
        if (
            not self.alpha_min <= self.target_alpha <= self.alpha_max
            or self.minimum_delay_hours > self.label_deadline_hours
        ):
            raise ValueError("invalid alpha/delay policy")
        return self

    @property
    def stream_key(self):
        return self.run_id + ":" + self.arm

    @property
    def method(self):
        return "pooled_marginal_mad" if self.target_kind == "marginal_hourly" else "daily_sup_norm"


def _latest(store, db, cfg, config_row):
    row = db.execute(
        "SELECT o.* FROM objects o JOIN events e ON e.object_key=o.key WHERE o.kind='aci_step' AND json_extract(o.payload,'$.stream_key')=? ORDER BY e.seq DESC LIMIT 1",
        (cfg.stream_key,),
    ).fetchone()
    if row is None:
        return dict(alpha=cfg.target_alpha, basis_sha256=config_row["sha256"], last_day=None)
    if digest(row["payload"].encode()) != row["sha256"]:
        raise IntegrityFailure("adaptive state hash mismatch")
    body = json.loads(row["payload"])
    return dict(
        alpha=body["alpha_after"],
        basis_sha256=row["sha256"],
        last_day=date.fromisoformat(body["delivery_date"]),
    )


def validate_forecast_basis(store, db, p, wrapper):
    """Called inside Journal.commit's write transaction, including static-bypass checks."""
    key = "aci-config:" + p.run_id + ":" + p.arm
    row = db.execute("SELECT key FROM objects WHERE key=?", (key,)).fetchone()
    if row is None:
        if "aci_stream" in wrapper:
            raise ValueError("unregistered adaptive stream")
        return
    content, meta = store._get(db, key)
    cfg = AdaptiveConfig.model_validate(content)
    if wrapper.get("aci_stream") != cfg.stream_key:
        raise ValueError("adaptive arm requires its versioned state snapshot")
    state = _latest(store, db, cfg, meta)
    if utc(datetime.fromisoformat(meta["recorded_at"])) > utc(p.issued_at):
        raise ValueError("adaptive configuration registered after issuance")
    if wrapper.get("basis_sha256") != state["basis_sha256"]:
        raise StaleAdaptiveState(
            "adaptive state changed during computation; rebuild before commitment"
        )
    if (
        wrapper.get("config_sha256") != meta["sha256"]
        or wrapper.get("base_calibration_sha256") != cfg.base_calibration_sha256
    ):
        raise ValueError("adaptive configuration/base mismatch")
    if (
        p.model_sha256 != cfg.model_sha256
        or p.route != cfg.head
        or p.feature_policy != cfg.feature_policy
        or p.calibration_method != cfg.method
    ):
        raise ValueError("adaptive head/model/feature/target mismatch")
    if not math.isclose(p.effective_alpha, state["alpha"], abs_tol=1e-12) or not math.isclose(
        p.nominal_coverage, 1 - cfg.target_alpha, abs_tol=1e-12
    ):
        raise ValueError("forecast alpha does not match registered stream state")
    if (
        p.calibration_state_version != wrapper.get("state_version")
        or state["last_day"] is not None
        and p.delivery_date <= state["last_day"]
    ):
        raise ValueError("invalid adaptive version or already processed origin")


class AdaptiveStreams:
    def __init__(self, journal: Journal):
        self.store = journal

    def register(self, config: AdaptiveConfig):
        cfg = AdaptiveConfig.model_validate_json(canonical(config))
        table = CalibrationTable.model_validate_json(
            self.store.snapshot_bytes(cfg.base_calibration_sha256)
        )
        if table.model_sha256 != cfg.model_sha256:
            raise ValueError("base calibration belongs to another model")
        with self.store.transaction() as db:
            body, run_meta = self.store._get(db, "run:" + cfg.run_id)
            run = Run.model_validate(body)
            if (
                not run.arm_models
                or next((a.model_sha256 for a in run.arm_models if a.arm == cfg.arm), None)
                != cfg.model_sha256
            ):
                raise ValueError("adaptive stream requires pinned run arm")
            existing = db.execute(
                "SELECT 1 FROM objects WHERE kind='forecast' AND json_extract(payload,'$.run_id')=? AND json_extract(payload,'$.arm')=?",
                (cfg.run_id, cfg.arm),
            ).fetchone()
            if existing:
                raise ValueError("cannot retrofit adaptation after forecast issuance")
            return self.store._add(
                db, "aci-config:" + cfg.stream_key, "aci_config", json.loads(canonical(cfg))
            )

    def state(self, run_id, arm):
        with self.store.connection() as db:
            db.execute("BEGIN")
            body, meta = self.store._get(db, "aci-config:" + run_id + ":" + arm)
            cfg = AdaptiveConfig.model_validate(body)
            state = _latest(self.store, db, cfg, meta)
            db.execute("COMMIT")
            return dict(
                stream_key=cfg.stream_key,
                head=cfg.head,
                config_sha256=meta["sha256"],
                **(state | {"last_day": str(state["last_day"]) if state["last_day"] else None}),
                guarantee=False,
            )

    def prepare(self, run_id, arm):
        self.store.verify()
        with self.store.connection() as db:
            db.execute("BEGIN")
            body, meta = self.store._get(db, "aci-config:" + run_id + ":" + arm)
            cfg = AdaptiveConfig.model_validate(body)
            state = _latest(self.store, db, cfg, meta)
            db.execute("COMMIT")
        version = "aci-v1:" + state["basis_sha256"]
        wrapper = dict(
            model_sha256=cfg.model_sha256,
            base_calibration_sha256=cfg.base_calibration_sha256,
            feature_policy=cfg.feature_policy,
            aci_stream=cfg.stream_key,
            config_sha256=meta["sha256"],
            basis_sha256=state["basis_sha256"],
            alpha=state["alpha"],
            state_version=version,
        )
        h = self.store.snapshot(
            encode(wrapper), "aci-state:" + cfg.stream_key, utc(self.store.clock())
        )
        return cfg, state, h, version

    def settle(self, run_id, arm, delivery_date: date | None = None):
        store = self.store
        store.verify()
        now = utc(store.clock())
        with store.transaction() as db:
            body, config_meta = store._get(db, "aci-config:" + run_id + ":" + arm)
            cfg = AdaptiveConfig.model_validate(body)
            run = Run.model_validate(store._get(db, "run:" + run_id)[0])
            state = _latest(store, db, cfg, config_meta)
            next_day = (
                state["last_day"] + timedelta(days=1) if state["last_day"] else run.start_date
            )
            if delivery_date is not None:
                prior = db.execute(
                    "SELECT key FROM objects WHERE key=?",
                    (f"aci-step:{cfg.stream_key}:{delivery_date}",),
                ).fetchone()
                if prior:
                    return store._get(db, prior["key"])[0] | {"duplicate": True}
                if delivery_date != next_day:
                    raise ValueError("feedback must follow chronological scheduled origins")
            if next_day > run.end_date:
                return dict(status="done", stream_key=cfg.stream_key)
            grid = delivery_grid(next_day)
            end = grid[-1] + timedelta(hours=1)
            eligible = end + timedelta(hours=cfg.minimum_delay_hours)
            deadline = end + timedelta(hours=cfg.label_deadline_hours)
            if now < eligible:
                return dict(
                    status="not_eligible",
                    stream_key=cfg.stream_key,
                    delivery_date=str(next_day),
                    eligible_after=eligible.isoformat(),
                )
            forecast_key = f"forecast:{run_id}:{next_day}:{arm}"
            row = db.execute("SELECT key FROM objects WHERE key=?", (forecast_key,)).fetchone()
            p, forecast_meta = store._get(db, forecast_key) if row else (None, None)
            actual_rows = []
            covered = []
            reason = None
            status = "processed"
            if p is None:
                reason = "no_forecast"
                status = "skipped"
            else:
                p = Forecast.model_validate(p)
                if (
                    p.route != cfg.head
                    or p.model_sha256 != cfg.model_sha256
                    or p.calibration_method not in (cfg.method, "unavailable")
                ):
                    raise IntegrityFailure("issued forecast incompatible with adaptive stream")
                if any(x.interval_status == "unavailable" for x in p.deliveries):
                    reason = "interval_unavailable"
                    status = "skipped"
                else:
                    assert forecast_meta is not None
                    candidates = db.execute(
                        "SELECT o.* FROM objects o JOIN events e ON e.object_key=o.key WHERE o.kind='actual' AND json_extract(o.payload,'$.prediction_sha256')=? ORDER BY e.seq",
                        (forecast_meta["sha256"],),
                    ).fetchall()
                    first = {}
                    for candidate in candidates:
                        a = json.loads(candidate["payload"])
                        t = utc(datetime.fromisoformat(a["delivery_timestamp"]))
                        recorded = utc(datetime.fromisoformat(candidate["recorded_at"]))
                        available = utc(datetime.fromisoformat(a["available_at"]))
                        if (
                            a["status"] == "observed"
                            and not set(a["quality_flags"]).intersection(
                                cfg.excluded_actual_quality_flags
                            )
                            and recorded <= now
                            and available <= now
                            and recorded <= deadline
                            and available <= deadline
                            and t not in first
                        ):
                            first[t] = (a, candidate)
                    if not all(t in first for t in grid):
                        if now < deadline:
                            return dict(
                                status="pending_actuals",
                                stream_key=cfg.stream_key,
                                delivery_date=str(next_day),
                                observed_hours=sum(t in first for t in grid),
                                required_hours=len(grid),
                            )
                        status = "skipped"
                        reason = "labels_missing_at_deadline"
                    else:
                        for d in p.deliveries:
                            a, meta = first[utc(d.timestamp)]
                            actual_rows.append(meta["sha256"])
                            covered.append(
                                d.interval_status == "unbounded"
                                or d.lower_mw <= a["value_mw"] <= d.upper_mw
                            )
            if status == "processed":
                misses = sum(not hit for hit in covered)
                error = (
                    misses / len(grid)
                    if cfg.target_kind == "marginal_hourly"
                    else float(misses > 0)
                )
                alpha = max(
                    cfg.alpha_min,
                    min(cfg.alpha_max, state["alpha"] + cfg.gamma * (cfg.target_alpha - error)),
                )
            else:
                misses = None
                error = None
                alpha = state["alpha"]
            result = dict(
                stream_key=cfg.stream_key,
                config_sha256=config_meta["sha256"],
                basis_before=state["basis_sha256"],
                run_id=run_id,
                arm=arm,
                head=cfg.head,
                delivery_date=str(next_day),
                status=status,
                reason=reason,
                applied_at=now.isoformat(),
                eligible_after=eligible.isoformat(),
                label_deadline=deadline.isoformat(),
                target_kind=cfg.target_kind,
                required_hours=len(grid),
                observed_hours=len(covered),
                covered_hours=sum(covered) if covered else None,
                missed_hours=misses,
                error_signal=error,
                alpha_before=state["alpha"],
                alpha_after=alpha,
                gamma=cfg.gamma,
                forecast_sha256=forecast_meta["sha256"] if forecast_meta else None,
                actual_record_sha256=actual_rows,
                guarantee=False,
            )
            store._add(
                db,
                f"aci-step:{cfg.stream_key}:{next_day}",
                "aci_step",
                result,
                recorded_at=now.isoformat(),
            )
            return result


def capture_adaptive48(store, run_id, day, arm, snapshot_sha256, experiment_dir):
    from chronogrid.operations.capture import build_fallback48

    cfg, state, h, version = AdaptiveStreams(store).prepare(run_id, arm)
    if cfg.head != "fallback":
        raise ValueError("current live builder supports the load-only fallback head")
    p = build_fallback48(store, run_id, day, arm, snapshot_sha256, experiment_dir)
    if p.model_sha256 != cfg.model_sha256 or p.feature_policy != cfg.feature_policy:
        raise ValueError("adaptive model/policy mismatch")
    table = CalibrationTable.model_validate_json(store.snapshot_bytes(cfg.base_calibration_sha256))
    pool = ConformalCalibrationPool.from_calibration_table(table)
    q = (
        pool.get_marginal_quantile(state["alpha"])
        if cfg.target_kind == "marginal_hourly"
        else pool.get_simultaneous_quantile(state["alpha"])
    )[0]
    import numpy as np

    pred = np.array([d.point_mw for d in p.deliveries])
    hours = np.array([d.timestamp.hour for d in p.deliveries])
    lo, hi = pool.normalizer.scale_intervals(pred, hours, q) if math.isfinite(q) else (None, None)
    deliveries = tuple(
        Delivery(
            timestamp=d.timestamp,
            point_mw=d.point_mw,
            lower_mw=float(lo[i]) if lo is not None else None,
            upper_mw=float(hi[i]) if hi is not None else None,
            interval_status="bounded" if lo is not None else "unbounded",
            quality_flags=d.quality_flags + ("delayed_adaptive_research",),
        )
        for i, d in enumerate(p.deliveries)
    )
    p = Forecast.model_validate(
        p.model_dump()
        | dict(
            issued_at=utc(store.clock()),
            deliveries=deliveries,
            calibration_method=cfg.method,
            calibration_state_sha256=h,
            calibration_state_version=version,
            nominal_coverage=1 - cfg.target_alpha,
            effective_alpha=state["alpha"],
        )
    )
    return p, store.commit(p)


def verify_adaptive_ledger(store, db, objects):
    """Recompute transition evidence from earlier journal events, not trusted summary flags."""
    configs = {}
    states = {}
    earlier = {}
    actuals = {}
    for event in db.execute(
        "SELECT e.seq,o.* FROM events e JOIN objects o ON o.key=e.object_key ORDER BY e.seq"
    ):
        body = json.loads(event["payload"])
        kind = event["kind"]
        key = event["key"]
        if kind == "aci_config":
            cfg = AdaptiveConfig.model_validate(body)
            run = Run.model_validate(earlier["run:" + cfg.run_id][0])
            if (
                not run.arm_models
                or next((a.model_sha256 for a in run.arm_models if a.arm == cfg.arm), None)
                != cfg.model_sha256
            ):
                raise IntegrityFailure("ACI/run model mismatch")
            base = earlier.get("snapshot:" + cfg.base_calibration_sha256)
            if base is None:
                raise IntegrityFailure("ACI missing base calibration")
            import base64

            table = CalibrationTable.model_validate_json(
                base64.b64decode(base[0]["content_base64"])
            )
            if table.model_sha256 != cfg.model_sha256:
                raise IntegrityFailure("ACI base/model mismatch")
            if any(
                obj[0].get("run_id") == cfg.run_id
                and obj[0].get("arm") == cfg.arm
                and obj[1]["kind"] == "forecast"
                for obj in earlier.values()
            ):
                raise IntegrityFailure("retrofit ACI configuration")
            configs[cfg.stream_key] = (cfg, event, run)
            states[cfg.stream_key] = dict(
                alpha=cfg.target_alpha, basis=event["sha256"], next_day=run.start_date
            )
        elif kind == "actual":
            actuals.setdefault(body["prediction_sha256"], []).append((body, event))
        elif kind == "aci_step":
            stream = body["stream_key"]
            if stream not in configs:
                raise IntegrityFailure("unregistered ACI transition")
            cfg, meta, run = configs[stream]
            state = states[stream]
            day = date.fromisoformat(body["delivery_date"])
            if (
                day != state["next_day"]
                or day > run.end_date
                or body["basis_before"] != state["basis"]
                or body["config_sha256"] != meta["sha256"]
            ):
                raise IntegrityFailure("nonchronological ACI transition/basis")
            applied = utc(datetime.fromisoformat(body["applied_at"]))
            if applied != utc(datetime.fromisoformat(event["recorded_at"])):
                raise IntegrityFailure("ACI application receipt mismatch")
            grid = delivery_grid(day)
            end = grid[-1] + timedelta(hours=1)
            deadline = end + timedelta(hours=cfg.label_deadline_hours)
            if applied < end + timedelta(hours=cfg.minimum_delay_hours):
                raise IntegrityFailure("premature ACI feedback")
            f = earlier.get(f"forecast:{cfg.run_id}:{day}:{cfg.arm}")
            error = None
            hashes = []
            misses = None
            observed = 0
            reason = None
            if f is None:
                reason = "no_forecast"
            else:
                p = Forecast.model_validate(f[0])
                forecast_hash = f[1]["sha256"]
                if (
                    p.route != cfg.head
                    or p.model_sha256 != cfg.model_sha256
                    or p.calibration_method not in (cfg.method, "unavailable")
                ):
                    raise IntegrityFailure("ACI forecast/head mismatch")
                if any(d.interval_status == "unavailable" for d in p.deliveries):
                    reason = "interval_unavailable"
                else:
                    selected = {}
                    for a, row in actuals.get(forecast_hash, []):
                        t = utc(datetime.fromisoformat(a["delivery_timestamp"]))
                        recorded = utc(datetime.fromisoformat(row["recorded_at"]))
                        available = utc(datetime.fromisoformat(a["available_at"]))
                        if (
                            a["status"] == "observed"
                            and not set(a["quality_flags"]).intersection(
                                cfg.excluded_actual_quality_flags
                            )
                            and recorded <= min(applied, deadline)
                            and available <= min(applied, deadline)
                            and t not in selected
                        ):
                            selected[t] = (a, row)
                    if not all(t in selected for t in grid):
                        if applied < deadline:
                            raise IntegrityFailure("premature missing-label skip")
                        reason = "labels_missing_at_deadline"
                    else:
                        coverage = []
                        for d in p.deliveries:
                            a, row = selected[utc(d.timestamp)]
                            hashes.append(row["sha256"])
                            coverage.append(
                                d.interval_status == "unbounded"
                                or d.lower_mw <= a["value_mw"] <= d.upper_mw
                            )
                        observed = len(grid)
                        misses = sum(not hit for hit in coverage)
                        error = (
                            misses / len(grid)
                            if cfg.target_kind == "marginal_hourly"
                            else float(misses > 0)
                        )
            expected = (
                state["alpha"]
                if error is None
                else max(
                    cfg.alpha_min,
                    min(cfg.alpha_max, state["alpha"] + cfg.gamma * (cfg.target_alpha - error)),
                )
            )
            if (
                body["head"] != cfg.head
                or body["target_kind"] != cfg.target_kind
                or body["gamma"] != cfg.gamma
                or body["required_hours"] != len(grid)
                or body["label_deadline"] != deadline.isoformat()
                or body["eligible_after"]
                != (end + timedelta(hours=cfg.minimum_delay_hours)).isoformat()
                or body["forecast_sha256"] != (f[1]["sha256"] if f else None)
                or body["covered_hours"] != (observed - misses if observed else None)
            ):
                raise IntegrityFailure("ACI transition summary mismatch")
            if (
                not math.isclose(body["alpha_before"], state["alpha"], abs_tol=1e-12)
                or not math.isclose(body["alpha_after"], expected, abs_tol=1e-12)
                or body["error_signal"] != error
                or body["actual_record_sha256"] != hashes
                or body["reason"] != reason
                or body["status"] != ("processed" if error is not None else "skipped")
                or body["observed_hours"] != observed
                or body["missed_hours"] != misses
            ):
                raise IntegrityFailure("ACI transition differs from issued/observed evidence")
            state.update(alpha=expected, basis=event["sha256"], next_day=day + timedelta(days=1))
        earlier[key] = (body, event)
