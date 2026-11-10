# Repository scripts bootstrap the checkout before importing the installed package.
# ruff: noqa: E402
"""Read-only replay comparison; preserve mismatches rather than altering frozen features."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from chronogrid.features.engineer import (
    FeatureEngineer,
)
from chronogrid.serving.feature_parity import (
    require_legacy_prospective_parity,
    unavailable_features,
)


def audit(day="2024-06-17"):
    source = ROOT / "data/raw/genuine_smard_dwd_observations.parquet"
    raw = pd.read_parquet(source)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    origin = pd.Timestamp(day, tz="Europe/Berlin") - pd.DateOffset(days=1) + pd.Timedelta(hours=12)
    rows = []
    inputs = {source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()}
    for name in (
        "chronogrid/features/engineer.py",
        "chronogrid/features/autoregressive.py",
        "chronogrid/features/temporal.py",
        "chronogrid/features/weather_features.py",
        "chronogrid/serving/feature_parity.py",
        "scripts/audit_legacy_feature_parity.py",
    ):
        inputs[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for head, filename, mode in [
        ("primary", "champion_metadata.json", "persistence"),
        ("fallback", "load_only_metadata.json", "load_only"),
    ]:
        path = ROOT / "artifacts/models" / filename
        metadata = json.loads(path.read_bytes())
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode=mode)
        fe.feature_cols, fe.fitted = metadata["feature_names"], True
        reference = fe.transform(raw).set_index("timestamp")
        served, market = fe.build_day_ahead_market_frame(
            raw, forecast_origin=origin.tz_convert("UTC")
        )
        for _, row in served.iterrows():
            ts = row.timestamp
            unavailable = set(
                unavailable_features(fe.feature_cols, ts.to_pydatetime(), origin.to_pydatetime())
            )
            for feature in fe.feature_cols:
                expected, actual = float(reference.loc[ts, feature]), float(row[feature])
                matches = bool(np.isclose(expected, actual, rtol=1e-10, atol=1e-8, equal_nan=False))
                rows.append(
                    dict(
                        head=head,
                        issued_at=origin.isoformat(),
                        delivery_timestamp=ts.isoformat(),
                        feature=feature,
                        dependency_available_at_gate=feature not in unavailable,
                        rowwise_replay_value=expected,
                        market_frame_value=actual,
                        numerical_match=matches,
                    )
                )
    unavailable_hours = {
        (r["head"], r["delivery_timestamp"]) for r in rows if not r["dependency_available_at_gate"]
    }
    summary = dict(
        decision="NO_GO",
        original_training_feature_provenance_verified=False,
        origin_count=1,
        delivery_date=day,
        unavailable_delivery_hours=len({ts for _, ts in unavailable_hours}),
        unavailable_head_delivery_hours=len(unavailable_hours),
        mismatched_cells=sum(not r["numerical_match"] for r in rows),
        compared_cells=len(rows),
        inputs_sha256=inputs,
        reference_kind="Historical rowwise feature replay; not archived original training features",
        runtime=dict(numpy=np.__version__, pandas=pd.__version__),
        limitations=[
            "One historical origin; not forecast performance or original training provenance",
            "Reference uses historical observations including post-gate values for diagnosis only",
            "No models, calibration, labels or frozen feature code changed",
        ],
    )
    summary["per_head"] = {
        head: dict(
            feature_count=len({r["feature"] for r in rows if r["head"] == head}),
            compared_cells=sum(r["head"] == head for r in rows),
            unavailable_feature_cells=sum(
                r["head"] == head and not r["dependency_available_at_gate"] for r in rows
            ),
            delivery_hours=len({r["delivery_timestamp"] for r in rows if r["head"] == head}),
            unavailable_delivery_hours=len(
                {
                    r["delivery_timestamp"]
                    for r in rows
                    if r["head"] == head and not r["dependency_available_at_gate"]
                }
            ),
            mismatched_cells=sum(not r["numerical_match"] for r in rows if r["head"] == head),
            available_dependency_mismatches=sum(
                not r["numerical_match"] and r["dependency_available_at_gate"]
                for r in rows
                if r["head"] == head
            ),
        )
        for head in ("primary", "fallback")
    }
    return summary, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--day", default="2024-06-17")
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="release gate; expected to reject the frozen legacy path",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary, rows = audit(args.day)
    with (args.output / "per_feature_horizon.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["per_feature_horizon_sha256"] = hashlib.sha256(
        (args.output / "per_feature_horizon.csv").read_bytes()
    ).hexdigest()
    (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["per_head"], indent=2))
    if args.require_pass:
        require_legacy_prospective_parity(summary)
