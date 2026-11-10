"""Save the completed-hour gate audit for the benchmark evaluation window."""

import json
from pathlib import Path

import pandas as pd

from chronogrid.features.engineer import FeatureEngineer
from chronogrid.features.gate_audit import audit_gate_availability


def main():
    raw = pd.read_parquet("data/raw/genuine_smard_dwd_observations.parquet")
    fe = FeatureEngineer()
    fe.fit(raw)
    result = audit_gate_availability(raw.tail(840).timestamp, fe.get_feature_names())
    report = {
        "hours": int(result.delivery_utc.nunique()),
        "feature_cells": len(result),
        "post_gate_cells": int((~result.available).sum()),
        "scope": "Complete observation intervals; publication vintages unverified",
        "leakage": fe.run_leakage_suite(raw),
        "integrity": fe.run_data_integrity_suite(raw),
    }
    path = Path("artifacts/reports/gate_audit.json")
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
