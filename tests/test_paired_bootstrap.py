"""Exercise the bootstrap CLI with a small, independently sampled fixture."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_bootstrap.py"


def test_paired_bootstrap_is_deterministic_and_paired(tmp_path):
    times = pd.date_range("2024-06-17", periods=72, freq="h", tz="Europe/Berlin")
    errors = {
        "RandomForest": [1, 7, 2],
        "Weekly_168h": [3, 2, 6],
        "HistGradientBoosting": [2, 5, 4],
        "LightGBM": [4, 1, 5],
        "XGBoost": [6, 3, 1],
    }
    rows = [
        {
            "policy": "corrected",
            "model": model,
            "timestamp": timestamp.isoformat(),
            "actual_mw": 100 + hour,
            "prediction_mw": 100 + hour + daily[hour // 24],
        }
        for model, daily in errors.items()
        for hour, timestamp in enumerate(times)
    ]
    # Irrelevant policy must not enter the paired calculation.
    rows.append({**rows[0], "policy": "leaky_comparison", "prediction_mw": 99999})
    source = tmp_path / "predictions.csv.gz"
    pd.DataFrame(rows).sample(frac=1, random_state=7).to_csv(source, index=False)
    outputs = [tmp_path / f"result_{i}.csv" for i in range(2)]
    for output in outputs:
        subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    result = pd.read_csv(outputs[0])
    assert list(result.columns) == [
        "comparison",
        "observed_diff_mw",
        "ci_low_mw",
        "ci_high_mw",
        "excludes_zero",
    ]
    pairs = [
        ("RandomForest", "Weekly_168h"),
        ("RandomForest", "HistGradientBoosting"),
        ("HistGradientBoosting", "Weekly_168h"),
        ("LightGBM", "Weekly_168h"),
        ("XGBoost", "Weekly_168h"),
    ]
    # Independent hour-level implementation checks circular wrap and odd-day truncation.
    starts = np.random.RandomState(42).randint(0, 3, size=(10_000, 2))
    hours = np.array(
        [
            [((start * 24 + offset) % 72) for start in sample for offset in range(48)][:72]
            for sample in starts
        ]
    )
    for row, (left, right) in zip(result.itertuples(index=False), pairs, strict=True):
        delta = np.repeat(np.array(errors[left]) - errors[right], 24)
        low, high = np.percentile(delta[hours].mean(axis=1), [2.5, 97.5])
        assert row.comparison == f"{left} minus {right}"
        np.testing.assert_allclose(
            [row.observed_diff_mw, row.ci_low_mw, row.ci_high_mw],
            [delta.mean(), low, high],
            atol=1e-9,
        )
        assert row.excludes_zero == bool(low > 0 or high < 0)
    assert len(result) == len(pairs)
