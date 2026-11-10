"""Paired MAE differences using circular blocks of two Berlin delivery days."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports" / "benchmark_corrected"
COMPARISONS = (
    ("RandomForest", "Weekly_168h"),
    ("RandomForest", "HistGradientBoosting"),
    ("HistGradientBoosting", "Weekly_168h"),
    ("LightGBM", "Weekly_168h"),
    ("XGBoost", "Weekly_168h"),
)


def paired_differences(source: Path) -> pd.DataFrame:
    """Use 10,000 shared resamples, seed 42, and percentile 95% intervals.

    For the saved 35-day evaluation, draw 18 circular two-day blocks and
    truncate to 35 days. Smaller complete-day inputs use ceil(days / 2) blocks.
    Sampling daily mean loss differences is equivalent to sampling all 24
    paired hours in each selected day. DST days are deliberately rejected.
    """
    frame = pd.read_csv(source)
    models = sorted({model for pair in COMPARISONS for model in pair})
    frame = frame[(frame.policy == "corrected") & frame.model.isin(models)].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    if frame.empty or frame.duplicated(["timestamp", "model"]).any():
        raise ValueError("Expected unique corrected predictions for every model and hour")
    if not frame.groupby("timestamp").actual_mw.nunique(dropna=False).eq(1).all():
        raise ValueError("Actual load must agree across paired models")
    predictions = frame.pivot(index="timestamp", columns="model", values="prediction_mw")
    predictions = predictions.reindex(columns=models).sort_index()
    actual = frame.groupby("timestamp").actual_mw.first().reindex(predictions.index)
    if not np.isfinite(predictions.to_numpy()).all() or not np.isfinite(actual).all():
        raise ValueError("Expected finite actuals and complete paired model coverage")
    times = predictions.index
    local = times.tz_convert("Europe/Berlin")
    n_days, remainder = divmod(len(times), 24)
    if (
        remainder
        or local[0].hour != 0
        or local[0].minute != 0
        or local[0].second != 0
        or local[0].microsecond != 0
        or local[0].nanosecond != 0
        or not times.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()
        or not pd.Series(local.date).value_counts().eq(24).all()
    ):
        raise ValueError("Expected consecutive complete 24-hour Berlin delivery days")

    rng = np.random.RandomState(42)
    starts = rng.randint(0, n_days, size=(10_000, (n_days + 1) // 2))
    days = ((starts[:, :, None] + np.arange(2)) % n_days).reshape(10_000, -1)[:, :n_days]
    losses = predictions.sub(actual, axis=0).abs()
    rows = []
    for left, right in COMPARISONS:
        differences = (losses[left] - losses[right]).to_numpy()
        daily = differences.reshape(n_days, 24).mean(axis=1)
        replicates = daily[days].mean(axis=1)
        low, high = np.percentile(replicates, [2.5, 97.5], method="linear")
        rows.append(
            {
                "comparison": f"{left} minus {right}",
                "observed_diff_mw": differences.mean(),
                "ci_low_mw": low,
                "ci_high_mw": high,
                "excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPORTS / "per_hour_predictions.csv.gz")
    parser.add_argument("--output", type=Path, default=REPORTS / "paired_differences.csv")
    args = parser.parse_args()
    result = paired_differences(args.input)
    result.to_csv(args.output, index=False, float_format="%.9f")
    print(result.to_csv(index=False, float_format="%.9f"), end="")


if __name__ == "__main__":
    main()
