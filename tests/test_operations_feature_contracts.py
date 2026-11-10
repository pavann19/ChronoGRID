"""Frozen numerical behavior, exhaustive horizon dependencies and pre-load drift checks."""

import json
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from chronogrid.features.engineer import FeatureEngineer
from chronogrid.operations.capture import delivery_grid
from chronogrid.operations.feature_contracts import (
    MANIFEST,
    ROOT,
    SOURCE_PATHS,
    feature_dependencies,
    load_binding,
    validate_dependencies,
)
from chronogrid.operations.local_calendar import LocalCalendarFeatures


@pytest.mark.parametrize("day,length", [("2024-02-15", 24), ("2024-03-31", 23), ("2024-10-27", 25)])
@pytest.mark.parametrize("local_calendar", [False, True])
def test_all_features_all_delivery_hours_match_declared_dependencies(day, length, local_calendar):
    grid = delivery_grid(date.fromisoformat(day))
    assert len(grid) == length
    index = pd.date_range(grid[0] - timedelta(hours=216), grid[-1], freq="h")
    values = 50000 + np.arange(len(index), dtype=float) ** 2
    history = dict(zip(index.to_pydatetime(), values))
    frame = pd.DataFrame(dict(timestamp=index, grid_load_mw=values))
    fe = (
        LocalCalendarFeatures()
        if local_calendar
        else FeatureEngineer(min_lag_hours=48, lags=[48, 72, 168], weather_mode="load_only")
    )
    features = fe._generate_raw_features(frame).set_index("timestamp").loc[list(grid)]
    rules = feature_dependencies()
    binding = dict(dependencies=rules)
    available = grid[-1] - timedelta(hours=47)  # End of latest required load hour.
    assert validate_dependencies(binding, grid, history, available, available) > 0
    for delivery in grid:
        for feature, offsets in rules.items():
            if not offsets:  # Calendar has no observed-load dependency.
                continue
            source = np.array([history[delivery - timedelta(hours=h)] for h in offsets])
            if "_lag_" in feature:
                expected = source[0]
            elif "_diff_" in feature:
                expected = source[0] - source[1]
            elif "_mean_" in feature:
                expected = source.mean()
            elif "_std_" in feature:
                expected = source.std(ddof=1)
            elif "_min_" in feature:
                expected = source.min()
            else:
                expected = source.max()
            assert features.loc[delivery, feature] == pytest.approx(expected)
    # Perturb every post-cutoff value, including pre-delivery future actuals.
    altered = frame.copy()
    altered.loc[altered.timestamp >= pd.Timestamp(available), "grid_load_mw"] = -999999
    changed = fe._generate_raw_features(altered).set_index("timestamp").loc[list(grid), list(rules)]
    pd.testing.assert_frame_equal(features[list(rules)], changed)
    with pytest.raises(ValueError, match="unavailable"):
        validate_dependencies(binding, grid, history, available - timedelta(seconds=1), available)
    history.pop(grid[0] - timedelta(hours=215))
    with pytest.raises(ValueError, match="missing source dependency"):
        validate_dependencies(binding, grid, history, available, available)


@pytest.mark.parametrize("experiment", ["issuance_v2", "local_calendar_v3"])
def test_existing_artifacts_have_exact_complete_bindings(experiment):
    directory = ROOT / "artifacts/experiments" / experiment
    metadata = json.loads((directory / "fallback_metadata.json").read_bytes())
    binding, raw, code_hash = load_binding(directory, metadata)
    assert len(binding["dependencies"]) == 31 and len(code_hash) == 64
    assert raw == (ROOT / MANIFEST).read_bytes()


@pytest.mark.parametrize(
    "target",
    [
        "fallback.joblib",
        "fallback_calibration.json",
        "fallback_metadata.json",
        "source",
        "inventory",
        "schema",
    ],
)
def test_drift_rejected_without_loading_model(tmp_path, target):
    directory = tmp_path / "experiment"
    shutil.copytree(ROOT / "artifacts/experiments/issuance_v2", directory)
    metadata = json.loads((directory / "fallback_metadata.json").read_bytes())
    for name in (*SOURCE_PATHS, MANIFEST):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    if target in {"source", "inventory"}:
        if target == "source":
            with (tmp_path / SOURCE_PATHS[0]).open("ab") as stream:
                stream.write(b"\n# simulated drift\n")
        else:
            manifest = json.loads((tmp_path / MANIFEST).read_bytes())
            policy = manifest["policies"][metadata["feature_policy"]]
            policy["sources"].pop(SOURCE_PATHS[0])
            (tmp_path / MANIFEST).write_text(json.dumps(manifest))
    elif target == "schema":
        metadata["feature_names"] = list(reversed(metadata["feature_names"]))
    else:
        with (directory / target).open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="binding|schema"):
        load_binding(directory, metadata, root=tmp_path)
