# Repository scripts bootstrap the checkout before importing the installed package.
# ruff: noqa: E402
"""
Train and Export Fallback Load-Only Model to ONNX.
Matches the HistGradientBoosting champion architecture using 34 load-only features.
Saves to artifacts/models/load_only_model.onnx and artifacts/models/load_only_metadata.json.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from chronogrid.features.engineer import (
    FeatureEngineer,
)
from chronogrid.models.evaluation import (
    compute_metrics,
)
from chronogrid.models.tree_models import (
    HistGradientBoostingModel,
)
from chronogrid.serving.onnx_exporter import (
    export_to_onnx,
)


def train_and_export_fallback():
    print("=== Training Fallback Load-Only Model for DDIR ===")
    raw_path = ROOT_DIR / "data/raw/genuine_smard_dwd_observations.parquet"
    df_raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(df_raw):,} records from {raw_path}")

    # Feature engineering in load_only mode (min_lag_hours=24)
    fe = FeatureEngineer(allow_post_gate_lags=True, min_lag_hours=24, weather_mode="load_only")
    df_feats = fe.fit_transform(df_raw, drop_warmup=True)
    feature_cols = fe.get_feature_names()
    print(f"Engineered {len(feature_cols)} load-only features (Zero weather dependency).")

    # Train on training split (e.g. prior to final 168h test window or full training corpus)
    # To match champion training, use the training corpus up to the final test window:
    train_df = df_feats.iloc[:-168]
    test_df = df_feats.iloc[-168:]

    X_train = train_df[feature_cols].values
    y_train = train_df["grid_load_mw"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["grid_load_mw"].values

    print(f"Training set: {len(X_train):,} hours | Test set: {len(X_test):,} hours")
    model = HistGradientBoostingModel(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=42
    )
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    metrics = compute_metrics(y_test, test_preds)
    print(
        f"Fallback Model Test Performance: MAE={metrics['mae']:.2f} MW | RMSE={metrics['rmse']:.2f} MW | MAPE={metrics['mape']:.3f}%"
    )

    # Fit on all available historical data (or keep train split) and export to ONNX
    onnx_out = ROOT_DIR / "artifacts/models/load_only_model.onnx"
    meta_out = ROOT_DIR / "artifacts/models/load_only_metadata.json"

    export_to_onnx(
        model_wrapper=model,
        feature_names=feature_cols,
        output_path=str(onnx_out),
        metadata_path=str(meta_out),
    )
    print(f"Exported Fallback ONNX model to: {onnx_out} ({onnx_out.stat().st_size:,} bytes)")
    print(f"Saved Fallback Metadata to: {meta_out}")


if __name__ == "__main__":
    train_and_export_fallback()
