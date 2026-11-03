"""P0 Task 5: Reproduce the forecasting benchmark from raw data."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelBenchmarkTrainer

# Load genuine SMARD/DWD observations
df = pd.read_parquet(ROOT_DIR / "data/raw/genuine_smard_dwd_observations.parquet")
print(f"Dataset: {len(df):,} rows | {df['timestamp'].min()} to {df['timestamp'].max()}")

# Build feature matrix with strict 24h information boundary
fe = FeatureEngineer(min_lag_hours=24, weather_mode="persistence")
df_feat = fe.fit_transform(df, drop_warmup=True)
feat_cols = fe.get_feature_names()
print(f"Feature matrix: {df_feat.shape[0]:,} rows x {len(feat_cols)} features")
print()

# Run 5-fold walk-forward benchmark
trainer = ModelBenchmarkTrainer(
    target_col="grid_load_mw",
    n_splits=5,
    test_size_hours=168,
    purge_gap_hours=24,
)
summary_df, all_results, champion, pairwise_df = trainer.run_benchmark(df_feat, feat_cols)

# Print leaderboard
print("=== REPRODUCED LEADERBOARD (5-fold walk-forward, persistence weather) ===")
cols = ["Model", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "R2", "Diebold-Mariano vs Naive"]
print(summary_df[cols].to_string(index=False))
print()

champion_model = summary_df.iloc[0]["Model"]
champion_mae = summary_df.iloc[0]["MAE (MW)"]
champion_mape = summary_df.iloc[0]["MAPE (%)"]
naive_mae = summary_df[summary_df["Model"].str.contains("Naive_24h|Naive Persistence", na=False)]["MAE (MW)"].values[0]
improvement = (naive_mae - champion_mae) / naive_mae * 100

print(f"Champion          : {champion_model}")
print(f"Champion MAE      : {champion_mae:.2f} MW")
print(f"Champion MAPE     : {champion_mape:.3f}%")
print(f"Naive 24h MAE     : {naive_mae:.2f} MW")
print(f"MAE improvement   : {improvement:.1f}% over naive persistence")
print(f"Test folds        : {trainer.cv.n_splits}")
print(f"Test hours/fold   : {trainer.cv.test_size_hours}")
print(f"Purge gap         : {trainer.cv.purge_gap_hours}h")
total_test_hours = trainer.cv.n_splits * trainer.cv.test_size_hours
print(f"Total OOF hours   : {total_test_hours}")
