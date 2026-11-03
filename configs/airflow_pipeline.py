"""
Airflow DAG definition for ChronoGrid Orchestration (P5.2).
Demonstrates end-to-end scheduled batch execution:
Ingestion -> Validation -> Feature Engineering -> Model Evaluation -> Drift Check.
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_ingest(**context):
    from src.ingestion.ingest_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    df = pipeline.run(use_live_api=False)
    return len(df)


def task_validate_clean(**context):
    import pandas as pd
    from src.pipeline.pipeline import DataPipeline
    df_raw = pd.read_parquet("data/raw/latest_raw.parquet")
    pipeline = DataPipeline()
    df_clean, metrics = pipeline.process(df_raw, sync_to_db=True)
    return len(df_clean)


def task_feature_engineering(**context):
    import pandas as pd
    from src.features.engineer import FeatureEngineer
    df_clean = pd.read_parquet("data/processed/smard_dwd_hourly.parquet")
    fe = FeatureEngineer()
    df_features = fe.fit_transform(df_clean)
    df_features.to_parquet("data/processed/features_hourly.parquet", index=False)
    return len(df_features)


def task_drift_check(**context):
    import pandas as pd
    from src.monitoring.drift_detector import DriftDetector
    df = pd.read_parquet("data/processed/smard_dwd_hourly.parquet")
    ref = df.iloc[-720:-168]  # Reference: prior month
    curr = df.iloc[-168:]     # Current: latest week
    detector = DriftDetector()
    report = detector.compute_drift(ref, curr)
    return report.drift_detected


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "chronogrid_mlops",
        "depends_on_past": False,
        "start_date": datetime(2024, 1, 1),
        "email_on_failure": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "chronogrid_production_forecasting_pipeline",
        default_args=default_args,
        description="Hourly/Daily ChronoGrid Electricity Forecasting and MLOps Pipeline",
        schedule_interval="@daily",
        catchup=False,
    )

    t1 = PythonOperator(
        task_id="ingest_smard_dwd",
        python_callable=task_ingest,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="validate_and_clean_data",
        python_callable=task_validate_clean,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="time_series_feature_engineering",
        python_callable=task_feature_engineering,
        dag=dag,
    )

    t4 = PythonOperator(
        task_id="monitor_drift_and_retrain_gate",
        python_callable=task_drift_check,
        dag=dag,
    )

    t1 >> t2 >> t3 >> t4

