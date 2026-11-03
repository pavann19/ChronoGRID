"""
End-to-End Validated Data Pipeline (P5.2).
Takes raw data, validates it, cleans it, updates PostgreSQL/SQLite DB, and exports processed artifacts.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from src.db.database import get_session, init_db
from src.db.models import ElectricityRecord, PipelineRun
from src.pipeline.validation import DataValidator
from src.pipeline.cleaning import DataCleaner

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Executes reproducible, validated data preparation.
    """
    def __init__(
        self,
        processed_dir: str = "data/processed",
        canonical_filename: str = "smard_dwd_hourly.parquet",
    ):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.canonical_path = self.processed_dir / canonical_filename
        self.validator = DataValidator()
        self.cleaner = DataCleaner()

    def process(self, df_raw: pd.DataFrame, sync_to_db: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Runs validation, cleaning, persistence, and optional database synchronization.
        """
        run_id = f"pipeline_{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()
        logger.info(f"Starting Data Pipeline run {run_id} on {len(df_raw)} raw rows.")

        with get_session() as session:
            run_rec = PipelineRun(
                run_id=run_id,
                pipeline_name="P5.2_ValidatedPipeline",
                status="RUNNING",
                start_time=start_time,
            )
            session.add(run_rec)

        try:
            # 1. Pre-cleaning Validation
            pre_report = self.validator.validate(df_raw)

            # 2. Cleaning & Gap Resolution
            df_cleaned, clean_stats = self.cleaner.clean(df_raw)

            # 3. Post-cleaning Validation
            post_report = self.validator.validate(df_cleaned)

            # 4. Save to Parquet
            df_cleaned.to_parquet(self.canonical_path, index=False)
            logger.info(f"Saved processed dataset to {self.canonical_path} ({len(df_cleaned)} rows).")

            # 5. Database Synchronization (Bulk upsert)
            if sync_to_db:
                self._sync_to_database(df_cleaned)

            pipeline_metrics = {
                "run_id": run_id,
                "pre_validation": pre_report,
                "cleaning": clean_stats,
                "post_validation": post_report,
                "processed_rows": len(df_cleaned),
                "canonical_file": str(self.canonical_path),
            }

            with get_session() as session:
                rec = session.query(PipelineRun).filter_by(run_id=run_id).first()
                if rec:
                    rec.status = "SUCCESS"
                    rec.end_time = datetime.utcnow()
                    rec.rows_ingested = len(df_raw)
                    rec.rows_validated = len(df_cleaned)
                    rec.metrics_json = json.dumps(pipeline_metrics, default=str)

            return df_cleaned, pipeline_metrics

        except Exception as e:
            logger.error(f"Data Pipeline run {run_id} failed: {e}")
            with get_session() as session:
                rec = session.query(PipelineRun).filter_by(run_id=run_id).first()
                if rec:
                    rec.status = "FAILED"
                    rec.end_time = datetime.utcnow()
                    rec.error_message = str(e)
            raise

    def _sync_to_database(self, df: pd.DataFrame, batch_size: int = 2000):
        """Syncs cleaned records into the database."""
        logger.info(f"Syncing {len(df)} records into the database...")
        init_db()
        with get_session() as session:
            # Wipe older staging if needed or insert missing
            existing_count = session.query(ElectricityRecord).count()
            if existing_count == 0:
                records = []
                for _, row in df.iterrows():
                    ts = row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"]
                    rec = ElectricityRecord(
                        timestamp=ts,
                        grid_load_mw=float(row["grid_load_mw"]),
                        day_ahead_price_eur_mwh=float(row.get("day_ahead_price_eur_mwh", 0.0)) if pd.notnull(row.get("day_ahead_price_eur_mwh")) else None,
                        solar_generation_mw=float(row.get("solar_generation_mw", 0.0)) if pd.notnull(row.get("solar_generation_mw")) else None,
                        wind_generation_mw=float(row.get("wind_generation_mw", 0.0)) if pd.notnull(row.get("wind_generation_mw")) else None,
                        temperature_c=float(row.get("temperature_c", 0.0)) if pd.notnull(row.get("temperature_c")) else None,
                        humidity_pct=float(row.get("humidity_pct", 0.0)) if pd.notnull(row.get("humidity_pct")) else None,
                        wind_speed_ms=float(row.get("wind_speed_ms", 0.0)) if pd.notnull(row.get("wind_speed_ms")) else None,
                        solar_radiation_wm2=float(row.get("solar_radiation_wm2", 0.0)) if pd.notnull(row.get("solar_radiation_wm2")) else None,
                        is_interpolated=bool(row.get("is_interpolated", False)),
                        quality_flag=str(row.get("quality_flag", "VALID")),
                    )
                    records.append(rec)
                    if len(records) >= batch_size:
                        session.bulk_save_objects(records)
                        session.commit()
                        records = []
                if records:
                    session.bulk_save_objects(records)
                    session.commit()
                logger.info("Database bulk sync completed successfully.")
            else:
                logger.info(f"Database already contains {existing_count} records. Skipping initial bulk re-insert.")
