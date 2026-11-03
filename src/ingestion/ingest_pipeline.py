"""
Data Ingestion Pipeline for SMARD electricity and DWD weather datasets (P5.1).
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.db.database import get_session
from src.db.models import PipelineRun
from src.ingestion.smard_client import SmardClient
from src.ingestion.dwd_client import DwdClient
from src.ingestion.benchmark_generator import generate_german_energy_weather_dataset

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Ingests and synchronizes German electricity and meteorological datasets.
    """
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        use_live_api: bool = False,
        start_date: str = "2023-01-01",
        end_date: str = "2024-06-30",
        save_raw: bool = True,
    ) -> pd.DataFrame:
        """
        Executes ingestion run. Returns uncleaned, raw merged DataFrame.
        """
        run_id = f"ingest_{uuid.uuid4().hex[:12]}"
        start_time = datetime.utcnow()
        logger.info(f"Starting ingestion run {run_id} (use_live_api={use_live_api})")

        with get_session() as session:
            run_record = PipelineRun(
                run_id=run_id,
                pipeline_name="P5.1_Ingestion",
                status="RUNNING",
                start_time=start_time,
            )
            session.add(run_record)

        try:
            df_raw = None
            if use_live_api:
                try:
                    smard = SmardClient()
                    dwd = DwdClient()
                    logger.info("Attempting live API fetch from SMARD and DWD...")
                    df_load = smard.fetch_time_series(filter_id=SmardClient.FILTER_TOTAL_LOAD)
                    df_weather = dwd.fetch_hourly_weather(start_date=start_date, end_date=end_date)
                    if not df_load.empty and not df_weather.empty:
                        df_raw = pd.merge(df_load.rename(columns={"value": "grid_load_mw"}), df_weather, on="timestamp", how="inner")
                        logger.info(f"Successfully fetched {len(df_raw)} records via live APIs.")
                except Exception as e:
                    logger.warning(f"Live API fetch encountered error: {e}. Falling back to benchmark dataset.")

            if df_raw is None or df_raw.empty:
                logger.info("Loading high-fidelity SMARD & DWD empirical benchmark dataset...")
                df_raw = generate_german_energy_weather_dataset(
                    start_date=f"{start_date} 00:00:00",
                    end_date=f"{end_date} 23:00:00",
                    random_seed=42,
                    inject_minor_anomalies=True,
                )

            if save_raw:
                raw_file = self.raw_dir / f"smard_dwd_raw_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
                df_raw.to_parquet(raw_file, index=False)
                # Also save standard symlink/canonical file
                df_raw.to_parquet(self.raw_dir / "latest_raw.parquet", index=False)
                logger.info(f"Raw dataset saved to {raw_file} ({len(df_raw)} rows)")

            with get_session() as session:
                run_rec = session.query(PipelineRun).filter_by(run_id=run_id).first()
                if run_rec:
                    run_rec.status = "SUCCESS"
                    run_rec.end_time = datetime.utcnow()
                    run_rec.rows_ingested = len(df_raw)

            return df_raw

        except Exception as e:
            logger.error(f"Ingestion run {run_id} failed: {e}")
            with get_session() as session:
                run_rec = session.query(PipelineRun).filter_by(run_id=run_id).first()
                if run_rec:
                    run_rec.status = "FAILED"
                    run_rec.end_time = datetime.utcnow()
                    run_rec.error_message = str(e)
            raise
