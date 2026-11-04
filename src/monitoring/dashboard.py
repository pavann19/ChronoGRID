"""
Interactive Operational Dashboard for ChronoGrid (P5.11).
Built with Streamlit for real-time forecast inspection, drift telemetry, and model registry exploration.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def run_dashboard():
    """Renders the Streamlit ChronoGrid operations dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="ChronoGrid MLOps Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚡ ChronoGrid — Research Platform")
    st.info("For corrected frozen research and evidence integrity, run the operations console. Missing registry values remain unavailable.")
    st.markdown("Real-world German Electricity Grid Load & DWD Meteorological Forecasting System")

    # Load Database state if available
    from src.db.database import get_session
    from src.db.models import ModelRegistryRecord, DriftReportRecord, PipelineRun

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    champion_rec = None
    try:
        with get_session() as session:
            champion_rec = session.query(ModelRegistryRecord).filter_by(is_champion=True).first()
            drift_rec = session.query(DriftReportRecord).order_by(DriftReportRecord.timestamp.desc()).first()
            runs_count = session.query(PipelineRun).count()
    except Exception:
        drift_rec = None
        runs_count = 0

    with col1:
        st.metric(
            "Champion Model",
            f"v{champion_rec.version} ({champion_rec.model_type})" if champion_rec else "Unavailable",
            help="Registry identity; production operation is not established by this display"
        )
    with col2:
        st.metric(
            "Champion MAE",
            f"{champion_rec.mae:.1f} MW" if champion_rec and champion_rec.mae else "Unavailable",
            help="Mean Absolute Error on holdout evaluation"
        )
    with col3:
        st.metric(
            "Champion MAPE",
            f"{champion_rec.mape:.2f}%" if champion_rec and champion_rec.mape else "Unavailable",
            help="Mean Absolute Percentage Error"
        )
    with col4:
        drift_val = f"{drift_rec.overall_drift_score:.3f}" if drift_rec else "Unavailable"
        st.metric(
            "Overall PSI Drift",
            drift_val,
            delta="Normal" if drift_rec and not drift_rec.drift_detected else "Action Required",
            delta_color="normal" if drift_rec and not drift_rec.drift_detected else "inverse"
        )

    st.divider()

    # Tabs for Views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast & Grid Telemetry",
        "🔍 Data & Model Drift",
        "🏛️ Model Registry",
        "🚀 Live API Testing"
    ])

    with tab1:
        st.subheader("Historical Grid Load & Day-Ahead Predictions")
        data_path = Path("data/processed/smard_dwd_hourly.parquet")
        if data_path.exists():
            df = pd.read_parquet(data_path)
            sample_df = df.tail(168).copy() # Last 7 days
            st.line_chart(sample_df.set_index("timestamp")[["grid_load_mw", "temperature_c"]])
            st.caption("Hourly German Grid Load (MW) and Synchronized Temperature (°C) for past 7 days.")
        else:
            st.info("Processed dataset not yet generated. Run ingestion pipeline first.")

    with tab2:
        st.subheader("Drift Monitoring & Evidently Telemetry")
        drift_report_html = Path("artifacts/reports/drift_report.html")
        if drift_report_html.exists():
            st.success("Evidently-compatible Drift Report available.")
            with open(drift_report_html, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        else:
            st.info("No drift report generated yet. Run monitoring task.")

    with tab3:
        st.subheader("ChronoGrid Model Registry Versions")
        try:
            with get_session() as session:
                records = session.query(ModelRegistryRecord).order_by(ModelRegistryRecord.version.desc()).all()
                if records:
                    table_data = [{
                        "Version": r.version,
                        "Model Type": r.model_type,
                        "Stage": r.stage,
                        "Champion": "⭐ YES" if r.is_champion else "No",
                        "MAE (MW)": r.mae,
                        "RMSE (MW)": r.rmse,
                        "MAPE (%)": r.mape,
                        "Promoted At": str(r.promoted_at) if r.promoted_at else "-",
                    } for r in records]
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                else:
                    st.info("Model registry empty. Train initial models first.")
        except Exception as e:
            st.error(f"Error loading registry: {e}")

    with tab4:
        st.subheader("Live Inference Simulator (FastAPI + ONNX)")
        st.write("Submit features to test sub-millisecond ONNX Runtime latency:")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            temp = st.number_input("Temperature (°C)", value=12.5)
            hour = st.slider("Hour of Day (UTC)", 0, 23, 11)
        with col_b:
            lag24 = st.number_input("Load Lag 24h (MW)", value=64000.0)
            lag168 = st.number_input("Load Lag 168h (MW)", value=65200.0)
        with col_c:
            solar = st.number_input("Solar Irradiance (W/m²)", value=450.0)
            wind = st.number_input("Wind Speed (m/s)", value=5.2)

        if st.button("Generate ONNX Prediction"):
            from src.serving.onnx_engine import OnnxInferenceEngine
            try:
                engine = OnnxInferenceEngine()
                features = {
                    "temperature_c": temp,
                    "grid_load_mw_lag_24": lag24,
                    "grid_load_mw_lag_168": lag168,
                    "solar_radiation_wm2": solar,
                    "wind_speed_ms": wind,
                    "sin_hour": np.sin(2 * np.pi * hour / 24),
                    "cos_hour": np.cos(2 * np.pi * hour / 24),
                }
                pred = engine.predict([features])[0]
                st.success(f"Predicted Grid Load: **{pred:,.1f} MW**")
            except Exception as e:
                st.error(f"Inference error: {e}")


if __name__ == "__main__":
    run_dashboard()

