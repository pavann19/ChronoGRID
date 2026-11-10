import numpy as np
import pandas as pd
import pytest
from src.features.engineer import FeatureEngineer
from src.features.gate_audit import audit_gate_availability
from src.pipeline.cleaning import DataCleaner
from src.models.baselines import WeeklySeasonalPersistence168h

@pytest.mark.parametrize("day,end,hours", [("2024-03-31", "2024-04-01",23),("2024-10-27","2024-10-28",25),("2024-06-17","2024-06-18",24)])
def test_gate_old_lag_fails_corrected_passes(day,end,hours):
    times=pd.date_range(day,end,inclusive="left",freq="h",tz="Europe/Berlin")
    assert len(times)==hours
    with pytest.raises(ValueError,match="Post-gate"):
        audit_gate_availability(times,["grid_load_mw_lag_24"],24)
    result=audit_gate_availability(times,["grid_load_mw_lag_48","grid_load_mw_roll_mean_24","forecast_temperature_c","sin_hour"],48)
    assert result.available.all()

def test_default_requires_explicit_leaky_opt_in():
    assert FeatureEngineer().min_lag_hours==48
    with pytest.raises(ValueError,match="allow_post_gate_lags"):
        FeatureEngineer(min_lag_hours=24)

def test_causal_cleaning_never_uses_later_observation():
    df=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=8,freq="h",tz="UTC"),"grid_load_mw":[np.nan,10,np.nan,np.nan,np.nan,np.nan,np.nan,999]})
    cleaned,_=DataCleaner(max_interpolation_gap_hours=3).clean(df)
    assert cleaned.grid_load_mw.tolist()==[10,10,10,10,999]
    assert cleaned.timestamp.tolist()==df.timestamp.iloc[[1,2,3,4,7]].tolist()

def test_baseline_preserves_unknown_rows_without_future_fill():
    pred=WeeklySeasonalPersistence168h().predict(pd.DataFrame({"grid_load_mw_lag_168":[np.nan,10,np.nan,999]}))
    assert np.isnan(pred[0])
    assert pred[2]==10

def test_perturbation_suite_detects_a_future_reader():
    df=pd.DataFrame({"timestamp":pd.date_range("2024-01-01",periods=500,freq="h",tz="UTC"),"grid_load_mw":np.arange(500,dtype=float)})
    class Broken(FeatureEngineer):
        def _generate_raw_features(self,df):
            out=super()._generate_raw_features(df)
            out["bad_future"]=df.grid_load_mw.shift(-1)
            return out
    assert FeatureEngineer(weather_mode="load_only").run_leakage_suite(df)["suite_status"]=="ALL_TESTS_PASSED"
    result=Broken(weather_mode="load_only").run_leakage_suite(df)
    assert not result["details"]["3_future_load_perturbation"]["passed"]


def test_completed_observation_at_gate_is_available_but_start_at_gate_is_not():
    times=pd.to_datetime(["2024-06-17T09:00Z","2024-06-17T10:00Z"],utc=True)
    report=audit_gate_availability(times,["grid_load_mw_lag_24"],24,False)
    assert report.available.tolist()==[True,False]
