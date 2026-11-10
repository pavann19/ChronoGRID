"""Versioned German-local modelling calendar; frozen feature code stays unchanged."""
from datetime import date,timedelta
from functools import lru_cache
import numpy as np
import pandas as pd
from dateutil.easter import easter
from chronogrid.features.engineer import FeatureEngineer
from chronogrid.features.weather_features import add_weather_features
from chronogrid.features.autoregressive import add_autoregressive_features

POLICY='fixed_gate_load48_local_calendar_v3'

@lru_cache(maxsize=8)
def holidays(year):
    # Deliberately bounded modelling rule set. Review legislation before extending.
    if not 2023<=year<=2030:raise ValueError('outside V3 calendar policy years 2023-2030')
    spring=easter(year)
    return frozenset({date(year,1,1),date(year,5,1),date(year,10,3),date(year,12,25),date(year,12,26),spring-timedelta(days=2),spring+timedelta(days=1),spring+timedelta(days=39),spring+timedelta(days=50)})

def add_local_calendar(df):
    out=df.copy()
    if any(pd.isna(t) or pd.Timestamp(t).tzinfo is None for t in out.timestamp):
        raise ValueError('calendar timestamps must be present and timezone-aware')
    ts=pd.to_datetime(out.timestamp,utc=True).dt.tz_convert('Europe/Berlin')
    values={'hour':(ts.dt.hour,24,0),'dayofweek':(ts.dt.dayofweek,7,0),'month':(ts.dt.month,12,1),'dayofyear':(ts.dt.dayofyear,365.25,1)}
    for key,(v,period,offset) in values.items():
        out['sin_'+key]=np.sin(2*np.pi*(v-offset)/period)
        out['cos_'+key]=np.cos(2*np.pi*(v-offset)/period)
    out['is_weekend']=(ts.dt.dayofweek>=5).astype(int)
    out['is_holiday']=[int(t.date() in holidays(t.year)) for t in ts]
    out['is_working_day']=((out.is_weekend==0)&(out.is_holiday==0)).astype(int)
    return out

class LocalCalendarFeatures(FeatureEngineer):
    """Elapsed UTC lags; local calendar. Fixed load-only configuration, no tuning."""
    def __init__(self):
        super().__init__(min_lag_hours=48,lags=[48,72,168],weather_mode='load_only')
    def _generate_raw_features(self,df):
        out=add_local_calendar(df)
        ts=pd.to_datetime(out.timestamp,utc=True)
        if ts.duplicated().any() or not ts.diff().dropna().eq(pd.Timedelta(hours=1)).all():
            raise ValueError('V3 features require unique consecutive UTC hours')
        out=add_weather_features(out,weather_mode='load_only',min_lag_hours=48)
        return add_autoregressive_features(out,target_col=self.target_col,lags=self.lags,rolling_windows=self.rolling_windows,min_lag_hours=48)
