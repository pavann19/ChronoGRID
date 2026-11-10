"""Corrected and explicitly post-gate comparisons on identical historical hours."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from src.features.engineer import FeatureEngineer
from src.features.gate_audit import audit_gate_availability
from src.models.walk_forward import TimeSeriesWalkForwardValidator

OUT = ROOT / 'artifacts/reports/benchmark_corrected'

def metrics(y, p):
    error = np.asarray(y) - np.asarray(p)
    return np.array([np.abs(error).mean(), np.sqrt((error**2).mean()), (np.abs(error)/np.abs(y)).mean()*100])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/raw/genuine_smard_dwd_observations.parquet'
    raw = pd.read_parquet(source).sort_values('timestamp').reset_index(drop=True)
    raw.timestamp = pd.to_datetime(raw.timestamp, utc=True)
    assert raw.timestamp.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    factories = {
        'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=10)),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=2),
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=.05, random_state=42, early_stopping=False),
        'LightGBM': lambda: LGBMRegressor(n_estimators=200, learning_rate=.05, num_leaves=31, random_state=42, n_jobs=2, verbosity=-1),
        'XGBoost': lambda: XGBRegressor(n_estimators=200, learning_rate=.05, max_depth=6, random_state=42, n_jobs=2, tree_method='hist'),
    }
    predictions = []; folds = []; gates = []
    for lag, label in [(48, 'corrected'), (24, 'leaky_comparison')]:
        fe = FeatureEngineer(min_lag_hours=lag, allow_post_gate_lags=lag==24)
        full = fe.fit_transform(raw)
        names = fe.get_feature_names()
        cv = TimeSeriesWalkForwardValidator(n_splits=5, test_size_hours=168, purge_gap_hours=24)
        for fold, (train_idx, test_idx) in enumerate(cv.split(full), 1):
            train, test = full.iloc[train_idx], full.iloc[test_idx]
            gate = audit_gate_availability(test.timestamp, names, lag, assert_available=lag==48)
            gates.append(dict(policy=label, fold=fold, delivery_hours=len(test), post_gate_hours=int(gate.groupby('delivery_utc').available.all().eq(False).sum())))
            folds.append(dict(policy=label, fold=fold, train_rows=len(train), train_end=train.timestamp.iloc[-1].isoformat(), test_start=test.timestamp.iloc[0].isoformat(), test_end=test.timestamp.iloc[-1].isoformat(), purge_hours=24))
            candidates = {f'Naive_{lag}h': test[f'grid_load_mw_lag_{lag}'].to_numpy(), 'Weekly_168h': test.grid_load_mw_lag_168.to_numpy()}
            for name, factory in factories.items():
                print(label, fold, name, flush=True)
                model = factory(); model.fit(train[names], train.grid_load_mw)
                candidates[name] = model.predict(test[names])
            for name, pred in candidates.items():
                for ts, actual, point in zip(test.timestamp, test.grid_load_mw, pred):
                    predictions.append(dict(policy=label, model=name, fold=fold, timestamp=ts.isoformat(), actual_mw=actual, prediction_mw=float(point)))
    frame = pd.DataFrame(predictions)
    # Latest downloadable TSO series; no archival issuance-vintage claim.
    tso = dict(series=411, status='unverified', vintage='latest downloaded revision; original issue timestamp unavailable', source='https://www.smard.de/page/en/wiki-article/6080/206318/forecast-data')
    try:
        session = requests.Session(); base='https://www.smard.de/app/chart_data/411/DE'
        response=session.get(base+'/index_hour.json', timeout=30);response.raise_for_status()
        (OUT/'tso_index.json').write_bytes(response.content)
        slices=response.json()['timestamps']; times=pd.DatetimeIndex(pd.to_datetime(frame.timestamp.unique(),utc=True)); start,end=times.min(),times.max()
        selected=[v for v in slices if pd.Timestamp(v,unit='ms',tz='UTC')<=end and pd.Timestamp(v,unit='ms',tz='UTC')+pd.Timedelta(days=8)>=start]
        values={}
        for item in selected:
            response=session.get(f'{base}/411_DE_hour_{item}.json',timeout=30);response.raise_for_status()
            (OUT/f'tso_411_{item}.json').write_bytes(response.content)
            for t,v in response.json()['series']:
                if v is not None: values[pd.Timestamp(t,unit='ms',tz='UTC')]=float(v)
        ref=frame[(frame.policy=='corrected')&(frame.model=='Weekly_168h')].copy()
        ref['prediction_mw']=[values.get(pd.Timestamp(t), np.nan) for t in ref.timestamp]
        tso['matched_hours']=int(ref.prediction_mw.notna().sum());tso['required_hours']=len(ref)
        if ref.prediction_mw.notna().all():
            ref['model']='TSO_latest_revision';frame=pd.concat([frame,ref],ignore_index=True);tso['status']='available_latest_revision_only'
        else:tso['status']='incomplete_same_hour_coverage_not_scored'
    except Exception as exc:
        tso.update(status='download_failed', error=str(exc))
    frame.to_csv(OUT/'per_hour_predictions.csv', index=False)
    rows=[]
    for (policy,model),g in frame.groupby(['policy','model'],sort=True):
        g=g.sort_values('timestamp');y=g.actual_mw.to_numpy();p=g.prediction_mw.to_numpy()
        estimate=metrics(y,p);n=len(y);assert n==840
        rng=np.random.RandomState(42);starts=rng.randint(0,n//24,size=(2000,int(np.ceil((n//24)/2))))
        days=((starts[:,:,None]+np.arange(2))%(n//24)).reshape(2000,-1)[:,:n//24]
        indices=(days[:,:,None]*24+np.arange(24)).reshape(2000,n)
        errors=y[indices]-p[indices]
        samples=np.column_stack([np.abs(errors).mean(axis=1),np.sqrt((errors**2).mean(axis=1)),(np.abs(errors)/np.abs(y[indices])).mean(axis=1)*100])
        low,high=np.percentile(samples,[2.5,97.5],axis=0)
        row=dict(policy=policy,model=model,hours=n)
        for i,key in enumerate(['mae_mw','rmse_mw','mape_pct']):row.update({key:float(estimate[i]),key+'_ci_low':float(low[i]),key+'_ci_high':float(high[i])})
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(OUT/'folds.csv',index=False)
    pd.DataFrame(gates).to_csv(OUT/'gate_audit.csv',index=False)
    report=dict(dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), dataset_rows=len(raw), start=raw.timestamp.min().isoformat(), end=raw.timestamp.max().isoformat(), evaluation_days=35, folds=5,purge_hours=24,bootstrap=dict(method='circular moving blocks of two complete Berlin delivery days',replicates=2000,seed=42,confidence=.95),tso=tso,limitations=['Reused historical evaluation days; no fresh holdout','Observation event times only, not archived publication timestamps','Persistence weather, not NWP vintages','Fold models stay fixed over each test week; features updated at each daily gate'],gates=gates)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(table.to_string(index=False));print(json.dumps(tso,indent=2))
if __name__=='__main__':main()
