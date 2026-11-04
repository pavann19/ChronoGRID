"""Same-origin read-only operations console. No legacy write endpoints mounted."""
import csv
import hashlib
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from src.operations.journal import Journal, IntegrityFailure
from src.operations.reporting import score_run

ROOT=Path(__file__).resolve().parents[2]

def create_app(database: Path | None=None,root: Path=ROOT):
    app=FastAPI(title='ChronoGrid Research Operations',version='2.0.0')
    store=Journal(database or Path(os.getenv('CHRONOGRID_JOURNAL','data/operations/journal.sqlite')))
    app.state.journal=store
    manifest_path=root/'artifacts/reports/research_closure_evidence_manifest.json'
    def historical(name):
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        sidecar=manifest_path.with_suffix('.sha256').read_text().split()[0]
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=sidecar: raise IntegrityFailure('historical manifest mismatch')
        path=root/'artifacts/reports'/name
        expected=next(x['sha256'] for x in manifest['files'] if x['path']==path.relative_to(root).as_posix())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: raise IntegrityFailure('historical file mismatch')
        return path
    @app.middleware('http')
    async def headers(request,call_next):
        try: response=await call_next(request)
        except (IntegrityFailure,ValueError,KeyError,StopIteration,OSError): response=JSONResponse(status_code=503,content={'detail':'Evidence unavailable or integrity verification failed'})
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response
    @app.get('/health')
    def health(): return dict(status='alive',service='research-operations',live_collector_running=False)
    @app.get('/ready')
    def ready(): return dict(journal=store.verify(),scope='read-only operations; not forecast or production readiness')
    @app.get('/api/overview')
    def overview():
        integrity=store.verify()
        audit=json.loads(historical('ddir_sprint_3_3_1_numerical_reconciliation.json').read_text())
        return dict(research_decision='CONDITIONAL',historical_metrics=audit['metrics'],weather_comparison=audit['weather_comparison'],integrity=integrity,prospective_forecasts=sum(x['payload']['experiment_kind']=='prospective' for x in store.rows('forecast')),runs=[x['payload'] for x in store.rows('run')],failures=[x['payload'] for x in store.rows('failure')],live_collector_running=False,independent_receipts=sum(x['payload']['trust_scope']=='external_configured' for x in store.rows('receipt')),limitations=audit['limitations'])
    @app.get('/api/historical/origins')
    def origins():
        with historical('ddir_sprint_3_3_1_per_origin.csv').open(newline='') as f:
            return [x for x in csv.DictReader(f) if x['series']=='primary_static_marginal_90']
    @app.get('/api/historical/forecast')
    def forecast(origin: int=Query(1,ge=1,le=35),head: str=Query('primary',pattern='^(primary|fallback)$')):
        with historical('ddir_sprint_3_3_1_reconstructed_per_hour.csv').open(newline='') as f:
            rows=[x for x in csv.DictReader(f) if x['series']==head+'_static_marginal_90' and int(x['origin_index'])==origin]
        return dict(evidence_kind='historical_reconstruction',original_hourly_record_available=False,rows=rows)
    @app.get('/api/evidence')
    def evidence():
        store.verify()
        forecasts=[dict(sha256=x['sha256'],recorded_at=x['recorded_at'],forecast=x['payload']) for x in store.rows('forecast')]
        snapshots=[]
        for x in store.rows('snapshot'):
            p=x['payload']; snapshots.append({k:v for k,v in p.items() if k!='content_base64'})
        return dict(forecasts=forecasts,snapshots=snapshots,receipts=[{k:v for k,v in x['payload'].items() if not k.endswith('_base64')} for x in store.rows('receipt')],failures=[x['payload'] for x in store.rows('failure')],worker_errors=[x['payload'] for x in store.rows('worker_error')])
    @app.get('/api/report/{run_id}')
    def report(run_id: str):
        try: return score_run(store,run_id,store.clock(),replicates=1000)
        except KeyError as exc: raise HTTPException(404,'Unknown registered run') from exc
    @app.get('/api/forecast/{run_id}/{day}/{arm}')
    def committed_forecast(run_id: str,day: str,arm: str):
        store.verify()
        try: record=store.get(f'forecast:{run_id}:{day}:{arm}')
        except KeyError as exc: raise HTTPException(404,'Unknown forecast commitment') from exc
        from src.operations.journal import encode,digest
        key=digest(encode(record)); actuals={}
        for row in store.rows('actual'):
            a=row['payload']
            if a['prediction_sha256']==key and a['delivery_timestamp'] not in actuals:
                actuals[a['delivery_timestamp']]=a
        return dict(forecast=record,actuals=list(actuals.values()),actual_policy='preliminary first-collected vintage; use maturity-aware report for scored outcomes',independent_timestamp=False)
    @app.get('/api/research/issuance-v2')
    def issuance_research():
        path=root/'artifacts/experiments/issuance_v2/report.json'
        if not path.exists(): raise HTTPException(404,'V2 research artifact unavailable')
        return json.loads(path.read_text())
    @app.get('/api/adaptive')
    def adaptive_streams():
        store.verify()
        from src.operations.adaptive import AdaptiveStreams
        configs=[r['payload'] for r in store.rows('aci_config')]
        return dict(configurations=configs,states=[AdaptiveStreams(store).state(c['run_id'],c['arm']) for c in configs],transitions=[r['payload'] for r in store.rows('aci_step')],scope='daily-batch delayed research adaptation; no unconditional guarantee')
    @app.get('/api/models')
    def models():
        metadata=[]
        for rel in ['artifacts/models/champion_model.onnx','artifacts/models/load_only_model.onnx','artifacts/models/research/research_primary_model_thru_20240331.joblib','artifacts/models/research/research_fallback_model_thru_20240331.joblib','artifacts/experiments/issuance_v2/primary.joblib','artifacts/experiments/issuance_v2/fallback.joblib']:
            path=root/rel
            metadata.append(dict(path=rel,present=path.exists(),sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None))
        return dict(artifacts=metadata,compatibility='Research calibration belongs to research joblib identities, not production ONNX. No validated full-day prospective model calibration is deployed.')
    @app.get('/')
    def index(): return FileResponse(root/'frontend/operations/index.html')
    app.mount('/assets',StaticFiles(directory=root/'frontend/operations'),name='assets')
    return app
