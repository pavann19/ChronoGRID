"""Local CLI for registered capture, observations, evaluation and evidence recovery."""
import argparse
from datetime import date,datetime,timezone,timedelta
import json
from pathlib import Path
from src.operations.journal import Journal,digest
from src.operations.contracts import Run
from src.operations.capture import capture_smard,capture_weekly,capture_fallback48
from src.operations.reporting import score_run
from src.evaluation.prospective import ActualObservation
from src.operations.timestamps import make_request,verify_receipt

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,help='Verified research assets; defaults to CHRONOGRID_WORKSPACE or recognized source checkout')
    parser.add_argument('--db',type=Path,help='Explicit journal, or selected workspace data/operations/journal.sqlite')
    sub=parser.add_subparsers(dest='command',required=True)
    register=sub.add_parser('register'); register.add_argument('config',type=Path)
    sub.add_parser('verify'); sub.add_parser('fetch-smard'); sub.add_parser('fetch-weather')
    sub.add_parser('doctor')
    status=sub.add_parser('status'); status.add_argument('--run')
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8765)
    weather_run=sub.add_parser('fetch-weather-run');weather_run.add_argument('initialization',type=datetime.fromisoformat)
    weather_day=sub.add_parser('weather-delivery');weather_day.add_argument('snapshot');weather_day.add_argument('day',type=date.fromisoformat);weather_day.add_argument('output',type=Path)
    worker=sub.add_parser('run-once'); worker.add_argument('run'); worker.add_argument('arm'); worker.add_argument('--model',choices=['weekly','fallback48','adaptive48','local48'],default='weekly')
    cap=sub.add_parser('capture-weekly'); cap.add_argument('run'); cap.add_argument('day',type=date.fromisoformat); cap.add_argument('arm'); cap.add_argument('snapshot')
    cap48=sub.add_parser('capture-fallback48'); cap48.add_argument('run'); cap48.add_argument('day',type=date.fromisoformat); cap48.add_argument('arm'); cap48.add_argument('snapshot'); cap48.add_argument('--experiment-dir',type=Path)
    adaptive=sub.add_parser('register-aci'); adaptive.add_argument('config',type=Path)
    import_cal=sub.add_parser('import-calibration'); import_cal.add_argument('path',type=Path)
    state=sub.add_parser('adaptive-state'); state.add_argument('run'); state.add_argument('arm')
    settle=sub.add_parser('settle-adaptive'); settle.add_argument('run'); settle.add_argument('arm'); settle.add_argument('--day',type=date.fromisoformat); settle.add_argument('--max-origins',type=int,default=1)
    acap=sub.add_parser('capture-adaptive48'); acap.add_argument('run'); acap.add_argument('day',type=date.fromisoformat); acap.add_argument('arm'); acap.add_argument('snapshot'); acap.add_argument('--experiment-dir',type=Path)
    cycle=sub.add_parser('run-cycle'); cycle.add_argument('run'); cycle.add_argument('--models',required=True,help='JSON arm-to-worker-model map')
    link=sub.add_parser('link-smard'); link.add_argument('run'); link.add_argument('snapshot')
    report=sub.add_parser('report'); report.add_argument('run'); report.add_argument('output',type=Path); report.add_argument('--cutoff',type=datetime.fromisoformat)
    for name in ('export','backup'):
        p=sub.add_parser(name); p.add_argument('output',type=Path)
    restore=sub.add_parser('restore'); restore.add_argument('export',type=Path)
    request=sub.add_parser('timestamp-request'); request.add_argument('sha256'); request.add_argument('output',type=Path)
    receipt=sub.add_parser('timestamp-verify'); receipt.add_argument('run'); receipt.add_argument('day'); receipt.add_argument('arm'); receipt.add_argument('query',type=Path); receipt.add_argument('response',type=Path); receipt.add_argument('ca',type=Path); receipt.add_argument('ca_sha256'); receipt.add_argument('authority'); receipt.add_argument('--trust-scope',choices=['external_configured','local_test'],required=True)
    args=parser.parse_args(argv)
    from src.operations.paths import workspace
    try:root=workspace(args.workspace,required=False)
    except ValueError as exc:parser.error(str(exc))
    if args.command=='doctor':
        from importlib.metadata import version,PackageNotFoundError
        packages={}
        for name in ['chronogrid','fastapi','scikit-learn','onnxruntime','SQLAlchemy','lightgbm','xgboost','onnx','skl2onnx','mlflow-skinny','statsmodels','pyarrow','streamlit']:
            try:packages[name]=version(name)
            except PackageNotFoundError:packages[name]=None
        print(json.dumps(dict(workspace=str(root) if root else None,journal=str(args.db) if args.db else None,packages=packages,scope='Installed package inventory; not model, security or prospective readiness'),indent=2));return
    if args.command=='timestamp-request':
        print(make_request(args.sha256,args.output)); return
    if args.db is None:
        if root is None:parser.error('Provide --db for a standalone journal, or --workspace for research assets')
        args.db=root/'data/operations/journal.sqlite'
    if hasattr(args,'experiment_dir') and args.experiment_dir is None:
        if root is None:parser.error('Model capture requires --experiment-dir or a research --workspace')
        args.experiment_dir=root/'artifacts/experiments/issuance_v2'
    if args.command=='status':
        from src.operations.status import journal_status
        print(json.dumps(journal_status(args.db,run_id=args.run),indent=2));return
    if args.command in ('verify','export','backup','report','adaptive-state','weather-delivery','timestamp-verify','link-smard','settle-adaptive') and not args.db.is_file():
        parser.error('Journal does not exist; read-only/link/feedback commands never initialize an empty journal')
    if args.command=='serve':
        if not 1024<=args.port<=65535:parser.error('Port must be between 1024 and 65535')
        try:root=workspace(args.workspace)
        except ValueError as exc:parser.error(str(exc))
        import uvicorn
        from src.operations.api import create_app
        uvicorn.run(create_app(args.db,root),host='127.0.0.1',port=args.port);return
    if args.command=='restore':
        print(json.dumps(Journal.restore(args.export,args.db).verify())); return
    store=Journal(args.db)
    if args.command=='register': print(store.register(Run.model_validate_json(args.config.read_bytes())))
    elif args.command=='run-once':
        from src.operations.worker import run_once
        print(json.dumps(run_once(store,args.run,args.arm,args.model,workspace=root)))
    elif args.command=='run-cycle':
        from src.operations.worker import run_cycle
        print(json.dumps(run_cycle(store,args.run,json.loads(args.models),workspace=root),indent=2))
    elif args.command=='import-calibration':
        from src.uncertainty.schemas import CalibrationTable
        raw=args.path.read_bytes(); CalibrationTable.model_validate_json(raw)
        print(store.snapshot(raw,'issuance-v2-static-calibration',store.clock()))
    elif args.command=='register-aci':
        from src.operations.adaptive import AdaptiveConfig,AdaptiveStreams
        print(AdaptiveStreams(store).register(AdaptiveConfig.model_validate_json(args.config.read_bytes())))
    elif args.command=='adaptive-state':
        from src.operations.adaptive import AdaptiveStreams
        print(json.dumps(AdaptiveStreams(store).state(args.run,args.arm),indent=2))
    elif args.command=='settle-adaptive':
        from src.operations.adaptive import AdaptiveStreams
        if not 1<=args.max_origins<=367 or args.day is not None and args.max_origins!=1:raise ValueError('invalid settlement budget')
        rows=[]
        for _ in range(args.max_origins):
            row=AdaptiveStreams(store).settle(args.run,args.arm,args.day);rows.append(row)
            if row['status'] not in ('processed','skipped'):break
        print(json.dumps(rows,indent=2))
    elif args.command=='verify': print(json.dumps(store.verify(),indent=2))
    elif args.command=='fetch-smard': print(capture_smard(store))
    elif args.command=='fetch-weather-run':
        from src.operations.weather_runs import RunRequest,capture_single_run
        print(capture_single_run(store,RunRequest(initialization=args.initialization)))
    elif args.command=='weather-delivery':
        from src.operations.weather_runs import delivery_weather
        result=delivery_weather(store,args.snapshot,args.day,store.clock())
        with args.output.open('x',encoding='utf-8') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(str(args.output))
    elif args.command=='fetch-weather':
        from src.operations.weather import capture_weather
        print(capture_weather(store))
    elif args.command in ('capture-weekly','capture-fallback48','capture-adaptive48'):
        try:
            if args.command=='capture-adaptive48':
                from src.operations.adaptive import capture_adaptive48
                p,h=capture_adaptive48(store,args.run,args.day,args.arm,args.snapshot,args.experiment_dir)
            else:p,h=(capture_weekly(store,args.run,args.day,args.arm,args.snapshot) if args.command=='capture-weekly' else capture_fallback48(store,args.run,args.day,args.arm,args.snapshot,args.experiment_dir))
            print(json.dumps(dict(prediction_sha256=h,record=p.model_dump(mode='json')),indent=2))
        except Exception as exc:
            store.failure(args.run,args.day,args.arm,str(exc)); raise
    elif args.command=='link-smard':
        from src.operations.observations import link_smard
        print(json.dumps(link_smard(store,args.run,args.snapshot)))
    elif args.command=='report':
        result=score_run(store,args.run,args.cutoff or store.clock())
        with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
        print(digest(args.output.read_bytes()))
    elif args.command=='export': print(store.export(args.output))
    elif args.command=='backup': print(json.dumps(store.backup(args.output)))
    elif args.command=='timestamp-verify': print(verify_receipt(store,args.run,args.day,args.arm,args.query,args.response,args.ca,args.ca_sha256,args.authority,args.trust_scope))
if __name__=='__main__': main()
