"""Transactional, append-only single-host evidence journal with portable exports."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import base64
import hashlib
import json
import sqlite3
from typing import Callable
from src.evaluation.prospective import ActualObservation, canonical, utc
from src.operations.contracts import Forecast, Run

class Conflict(ValueError): pass
class IntegrityFailure(ValueError): pass

def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def digest(value: bytes): return hashlib.sha256(value).hexdigest()
def now_utc(): return datetime.now(timezone.utc)

def wal_reset_safe(version):
    return version >= (3,51,3) or (3,44,6) <= version < (3,45,0) or (3,50,7) <= version < (3,51,0)

class Journal:
    def __init__(self, path: Path, clock: Callable = now_utc):
        self.path=Path(path); self.clock=clock
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            # Upstream WAL-reset fix: 3.51.3+, or documented 3.44.6/3.50.7 backports.
            mode='WAL' if wal_reset_safe(sqlite3.sqlite_version_info) else 'DELETE'
            self.journal_mode=db.execute('PRAGMA journal_mode='+mode).fetchone()[0]
            db.executescript('''
            CREATE TABLE IF NOT EXISTS objects (
              key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
              sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY, object_key TEXT UNIQUE NOT NULL REFERENCES objects(key),
              previous TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_objects_update BEFORE UPDATE ON objects BEGIN SELECT RAISE(ABORT,'immutable objects'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_objects_delete BEFORE DELETE ON objects BEGIN SELECT RAISE(ABORT,'immutable objects'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_events_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'immutable events'); END;
            CREATE TRIGGER IF NOT EXISTS immutable_events_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'immutable events'); END;
            ''')

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.path,timeout=15,isolation_level=None)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON'); db.execute('PRAGMA synchronous=FULL'); db.execute('PRAGMA busy_timeout=15000')
        try: yield db
        finally: db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK'); raise

    def _add(self,db,key,kind,payload,recorded_at=None):
        raw=encode(payload); h=digest(raw); t=recorded_at or utc(self.clock()).isoformat()
        last=db.execute('SELECT seq,sha256 FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        seq=last['seq']+1 if last else 1; prev=last['sha256'] if last else '0'*64
        event_hash=digest(encode(dict(seq=seq,key=key,kind=kind,payload_sha256=h,recorded_at=t,previous=prev)))
        try:
            db.execute('INSERT INTO objects VALUES (?,?,?,?,?)',(key,kind,raw.decode(),h,t))
            db.execute('INSERT INTO events VALUES (?,?,?,?)',(seq,key,prev,event_hash))
        except sqlite3.IntegrityError as exc: raise Conflict('duplicate evidence identity') from exc
        return h

    def _get(self,db,key):
        row=db.execute('SELECT * FROM objects WHERE key=?',(key,)).fetchone()
        if row is None: raise KeyError(key)
        if digest(row['payload'].encode()) != row['sha256']: raise IntegrityFailure('object hash mismatch')
        return json.loads(row['payload']),row

    def get(self,key):
        with self.connection() as db: return self._get(db,key)[0]

    def rows(self,kind):
        with self.connection() as db:
            return [dict(row) | {'payload':json.loads(row['payload'])} for row in db.execute('SELECT o.*,e.seq FROM objects o JOIN events e ON e.object_key=o.key WHERE o.kind=? ORDER BY e.seq',(kind,))]

    def register(self,run: Run):
        run=Run.model_validate_json(canonical(run))
        if not run.arm_models: raise ValueError('new multi-model runs require explicit per-arm model identities')
        data=json.loads(canonical(run))
        with self.transaction() as db:
            h=self._add(db,'run:'+run.run_id,'run',data)
            day=run.start_date
            while day<=run.end_date:
                for arm in run.arms:
                    self._add(db,f'slot:{run.run_id}:{day}:{arm}','slot',dict(run_id=run.run_id,delivery_date=str(day),arm=arm))
                day+=timedelta(days=1)
        return h

    def snapshot(self,content: bytes,source: str,available_at: datetime,media_type='application/json'):
        now=utc(self.clock()); available=utc(available_at)
        if available>now: raise ValueError('snapshot availability is in the future')
        h=digest(content)
        with self.transaction() as db:
            key='snapshot:'+h
            existing=db.execute('SELECT key FROM objects WHERE key=?',(key,)).fetchone()
            if existing:
                previous,_=self._get(db,key)
                if previous['source']!=source: raise Conflict('identical source bytes presented under another source identity')
                return h # preserve first capture, never overwrite provenance
            self._add(db,key,'snapshot',dict(content_sha256=h,source=source,available_at=available.isoformat(),captured_at=now.isoformat(),media_type=media_type,content_base64=base64.b64encode(content).decode()))
        return h

    def snapshot_bytes(self,h):
        item=self.get('snapshot:'+h); raw=base64.b64decode(item['content_base64'],validate=True)
        if digest(raw)!=h: raise IntegrityFailure('snapshot content mismatch')
        return raw

    def commit(self,prediction: Forecast):
        p=Forecast.model_validate_json(canonical(prediction)); now=utc(self.clock())
        if utc(p.issued_at)>now: raise ValueError('future issuance')
        with self.transaction() as db:
            data,meta=self._get(db,'run:'+p.run_id); run=Run.model_validate(data)
            if not run.arm_models or next((x.model_sha256 for x in run.arm_models if x.arm==p.arm),None)!=p.model_sha256:
                raise ValueError('arm/model identity mismatch; legacy unpinned runs cannot receive new forecasts')
            slot=f'{p.run_id}:{p.delivery_date}:{p.arm}'
            self._get(db,'slot:'+slot)
            if p.experiment_kind != run.experiment_kind or p.protocol_sha256 != run.protocol_sha256 or p.model_sha256 not in run.allowed_model_sha256:
                raise ValueError('run protocol/model/kind mismatch')
            if p.experiment_kind=='prospective':
                if not utc(datetime.fromisoformat(meta['recorded_at']))<=utc(p.issued_at)<=now<utc(p.deliveries[0].timestamp):
                    raise ValueError('prospective run must be registered before issuance; commit before delivery')
                from zoneinfo import ZoneInfo
                gate=datetime.combine(p.delivery_date-timedelta(days=1),datetime.min.time(),ZoneInfo('Europe/Berlin')).replace(hour=12)
                if now>utc(gate): raise ValueError('prospective commitment missed day-ahead gate')
            for source in p.inputs:
                snapshot,_=self._get(db,'snapshot:'+source.snapshot_sha256)
                if snapshot['source']!=source.source or utc(datetime.fromisoformat(snapshot['available_at'])) != utc(source.available_at) or utc(datetime.fromisoformat(snapshot['captured_at'])) != utc(source.captured_at):
                    raise ValueError('snapshot provenance mismatch')
                if utc(source.captured_at)>utc(p.issued_at): raise ValueError('snapshot captured after issuance')
            if p.calibration_state_sha256:
                state,_=self._get(db,'snapshot:'+p.calibration_state_sha256)
                content=json.loads(base64.b64decode(state['content_base64']))
                if content.get('model_sha256')!=p.model_sha256:
                    raise ValueError('calibration state belongs to another model')
                if utc(datetime.fromisoformat(state['captured_at']))>utc(p.issued_at):
                    raise ValueError('calibration state captured after issuance')
                from src.operations.adaptive import validate_forecast_basis
                validate_forecast_basis(self,db,p,content)
            return self._add(db,'forecast:'+slot,'forecast',json.loads(canonical(p)))

    def actual(self,run_id,day,arm,actual: ActualObservation):
        a=ActualObservation.model_validate_json(canonical(actual)); now=utc(self.clock())
        with self.transaction() as db:
            payload,row=self._get(db,f'forecast:{run_id}:{day}:{arm}'); p=Forecast.model_validate(payload)
            if row['sha256'] != a.prediction_sha256: raise ValueError('prediction hash mismatch')
            if utc(a.delivery_timestamp) not in {utc(x.timestamp) for x in p.deliveries}: raise ValueError('unlinked actual hour')
            snap,_=self._get(db,'snapshot:'+a.source_snapshot_sha256)
            if snap['source']!=a.source or utc(datetime.fromisoformat(snap['available_at']))!=utc(a.available_at): raise ValueError('actual snapshot provenance mismatch')
            if not utc(a.available_at)<=now: raise ValueError('future actual availability')
            if p.experiment_kind=='prospective' and utc(datetime.fromisoformat(row['recorded_at']))>=utc(a.observation_end): raise ValueError('late prediction cannot become prospective')
            hour=utc(a.delivery_timestamp).isoformat()
            return self._add(db,f'actual:{run_id}:{day}:{arm}:{hour}:{a.revision}','actual',json.loads(canonical(a)))

    def failure(self,run_id,day,arm,reason):
        with self.transaction() as db:
            self._get(db,f'slot:{run_id}:{day}:{arm}')
            seq=db.execute('SELECT count(*) FROM events').fetchone()[0]+1
            return self._add(db,f'failure:{seq}','failure',dict(run_id=run_id,delivery_date=str(day),arm=arm,reason=reason))

    def verify(self):
        with self.connection() as db:
            db.execute('BEGIN')
            result=self._verify_db(db)
            db.execute('COMMIT')
            return result

    def _verify_db(self,db):
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise IntegrityFailure('SQLite integrity')
        previous='0'*64; count=0
        for row in db.execute('SELECT e.*,o.kind,o.payload,o.sha256 AS content_hash,o.recorded_at FROM events e JOIN objects o ON o.key=e.object_key ORDER BY seq'):
            count+=1
            if row['seq']!=count or row['previous']!=previous: raise IntegrityFailure('event chain gap')
            if digest(row['payload'].encode())!=row['content_hash']: raise IntegrityFailure('payload modified')
            h=digest(encode(dict(seq=count,key=row['object_key'],kind=row['kind'],payload_sha256=row['content_hash'],recorded_at=row['recorded_at'],previous=previous)))
            if h!=row['sha256']: raise IntegrityFailure('event hash mismatch')
            obj=json.loads(row['payload'])
            if row['kind']=='snapshot':
                raw=base64.b64decode(obj['content_base64'],validate=True)
                if digest(raw)!=obj['content_sha256'] or row['object_key']!='snapshot:'+digest(raw): raise IntegrityFailure('snapshot mismatch')
            elif row['kind']=='forecast': Forecast.model_validate(obj)
            elif row['kind']=='actual': ActualObservation.model_validate(obj)
            previous=h
        if count!=db.execute('SELECT count(*) FROM objects').fetchone()[0]: raise IntegrityFailure('unjournaled object')
        # Validate referential meaning as well as byte hashes.
        objects={r['key']:(json.loads(r['payload']),r) for r in db.execute('SELECT * FROM objects')}
        forecasts={r['sha256']:obj for obj,r in objects.values() if r['kind']=='forecast'}
        for key,(obj,row) in objects.items():
            if row['kind']=='forecast':
                run=Run.model_validate(objects['run:'+obj['run_id']][0])
                slot=f"slot:{obj['run_id']}:{obj['delivery_date']}:{obj['arm']}"
                if slot not in objects or obj['experiment_kind']!=run.experiment_kind or obj['protocol_sha256']!=run.protocol_sha256 or obj['model_sha256'] not in run.allowed_model_sha256:
                    raise IntegrityFailure('forecast/run registry mismatch')
                if run.arm_models and next((x.model_sha256 for x in run.arm_models if x.arm==obj['arm']),None)!=obj['model_sha256']:
                    raise IntegrityFailure('forecast/arm model mismatch')
                for item in obj['inputs']:
                    if 'snapshot:'+item['snapshot_sha256'] not in objects: raise IntegrityFailure('missing input snapshot')
            elif row['kind']=='actual':
                forecast=forecasts.get(obj['prediction_sha256'])
                if not forecast or utc(datetime.fromisoformat(obj['delivery_timestamp'])) not in {utc(datetime.fromisoformat(d['timestamp'])) for d in forecast['deliveries']}:
                    raise IntegrityFailure('actual linkage mismatch')
                if 'snapshot:'+obj['source_snapshot_sha256'] not in objects: raise IntegrityFailure('missing actual source snapshot')
        from src.operations.adaptive import verify_adaptive_ledger
        verify_adaptive_ledger(self,db,objects)
        return dict(objects=count,head_sha256=previous,sqlite_integrity='ok',independent_timestamp=False,sqlite_version=sqlite3.sqlite_version,journal_mode=self.journal_mode)


    def export(self,path: Path):
        self.verify()
        with self.connection() as db:
            db.execute('BEGIN')
            payload=dict(format='chronogrid-journal-v1',objects=[dict(r) for r in db.execute('SELECT * FROM objects ORDER BY key')],events=[dict(r) for r in db.execute('SELECT * FROM events ORDER BY seq')])
            db.execute('COMMIT')
        raw=encode(payload)
        with Path(path).open('xb') as f: f.write(raw)
        return digest(raw)

    @classmethod
    def restore(cls,export: Path,target: Path):
        if Path(target).exists(): raise Conflict('restore target already exists')
        payload=json.loads(Path(export).read_bytes())
        if payload['format']!='chronogrid-journal-v1': raise ValueError('unknown export format')
        store=cls(target)
        with store.transaction() as db:
            for x in payload['objects']: db.execute('INSERT INTO objects VALUES (:key,:kind,:payload,:sha256,:recorded_at)',x)
            for x in payload['events']: db.execute('INSERT INTO events VALUES (:seq,:object_key,:previous,:sha256)',x)
            # Check before commit; exception rolls back any imported rows.
            previous='0'*64
            for i,x in enumerate(payload['events'],1):
                obj=next((o for o in payload['objects'] if o['key']==x['object_key']),None)
                if not obj or digest(obj['payload'].encode())!=obj['sha256']: raise IntegrityFailure('invalid imported object')
                expected=digest(encode(dict(seq=i,key=obj['key'],kind=obj['kind'],payload_sha256=obj['sha256'],recorded_at=obj['recorded_at'],previous=previous)))
                if x['seq']!=i or x['previous']!=previous or x['sha256']!=expected: raise IntegrityFailure('invalid imported chain')
                previous=expected
            if len(payload['objects'])!=len(payload['events']): raise IntegrityFailure('imported orphan objects')
            store._verify_db(db) # semantic validation before the import transaction can commit
        store.verify(); return store

    def backup(self,target: Path):
        target=Path(target)
        with target.open('xb'): pass
        with self.connection() as source:
            destination=sqlite3.connect(target)
            try: source.backup(destination)
            finally: destination.close()
        return Journal(target).verify()
