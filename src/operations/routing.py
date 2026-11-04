"""Shared single-host DDIR hysteresis; serializable across local processes."""
from datetime import datetime,timedelta
from src.evaluation.prospective import utc
from src.operations.journal import digest,encode

class SharedRouting:
    def __init__(self,journal,stream='weather-v2',threshold=2,max_age_hours=28):
        self.journal=journal; self.stream=stream; self.threshold=threshold; self.max_age=timedelta(hours=max_age_hours)
        if threshold<1: raise ValueError('invalid threshold')
    def observe(self,source_time: datetime,payload: dict | None,valid: bool,primary_available=True):
        now=utc(self.journal.clock()); timestamp=utc(source_time)
        content=digest(encode(payload)); raw_healthy=valid and payload is not None and timestamp<=now and now-timestamp<=self.max_age
        with self.journal.transaction() as db:
            rows=db.execute("SELECT payload FROM objects WHERE kind='routing' ORDER BY rowid DESC").fetchall()
            previous=next((__import__('json').loads(r['payload']) for r in rows if __import__('json').loads(r['payload'])['stream']==self.stream),None)
            if previous and previous['threshold']!=self.threshold: raise ValueError('routing threshold changed within stream')
            count=previous['healthy_count'] if previous else 0
            last_time=previous['last_healthy_time'] if previous else None
            last_content=previous['last_healthy_content'] if previous else None
            distinct=last_time is None or timestamp>utc(datetime.fromisoformat(last_time)) and content!=last_content
            if not raw_healthy:
                count=0; state='FALLBACK'; reason='invalid, missing, stale or future telemetry'
            elif distinct:
                count=min(self.threshold,count+1); last_time=timestamp.isoformat();last_content=content
                state='PRIMARY' if count>=self.threshold else 'FALLBACK'; reason='healthy' if state=='PRIMARY' else 'recovery pending distinct healthy updates'
            else:
                state=previous['state'] if previous else 'FALLBACK';reason='duplicate, non-advancing or frozen telemetry; no recovery progress'
            if not primary_available:
                state='FALLBACK';reason='primary model unavailable'
            decision=dict(stream=self.stream,state=state,reason=reason,healthy_count=count,threshold=self.threshold,last_healthy_time=last_time,last_healthy_content=last_content,source_time=timestamp.isoformat(),raw_healthy=raw_healthy)
            seq=db.execute('SELECT count(*) FROM events').fetchone()[0]+1
            self.journal._add(db,f'routing:{seq}','routing',decision)
        return decision
