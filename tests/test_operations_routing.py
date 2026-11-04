from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
import pytest
from src.operations.journal import Journal
from src.operations.routing import SharedRouting

def test_shared_routing_recovers_once_with_distinct_telemetry(tmp_path):
    now=datetime(2027,1,1,12,tzinfo=timezone.utc);j=Journal(tmp_path/'j.sqlite',lambda:now)
    a=SharedRouting(j); assert a.observe(now-timedelta(hours=2),{'temperature':10},False)['state']=='FALLBACK'
    def observe(_): return SharedRouting(Journal(j.path,lambda:now)).observe(now-timedelta(hours=1),{'temperature':11},True)
    with ThreadPoolExecutor(max_workers=6) as pool: decisions=list(pool.map(observe,range(12)))
    assert all(x['healthy_count']==1 and x['state']=='FALLBACK' for x in decisions)
    b=SharedRouting(Journal(j.path,lambda:now))
    assert b.observe(now,{'temperature':12},True)['state']=='PRIMARY'
    assert b.observe(now+timedelta(hours=1),{'temperature':13},True)['state']=='FALLBACK'
    assert b.observe(now,{'temperature':12},True)['state']=='FALLBACK'
    assert b.observe(now,{'temperature':12},True,primary_available=False)['reason']=='primary model unavailable'
    assert j.verify()['objects']==17
    with pytest.raises(ValueError): SharedRouting(j,threshold=3).observe(now,{},True)
