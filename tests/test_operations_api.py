from pathlib import Path
from fastapi.testclient import TestClient
from src.operations.api import create_app

def test_readonly_console_and_historical_counts(tmp_path):
    client=TestClient(create_app(tmp_path/'journal.sqlite'))
    response=client.get('/api/overview'); assert response.status_code==200
    data=response.json(); assert data['prospective_forecasts']==0 and not data['live_collector_running']
    assert data['historical_metrics'][0]['covered']==814
    assert client.get('/api/evidence').json()['forecasts']==[]
    assert client.get('/api/adaptive').json()['states']==[]
    assert client.get('/api/historical/forecast?origin=36').status_code==422
    assert client.get('/api/historical/forecast?head=unknown').status_code==422
    rows=client.get('/api/historical/forecast?origin=2').json()
    assert rows['evidence_kind']=='historical_reconstruction' and len(rows['rows'])==24
    assert sum(int(x['covered']) for x in client.get('/api/historical/origins').json())==814
    assert client.post('/api/evidence',json={}).status_code==405
    assert client.post('/forecast/dqcp',json={}).status_code==404
    assert client.get('/api/report/missing').status_code==404
    assert 'script-src' in response.headers['Content-Security-Policy']
    assert client.get('/').status_code==200 and client.get('/assets/app.js').status_code==200
    assert client.get('/ready').json()['journal']['sqlite_integrity']=='ok'
