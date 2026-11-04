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
    assert client.get('/api/weather/runs').json()['runs']==[]
    calendar=client.get('/api/research/local-calendar-v3').json()
    assert calendar['experiment_kind']=='historical_replay' and calendar['evaluation_observations']==840
    assert calendar['metadata']['feature_policy']=='fixed_gate_load48_local_calendar_v3'
    weather=client.get('/api/research/matched-weather-v4').json()
    assert weather['decision']=='NO_GO_FOR_PROMOTION' and weather['report']['models']['primary']['observations']==817
    assert weather['report']['models']['weekly168']['mae_mw']<weather['report']['models']['primary']['mae_mw']
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
