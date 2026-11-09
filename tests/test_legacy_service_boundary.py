from datetime import datetime,timezone
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from fastapi import Response
from fastapi.testclient import TestClient
from src.serving import app as service
from src.serving.onnx_engine import OnnxInferenceEngine
from src.reliability.router import DualHeadInferenceRouter

ROOT=Path(__file__).resolve().parents[1]

def test_missing_nonfinite_empty_and_misdimensioned_features_rejected():
    e=OnnxInferenceEngine()
    for features in [[{'sin_hour':1}],[],np.empty((0,43)),np.ones((1,42)),np.ones((1,44)),np.ones((1,1,43)),np.full((1,43),np.nan),np.full((1,43),np.inf)]:
        with pytest.raises(ValueError):e.predict(features)
    row={n:1.0 for n in e.feature_names};row['unexpected']=1
    with pytest.raises(ValueError,match='schema'):e.predict([row])

def test_primary_array_projects_fallback_by_names_and_rejects_missing_columns():
    router=DualHeadInferenceRouter()
    from src.features.engineer import FeatureEngineer
    raw=pd.read_parquet(ROOT/'data/raw/genuine_smard_dwd_observations.parquet').iloc[:700]
    frame=FeatureEngineer(min_lag_hours=24,weather_mode='persistence').fit_transform(raw,drop_warmup=True).iloc[:24][router.primary_engine.feature_names]
    expected=router.fallback_engine.predict(frame[router.fallback_engine.feature_names].values)
    prefix=router.fallback_engine.predict(frame.values[:,:len(router.fallback_engine.feature_names)])
    assert np.max(np.abs(prefix-expected))>1 # this data actually reproduces the old positional bug
    array=router._predict_engine(router.fallback_engine,frame[router.primary_engine.feature_names].values)
    mapping=router._predict_engine(router.fallback_engine,frame.to_dict('records'))
    np.testing.assert_allclose(array,expected);np.testing.assert_allclose(mapping,expected)
    with pytest.raises(ValueError,match='Missing'):router._predict_engine(router.fallback_engine,frame.drop(columns=router.fallback_engine.feature_names[0]))
    with pytest.raises(ValueError,match='Duplicate'):router._predict_engine(router.fallback_engine,pd.concat([frame,frame.iloc[:,:1]],axis=1))

def test_legacy_http_retired_by_default_and_optin_requires_token(monkeypatch):
    client=TestClient(service.app)
    monkeypatch.delenv('CHRONOGRID_ENABLE_LEGACY_REPLAY',raising=False)
    expected={'/forecast/day-ahead','/forecast/resilient','/forecast/dqcp','/forecast/dqcp/feedback'}
    assert service.LEGACY_REPLAY_ROUTES==expected
    for route in expected:assert client.post(route,json={}).status_code==410
    monkeypatch.setenv('CHRONOGRID_ENABLE_LEGACY_REPLAY','1');monkeypatch.delenv('CHRONOGRID_LEGACY_REPLAY_TOKEN',raising=False)
    assert client.post('/forecast/day-ahead',json={}).status_code==503
    monkeypatch.setenv('CHRONOGRID_LEGACY_REPLAY_TOKEN','ephemeral-test-credential')
    assert client.post('/forecast/day-ahead',json={}).status_code==401
    assert client.post('/forecast/day-ahead',headers={b'Authorization':b'Bearer \xff'},json={}).status_code==401
    # Authorization reaches request validation; no simulated forecast is generated here.
    allowed=client.post('/forecast/day-ahead',headers={'Authorization':'Bearer ephemeral-test-credential'},json={'horizon_hours':0})
    assert allowed.status_code==422 and allowed.headers['X-ChronoGrid-Evidence-Kind']=='legacy_replay_or_synthetic'
    origin=client.options('/predict',headers={'Origin':'https://unapproved.example','Access-Control-Request-Method':'POST'})
    assert 'access-control-allow-origin' not in origin.headers

def test_predict_labels_caller_features_and_validates_complete_schema(monkeypatch):
    monkeypatch.setenv('CHRONOGRID_DIAGNOSTICS_TOKEN','test-only-diagnostics')
    e=OnnxInferenceEngine();monkeypatch.setattr(service,'ENGINE',e)
    monkeypatch.setattr(service,'CHAMPION_META',{'version':None,'model_type':e.metadata['model_type']})
    client=TestClient(service.app,headers={'Authorization':'Bearer test-only-diagnostics'})
    assert client.post('/predict',json={'features':[]}).status_code==422
    assert client.post('/predict',json={'features':[{'sin_hour':1}]}).status_code==400
    response=client.post('/predict',json={'features':[{n:1 for n in e.feature_names}]})
    assert response.status_code==200 and response.json()['model_version'] is None
    assert response.json()['evidence_kind']=='caller_supplied_features'
    assert response.headers['X-Model-Version']=='unverified'
    assert response.headers['X-ChronoGrid-Evidence-Kind']=='caller_supplied_features'

def test_registry_without_matching_hash_cannot_override_artifact_identity(monkeypatch,tmp_path):
    rec=SimpleNamespace(version=999,model_type='WrongModel',onnx_uri='unverified',metadata_json='{}',promoted_at=None)
    class Query:
        def filter_by(self,**kwargs):return self
        def first(self):return rec
    class Session:
        def query(self,*args):return Query()
    @contextmanager
    def session():yield Session()
    monkeypatch.setattr(service,'get_session',session)
    monkeypatch.setenv('DDIR_STATE_FILE',str(tmp_path/'state.json'))
    service.load_engine()
    assert service.CHAMPION_META['model_type']=='HistGradientBoosting'
    assert service.CHAMPION_META['version'] is None and not service.CHAMPION_META['registry_identity_verified']


def test_model_metadata_missing_duplicate_or_wrong_width_fails_closed(tmp_path):
    import json
    model=ROOT/'artifacts/models/champion_model.onnx'
    metadata=json.loads((ROOT/'artifacts/models/champion_metadata.json').read_text())
    with pytest.raises(FileNotFoundError):OnnxInferenceEngine(str(model),str(tmp_path/'missing.json'))
    for names in [metadata['feature_names'][:-1],metadata['feature_names'][:-1]+[metadata['feature_names'][0]],[]]:
        path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata|{'feature_names':names}))
        with pytest.raises(ValueError):OnnxInferenceEngine(str(model),str(path))


@pytest.mark.parametrize('route', ['/predict', '/model/info', '/metrics', '/openapi.json', '/docs', '/future-diagnostic'])
def test_diagnostics_authentication_fails_closed(monkeypatch, route):
    client = TestClient(service.app)
    method = 'POST' if route == '/predict' else 'GET'
    monkeypatch.delenv('CHRONOGRID_DIAGNOSTICS_TOKEN', raising=False)
    assert client.request(method, route).status_code == 503
    monkeypatch.setenv('CHRONOGRID_DIAGNOSTICS_TOKEN', '   ')
    assert client.request(method, route).status_code == 503
    monkeypatch.setenv('CHRONOGRID_DIAGNOSTICS_TOKEN', 'test-only-diagnostics')
    for supplied in ['', 'Bearer wrong', 'Bearer replay-only']:
        response = client.request(method, route, headers={'Authorization': supplied})
        assert response.status_code == 401
        assert response.headers['WWW-Authenticate'] == 'Bearer'
    authorized = client.request(method, route, headers={'Authorization': 'Bearer test-only-diagnostics'})
    assert authorized.status_code not in {401, 403}
    if route == '/predict':
        assert authorized.status_code == 422
    if route == '/metrics':
        assert authorized.status_code == 200


def test_probes_remain_accessible_and_replay_token_is_separate(monkeypatch):
    monkeypatch.setenv('CHRONOGRID_DIAGNOSTICS_TOKEN', 'diagnostics-only')
    monkeypatch.setenv('CHRONOGRID_ENABLE_LEGACY_REPLAY', '1')
    monkeypatch.setenv('CHRONOGRID_LEGACY_REPLAY_TOKEN', 'replay-only')
    client = TestClient(service.app)
    assert client.get('/health').status_code == 200
    assert client.get('/ready').status_code in {200, 503}
    assert client.post('/forecast/day-ahead', headers={'Authorization': 'Bearer diagnostics-only'}, json={}).status_code == 401
    assert client.get('/metrics', headers={'Authorization': 'Bearer replay-only'}).status_code == 401


@pytest.mark.parametrize('route,status', [('/health', 200), ('/predict', 503),
                                        ('/forecast/day-ahead', 410)])
def test_diagnostic_security_headers_include_auth_and_retired_responses(monkeypatch, route, status):
    monkeypatch.delenv('CHRONOGRID_DIAGNOSTICS_TOKEN', raising=False)
    monkeypatch.delenv('CHRONOGRID_ENABLE_LEGACY_REPLAY', raising=False)
    client = TestClient(service.app)
    response = client.get(route) if route == '/health' else client.post(route, json={})
    assert response.status_code == status
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert response.headers['X-Frame-Options'] == 'DENY'
    assert response.headers['Referrer-Policy'] == 'no-referrer'
