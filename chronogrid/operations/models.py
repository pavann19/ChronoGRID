"""Strict frozen-model adapter: no missing-feature defaults or calibration mismatch."""
from pathlib import Path
import hashlib,json
import numpy as np
from chronogrid.serving.onnx_engine import OnnxInferenceEngine
from chronogrid.uncertainty.schemas import CalibrationTable
from chronogrid.uncertainty.conformal import ConformalCalibrationPool

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class CheckedOnnx:
    def __init__(self,model,metadata,expected_model_sha256,expected_metadata_sha256):
        if sha(model)!=expected_model_sha256 or sha(metadata)!=expected_metadata_sha256:
            raise ValueError('model/metadata checksum mismatch')
        self.sha256=expected_model_sha256
        self.engine=OnnxInferenceEngine(str(model),str(metadata))
        if not self.engine.feature_names or len(set(self.engine.feature_names))!=len(self.engine.feature_names):
            raise ValueError('invalid feature schema')
    def predict(self,rows):
        required=set(self.engine.feature_names)
        if not rows or any(set(row)!=required for row in rows): raise ValueError('exact model feature schema required; no zero filling')
        matrix=np.asarray([[row[c] for c in self.engine.feature_names] for row in rows],dtype=np.float32)
        if not np.isfinite(matrix).all(): raise ValueError('non-finite model input')
        pred=self.engine.predict(matrix)
        if not np.isfinite(pred).all(): raise ValueError('non-finite model output')
        return pred

def static_intervals(predictions,hours,table: CalibrationTable,model_sha256,method,coverage,feature_policy,calibrated_feature_policy):
    if table.model_sha256!=model_sha256: raise ValueError('calibration/model identity mismatch')
    if feature_policy!=calibrated_feature_policy: raise ValueError('calibration feature policy mismatch')
    if method not in ('pooled_marginal_mad','daily_sup_norm') or coverage not in (.9,.95): raise ValueError('unregistered calibration target')
    pred=np.asarray(predictions,dtype=float); h=np.asarray(hours)
    if len(pred) not in (23,24,25) or len(pred)!=len(h) or not np.isfinite(pred).all() or not np.isin(h,np.arange(24)).all(): raise ValueError('invalid daily vector')
    pool=ConformalCalibrationPool.from_calibration_table(table)
    q=(pool.get_simultaneous_quantile(1-coverage) if method=='daily_sup_norm' else pool.get_marginal_quantile(1-coverage))[0]
    if not np.isfinite(q): return dict(status='unbounded',lower=None,upper=None,guarantee=False)
    lo,hi=pool.normalizer.scale_intervals(pred,h,q)
    return dict(status='bounded',lower=lo.tolist(),upper=hi.tolist(),guarantee=False)
