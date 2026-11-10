"""V2 prospective/replay contracts; preserves the hash-pinned V1 helper."""
from datetime import date
from typing import Literal
from pydantic import Field, model_validator
from chronogrid.evaluation.prospective import PredictionRecord, Frozen, Digest, Identifier

class Forecast(PredictionRecord):
    schema_version: Literal['2.0'] = '2.0'
    experiment_kind: Literal['prospective', 'historical_replay', 'synthetic_fault']
    run_id: Identifier
    arm: Identifier
    feature_policy: str = Field(min_length=1)
    calibration_method: Literal['pooled_marginal_mad', 'daily_sup_norm', 'unavailable']
    calibration_state_sha256: Digest | None
    effective_alpha: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode='after')
    def compatible_intervals(self):
        if self.calibration_method == 'unavailable':
            if self.calibration_state_sha256 is not None or self.effective_alpha is not None:
                raise ValueError('unavailable calibration cannot carry a state or effective alpha')
            if any(x.interval_status != 'unavailable' for x in self.deliveries):
                raise ValueError('uncalibrated forecasts cannot claim interval bounds')
        elif self.calibration_state_sha256 is None or self.effective_alpha is None:
            raise ValueError('calibrated interval requires state hash and alpha')
        return self

class ArmModel(Frozen):
    arm: Identifier
    model_sha256: Digest

class Run(Frozen):
    run_id: Identifier
    experiment_kind: Literal['prospective', 'historical_replay', 'synthetic_fault']
    protocol_sha256: Digest
    start_date: date
    end_date: date
    arms: tuple[Identifier, ...] = Field(min_length=1)
    allowed_model_sha256: tuple[Digest, ...] = Field(min_length=1)
    arm_models: tuple[ArmModel, ...] = ()
    demand_low_mw: float = Field(gt=0)
    demand_high_mw: float = Field(gt=0)
    maturity_days: int = Field(default=7, ge=1, le=90)

    @model_validator(mode='after')
    def bounds(self):
        if not 0 <= (self.end_date-self.start_date).days <= 366:
            raise ValueError('run must cover 1 to 367 local days')
        if not self.arm_models and len(self.allowed_model_sha256)==1:
            object.__setattr__(self,'arm_models',tuple(ArmModel(arm=a,model_sha256=self.allowed_model_sha256[0]) for a in self.arms))
        if self.arm_models and (len(self.arm_models)!=len(self.arms) or {x.arm for x in self.arm_models}!=set(self.arms) or any(x.model_sha256 not in self.allowed_model_sha256 for x in self.arm_models)):
            raise ValueError('arm/model mapping must cover every arm exactly once')
        if len(set(self.arms)) != len(self.arms) or self.demand_low_mw >= self.demand_high_mw:
            raise ValueError('duplicate arms or invalid demand thresholds')
        return self
