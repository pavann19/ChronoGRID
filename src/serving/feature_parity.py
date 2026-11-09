"""Explicit legacy feature dependencies and a fail-closed prospective release gate."""
from datetime import timedelta
from src.evaluation.prospective import utc
from src.operations.feature_contracts import CALENDAR

WEATHER = ('forecast_temperature_c', 'forecast_wind_speed_ms', 'forecast_solar_radiation_wm2',
           'forecast_humidity_pct', 'forecast_heating_degree_hours', 'forecast_cooling_degree_hours',
           'forecast_temperature_squared', 'forecast_wind_chill_c', 'forecast_temp_solar_interaction')


def legacy_dependencies(names):
    rules = {name: () for name in CALENDAR}
    rules.update({name: (24,) for name in WEATHER})
    for lag in (24, 25, 48, 72, 168):
        rules[f'grid_load_mw_lag_{lag}'] = (lag,)
    for window in (6, 12, 24, 168):
        for statistic in ('mean', 'std', 'min', 'max'):
            rules[f'grid_load_mw_roll_{statistic}_{window}'] = tuple(range(24, 24 + window))
    rules['grid_load_mw_diff_24_48'] = (24, 48)
    rules['grid_load_mw_diff_24_168'] = (24, 168)
    if not names or len(set(names)) != len(names) or any(name not in rules for name in names):
        raise ValueError('unknown or duplicate legacy feature dependency')
    return {name: rules[name] for name in names}


def unavailable_features(names, delivery, issued_at):
    """Observation timestamps label hourly interval starts, not instantaneous values."""
    delivery, issued = utc(delivery), utc(issued_at)
    return [name for name, offsets in legacy_dependencies(names).items()
            if any(delivery - timedelta(hours=offset) + timedelta(hours=1) > issued for offset in offsets)]


def require_legacy_prospective_parity(report):
    """Consumers must not turn a passing legacy regression suite into release approval."""
    if (report.get('decision') != 'PASS' or report.get('unavailable_delivery_hours') != 0
            or report.get('mismatched_cells') != 0
            or report.get('original_training_feature_provenance_verified') is not True):
        raise ValueError('legacy prospective feature parity not established; frozen model remains diagnostic/replay only')
