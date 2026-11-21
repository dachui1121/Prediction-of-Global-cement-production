import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from cement_bau.runtime import ROOT, input_path, output_dir

CONFIG = json.loads((ROOT / 'cement_bau/history_config.json').read_text(encoding='utf-8'))
_PREPARED = {}


def prepare_shared_history(country, variant):
    scenario = os.environ.get('CEMENT_SCENARIO', 'bau')
    config = CONFIG[scenario][country]
    if variant != config['nonrep']:
        return
    folder = output_dir(country, variant)
    snapshot = folder / 'rep_history_state.npz'
    environment = dict(os.environ, CEMENT_BAU_OUTPUT=str(folder / 'history_source'), MPLBACKEND='Agg')
    with (folder / 'rep_history.log').open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'cement_bau._history_worker', country, str(snapshot)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    _PREPARED[(scenario, country, variant)] = (snapshot, hashlib.sha256(input_path(country).read_bytes()).hexdigest())


def apply_shared_history(country, variant, years, forecast_series, actual):
    scenario = os.environ.get('CEMENT_SCENARIO', 'bau')
    config = CONFIG[scenario][country]
    if variant != config['nonrep']:
        raise ValueError(f'{country}/{variant} is not the configured non-rep model')
    years = np.asarray(years)
    original = np.asarray(forecast_series, dtype=float).copy()
    actual = np.asarray(actual, dtype=float)
    folder = output_dir(country, variant)
    key = (scenario, country, variant)
    if key not in _PREPARED:
        prepare_shared_history(country, variant)
    snapshot, expected_input_hash = _PREPARED.pop(key)
    if hashlib.sha256(input_path(country).read_bytes()).hexdigest() != expected_input_hash:
        raise RuntimeError('Input workbook changed during the model run')
    with np.load(snapshot, allow_pickle=False) as state:
        rep_years, rep_values = state['years'], state['values']
    lookup = {int(y): float(v) for y, v in zip(rep_years, rep_values)}
    history = (years >= 1990) & (years <= 2024)
    if set(years[history].tolist()) != set(range(1990, 2025)):
        raise ValueError('Shared history requires every year from 1990 through 2024')
    combined = original.copy()
    for i in np.flatnonzero(history):
        value = lookup.get(int(years[i]), np.nan)
        if not np.isfinite(value):
            raise ValueError(f'Nonfinite or absent rep fitted value: {country}/{years[i]}')
        combined[i] = value
    future = (years >= 2025) & (years <= 2100)
    unchanged = original[future].tobytes() == combined[future].tobytes()
    if not unchanged:
        raise AssertionError('Shared history changed forecast bytes')
    if original[~history].tobytes() != combined[~history].tobytes():
        raise AssertionError('Values outside the requested historical period changed')
    np.savez(folder / 'shared_history_state.npz', years=years, forecast_original=original, combined=combined, rep_years=rep_years, rep_values=rep_values)
    fitted = combined[history]
    observed = actual[history]
    valid = np.isfinite(fitted) & np.isfinite(observed)
    residual = observed[valid] - fitted[valid]
    variance = np.sum((observed[valid] - np.mean(observed[valid]))**2)
    nonzero = valid & (np.abs(observed) > 1e-12)
    metrics = {
        'period': '1990-2024', 'N': int(valid.sum()),
        'MAE': float(np.mean(np.abs(residual))),
        'RMSE': float(np.sqrt(np.mean(residual**2))),
        'Bias': float(np.mean(fitted[valid] - observed[valid])),
        'MAPE_percent': float(np.mean(np.abs((fitted[nonzero]-observed[nonzero])/observed[nonzero])) * 100),
        'R2': float(1-np.sum(residual**2)/variance) if variance > 0 else None,
        'DW': float(np.sum(np.diff(residual)**2)/np.sum(residual**2)) if np.sum(residual**2)>0 else None,
    }
    audit = {
        'scenario':scenario, 'country':country, 'nonrep_variant':variant, 'rep_history_variant':config['rep'],
        'history_years':int(history.sum()), 'forecast_years':int(future.sum()),
        'forecast_bitwise_unchanged':unchanged,
        'forecast_bytes_sha256':hashlib.sha256(original[future].tobytes()).hexdigest(),
        'rep_model_sha256':hashlib.sha256((ROOT / (config['rep_module'].replace('.', '/') + '.py')).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(input_path(country).read_bytes()).hexdigest(),
        'history_metrics':metrics,
        'forecast_state':'Original non-rep calibration and bridge state retained; common fitted history is not fed back into forecasting.',
    }
    (folder / 'shared_history_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(f'Shared rep history: {country}, 1990-2024; original forecast bytes unchanged.')
    return combined
