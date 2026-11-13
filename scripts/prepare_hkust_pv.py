"""Prepare HKUST sensor/PV tables and indexed 16-step forecasting samples.

No downloads, weather imputation, synthetic power, or model training.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/public_datasets/hkust_rooftop_pv_2021_2023/extracted/Dataset'
OUTPUT = ROOT / 'data/processed/hkust_rooftop_pv'
WEATHER = ['temperature_c', 'pressure_hpa', 'humidity_percent', 'irradiance_w_m2']
FIELDS = WEATHER + ['power_mw']
BOUNDS = dict(zip(WEATHER, [(-50, 70), (300, 1100), (0, 100), (0, 1500)]))
HISTORY = 16
TRAIN_END = pd.Timestamp('2022-07-01')
VAL_END = pd.Timestamp('2023-01-01')
MET = {
    'Temperature': [('temperature_c', (-50, 70))],
    'Sea Level Pressure': [('pressure_hpa', (300, 1100))],
    'Relative Humidity': [('humidity_percent', (0, 100))],
    'Irradiance': [('irradiance_w_m2', (0, 1500))],
    'Rainfall': [('rainfall_source_mm', (0, None))],
    'Visibility': [('visibility_km', (0, None))],
    'Wind': [('wind_speed_m_s', (0, 100)), ('wind_direction_deg', (0, 360))],
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, data):
    data.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.12g',
                compression='gzip' if str(path).endswith('.gz') else None)


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def canonical(name):
    return re.sub('[^a-z0-9]', '', name.lower())


def read_table(path, inventory, quarantine, out):
    frame = pd.read_excel(path, engine='openpyxl') if path.suffix == '.xlsx' else pd.read_csv(path, low_memory=False)
    if frame.columns[0] != 'Time':
        raise ValueError(f'Unexpected timestamp header: {path}')
    raw_times = frame.pop('Time')
    dates = pd.to_datetime(raw_times, format='mixed', errors='coerce').astype('datetime64[ns]')
    numeric = frame.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    numeric.index = pd.DatetimeIndex(dates, name='timestamp')
    invalid_time = dates.isna().to_numpy()
    duplicate = dates.duplicated(keep=False).to_numpy() & ~invalid_time
    # Every duplicate row is quarantined; conflicting rows are never averaged.
    bad = invalid_time | duplicate
    if bad.any():
        rejected = numeric.iloc[np.flatnonzero(bad)].reset_index()
        rejected.insert(1, 'raw_timestamp', raw_times.iloc[np.flatnonzero(bad)].to_numpy())
        rejected['reason'] = np.where(invalid_time[bad], 'invalid_timestamp', 'duplicate_timestamp')
        name = f'{len(quarantine):03d}_{path.stem}.csv.gz'
        write_csv(out / 'quarantine' / name, rejected)
        quarantine.append({'source': path.relative_to(SOURCE).as_posix(), 'file': name, 'rows': int(bad.sum())})
    result = numeric.iloc[np.flatnonzero(~bad)].sort_index()
    profile = {
        'path': path.relative_to(SOURCE).as_posix(), 'sha256': sha256(path),
        'raw_rows': len(frame), 'retained_rows': len(result),
        'invalid_time_rows': int(invalid_time.sum()), 'duplicate_rows': int(duplicate.sum()),
        'columns': list(frame.columns),
        'missing_or_nonfinite_by_column': {c: int(numeric[c].isna().sum()) for c in numeric},
        'numeric_parse_failures_by_column': {
            c: int((frame[c].notna().to_numpy() & numeric[c].isna().to_numpy()).sum()) for c in frame},
        'first_timestamp': str(result.index.min()), 'last_timestamp': str(result.index.max()),
        'dominant_interval_seconds': int(result.index.to_series().diff().dt.total_seconds().mode().iloc[0]) if len(result) > 1 else None,
    }
    inventory.append(profile)
    return result, profile


def clean_bounds(series, bounds):
    low, high = bounds
    invalid = series.notna() & ((series < low) | ((series > high) if high is not None else False))
    return series.mask(invalid), int(invalid.sum())


def trailing_weather(frame):
    """At T, use only minutes in (T-15min,T]; require all 15 observations."""
    values = frame.resample('15min', closed='right', label='right', origin='start_day').mean()
    counts = frame.resample('15min', closed='right', label='right', origin='start_day').count()
    if 'wind_direction_deg' in frame:
        angle = np.deg2rad(frame.wind_direction_deg)
        sine = np.sin(angle).resample('15min', closed='right', label='right', origin='start_day').mean()
        cosine = np.cos(angle).resample('15min', closed='right', label='right', origin='start_day').mean()
        direction = np.rad2deg(np.arctan2(sine, cosine)) % 360
        values['wind_direction_deg'] = direction.where(np.hypot(sine, cosine) > 1e-8)
    return values.where(counts == 15), counts


def parse_stations():
    path = next((SOURCE / 'Metadata').glob('*.ttl'))
    text = path.read_text(encoding='utf-8')
    blocks = re.findall(r'^pvsystem:(\S+) a brick:PV_Generation_System\s*;(.*?)(?=\n\npvsystem:|\Z)', text, re.M | re.S)
    registry = {}
    for name, block in blocks:
        def value(pattern):
            match = re.search(pattern, block, re.S)
            if match is None:
                raise ValueError(f'Missing metadata for {name}: {pattern}')
            return float(match.group(1))
        key = canonical(name)
        registry[key] = {
            'metadata_entity': name, 'capacity_mw': value(r'ext:ratedPowerOutput\s*\[.*?brick:value\s+([\d.]+)') / 1000,
            'source_latitude': value(r'brick:latitude\s+([\d.]+)'),
            'source_longitude': value(r'brick:longitude\s+([\d.]+)'),
            'source_altitude_m': value(r'ext:altitude\s*\[.*?brick:value\s+([\d.]+)'),
            'capacity_status': 'publisher_TTL_ratedPowerOutput_kW_not_independently_verified',
            'campus_reference_latitude': 22.3363, 'campus_reference_longitude': 114.2634,
            'coordinates_verified': False, 'timezone_verified': False,
            'timezone_assumption': 'UTC+08:00',
            'timezone_basis': 'Hong Kong local-time hypothesis; noon irradiance supports it, source timestamps have no timezone',
            'pressure_reference': 'sea_level', 'metadata_verified': False,
        }
    if len(registry) != 60:
        raise ValueError(f'Expected 60 metadata stations, got {len(registry)}')
    return registry, path


def split_codes(dates):
    return np.where(dates < TRAIN_END, 0, np.where(dates < VAL_END, 1, 2)).astype(np.int8)


def target_rows(frame, capacity):
    """All history rows and target must be in one calendar split."""
    dates = pd.DatetimeIndex(frame.timestamp).as_unit('ns')
    values = frame[FIELDS].to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    splits = split_codes(dates)
    targets = np.arange(HISTORY, len(frame), dtype=np.int64)
    if not len(targets):
        return {s: np.array([], dtype=np.int64) for s in ['train', 'val', 'test']}
    prefix = np.r_[0, np.cumsum(~valid)]
    history_valid = prefix[targets] == prefix[targets - HISTORY]
    y = values[targets, -1]
    target_valid = np.isfinite(y) & (y >= 0) & (y <= capacity)
    contiguous = (dates.asi8[targets] - dates.asi8[targets - HISTORY]) == HISTORY * 900 * 10**9
    same_split = splits[targets] == splits[targets - HISTORY]
    keep = history_valid & target_valid & contiguous & same_split
    return {s: targets[keep & (splits[targets] == i)] for i, s in enumerate(['train', 'val', 'test'])}


def prepare(out):
    if (out / 'data_report.json').exists():
        raise ValueError('Completed output already exists; choose --output for a new build')
    for folder in ['', 'weather', 'cleaned', 'hourly', 'auxiliary/inverters', 'datasets', 'quarantine', 'examples']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    write_json(out / 'build_status.json', {'completed': False, 'started_at': datetime.now(timezone.utc).isoformat()})
    inventory, quarantine, reports = [], [], []
    met_root = SOURCE / 'Time series dataset/Meteorological dataset'
    met_parts = []
    weather_reports = []
    for folder, specs in MET.items():
        parts = []
        for path in sorted(p for p in (met_root / folder).iterdir() if p.suffix in ('.csv', '.xlsx')):
            raw, profile = read_table(path, inventory, quarantine, out)
            if raw.shape[1] != len(specs):
                raise ValueError(f'Unexpected columns: {path}')
            raw.columns = [x[0] for x in specs]
            for column, bounds in specs:
                raw[column], invalid = clean_bounds(raw[column], bounds)
                weather_reports.append({'source_file': path.name, 'field': column, 'range_rejected': invalid})
            parts.append(raw)
        combined = pd.concat(parts).sort_index()
        if combined.index.duplicated().any():
            raise ValueError(f'Cross-file weather duplicates: {folder}')
        if ((combined.index.second != 0) | (combined.index.microsecond != 0)).any():
            raise ValueError('Weather timestamps not on minute boundaries')
        met_parts.append(combined)
    minute = pd.concat(met_parts, axis=1, sort=True).sort_index()
    # Keep source wall clock; UTC conversion is an explicitly provisional column.
    minute.index.name = 'timestamp'
    weather, counts = trailing_weather(minute)
    weather.index.name = counts.index.name = 'timestamp'
    write_csv(out / 'weather/weather_1min_cleaned.csv.gz', minute.reset_index())
    write_csv(out / 'weather/weather_15min_trailing.csv', weather.reset_index())
    write_csv(out / 'weather/weather_15min_counts.csv', counts.add_suffix('_valid_count').reset_index())
    write_json(out / 'weather/range_rejections.json', weather_reports)
    # Diagnose source offsets/drift rather than force nighttime irradiance to zero.
    daylight = minute.groupby(minute.index.hour)['irradiance_w_m2'].mean()
    diagnostic = {'hourly_mean_irradiance': {str(k): float(v) for k, v in daylight.items()},
                  'nighttime_irradiance_not_corrected': True,
                  'sea_level_pressure_not_converted_to_station_pressure': True}
    write_json(out / 'weather/diagnostics.json', diagnostic)
    print('Weather: all 21 sources cleaned; trailing 15-minute table written', flush=True)
    metadata, ttl_path = parse_stations()
    sites = []
    used_entities = set()
    pv_root = SOURCE / 'Time series dataset/PV generation dataset'
    examples = {}
    for group in ['PV stations with panel level optimizer', 'PV stations without panel level optimizer']:
        interval = 15 if group == 'PV stations with panel level optimizer' else 60
        for path in sorted((pv_root / group / 'Site level dataset').glob('*.csv')):
            raw, profile = read_table(path, inventory, quarantine, out)
            key = canonical(path.stem)
            if key == 'indoorsportscentre':
                key = 'indoorsportcenter'
            meta = dict(metadata[key])
            if key in used_entities:
                raise ValueError('Duplicate station metadata mapping')
            used_entities.add(key)
            site = 'hkust_' + re.sub(r'[^a-z0-9]+', '_', path.stem.lower()).strip('_')
            cap = meta['capacity_mw']
            power_source = pd.to_numeric(raw['power(W)'], errors='coerce') / 1_000_000
            power, rejected_power = clean_bounds(power_source, (0, cap))
            energy, rejected_energy = clean_bounds(raw['generation(kWh)'], (0, None))
            data = pd.DataFrame({'power_mw': power, 'generation_kwh': energy,
                                 'power_missing': power_source.isna(),
                                 'power_out_of_range': power_source.notna() & power.isna()})
            on_grid = ((data.index.minute % interval) == 0) & (data.index.second == 0)
            if (~on_grid).any():
                rejected = data.loc[~on_grid].reset_index()
                name = site + '_off_grid.csv.gz'
                write_csv(out / 'quarantine' / name, rejected)
                quarantine.append({'source': path.relative_to(SOURCE).as_posix(), 'file': name, 'rows': len(rejected)})
            data = data.loc[on_grid]
            if data.empty:
                raise ValueError('Empty station')
            original_index = data.index
            grid = pd.date_range(data.index.min(), data.index.max(), freq=f'{interval}min', name='timestamp').as_unit('ns')
            data = data.reindex(grid)
            data['timestamp_missing_from_source'] = ~data.index.isin(original_index)
            # Hourly sources stay hourly; attach only weather observed by that time.
            data = data.join(weather, how='left')
            data['weather_valid'] = data[WEATHER].notna().all(axis=1)
            data['power_valid'] = data.power_mw.notna()
            data['row_valid'] = data.weather_valid & data.power_valid
            data['split'] = np.array(['train', 'val', 'test'])[split_codes(data.index)]
            frame = data.reset_index()
            frame.insert(0, 'station_id', site)
            frame.insert(2, 'timestamp_assumed_utc', (pd.DatetimeIndex(frame.timestamp) - pd.Timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%SZ'))
            folder = 'cleaned' if interval == 15 else 'hourly'
            write_csv(out / folder / f'{site}.csv', frame)
            meta.update({'site_id': site, 'station_name': path.stem, 'interval_minutes': interval,
                         'source_file': path.relative_to(SOURCE).as_posix(), 'source_sha256': profile['sha256'],
                         'normalized_file': f'{folder}/{site}.csv', 'selected_for_15min_samples': interval == 15})
            counts_sample = {'train': 0, 'val': 0, 'test': 0}
            train_rows = (frame.split == 'train') & frame.row_valid
            eligible = interval == 15 and train_rows.sum() >= HISTORY + 1
            meta['selected_for_15min_samples'] = bool(eligible)
            meta['sample_exclusion_reason'] = ('native_hourly_source' if interval != 15 else
                                               'insufficient_complete_training_period_rows' if not eligible else None)
            if eligible:
                ids = target_rows(frame, cap)
                mean = frame.loc[train_rows, WEATHER].mean().to_numpy(dtype=np.float64)
                std = frame.loc[train_rows, WEATHER].std(ddof=0).to_numpy(dtype=np.float64)
                std = np.where(std < 1e-8, 1.0, std)
                scaled = frame[FIELDS].to_numpy(dtype=np.float64)
                scaled[:, :4] = (scaled[:, :4] - mean) / std
                scaled[:, 4] /= cap
                schema = {
                    **meta, 'raw_fields': FIELDS, 'model_fields': WEATHER + ['power_pu'],
                    'history_len': HISTORY, 'horizon_minutes': 15, 'weather_mean': mean.tolist(), 'weather_std': std.tolist(),
                    'scaler_fit': 'finite complete training-period station rows only; no validation/test values',
                    'bounds': {**BOUNDS, 'power_mw': [0, cap]},
                    'split_boundaries': {'train_end_exclusive': str(TRAIN_END), 'val_end_exclusive': str(VAL_END)},
                    'sample_format': 'indexed_timeline_v1',
                    'sample_usage': 'For target row j use model_features[j-16:j], y=raw_features[j,4]/capacity_mw',
                    'timezone': 'UTC+08:00 assumed, unverified; original wall-clock timestamp retained',
                    'training_missing_policy': 'reject incomplete histories or invalid target power; no imputation',
                    'meteorology_aggregation': 'mean over (T-15min,T], all 15 valid minute observations required per field',
                    'power_timestamp_semantics': 'source label retained; interval start/end not documented',
                    'research_samples_available': True, 'full_fuxi_training_ready': False,
                    'hardware_deployment_ready': False,
                }
                payload = {'raw_features': frame[FIELDS].to_numpy(dtype=np.float32),
                           'model_features': scaled.astype(np.float32),
                           'timestamp_local_ns': pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8,
                           'row_valid': frame.row_valid.to_numpy(dtype=bool),
                           'split_code': split_codes(pd.DatetimeIndex(frame.timestamp))}
                for split, indices in ids.items():
                    payload['target_row_' + split] = indices
                    payload['y_' + split] = frame.power_mw.to_numpy()[indices].astype(np.float32) / cap
                    counts_sample[split] = len(indices)
                    if len(indices) and split not in examples:
                        # Prefer a daytime target so the example is informative.
                        positive = indices[frame.power_mw.to_numpy()[indices] > cap * 0.1]
                        target = int(positive[0] if len(positive) else indices[0])
                        history = []
                        for row_id in range(target - HISTORY, target):
                            row = {f: float(frame.iloc[row_id][f]) for f in FIELDS}
                            row['timestamp'] = frame.iloc[row_id].timestamp.isoformat() + '+08:00'
                            history.append(row)
                        request = {'station_id': site, 'capacity_mw': cap, 'interval_minutes': 15,
                                   'as_of': history[-1]['timestamp'], 'history': history}
                        examples[split] = {'request': request, 'target_timestamp': frame.iloc[target].timestamp.isoformat() + '+08:00',
                                           'target_power_mw': float(frame.iloc[target].power_mw),
                                           'split': split, 'target_row': target,
                                           'schema': schema, 'timezone_is_assumed': True,
                                           'pressure_reference': 'sea_level', 'deployable': False}
                np.savez_compressed(out / 'datasets' / f'{site}.npz', **payload)
                write_json(out / 'datasets' / f'{site}_schema.json', schema)
            sites.append(meta)
            reports.append({'site_id': site, 'interval_minutes': interval, 'source_rows': profile['raw_rows'],
                            'timeline_rows': len(frame), 'missing_timestamps': int(data.timestamp_missing_from_source.sum()),
                            'power_range_rejected': rejected_power, 'energy_range_rejected': rejected_energy,
                            'off_grid_rejected': int((~on_grid).sum()),
                            'complete_rows': int(frame.row_valid.sum()), 'samples': counts_sample})
            print(f'{site}: {interval}min, {len(frame)} rows, samples={counts_sample}', flush=True)
    if len(used_entities) != 60:
        raise ValueError('Not all 60 station metadata entities mapped')
    inverter_files = sorted((pv_root / 'PV stations with panel level optimizer/Inverter level dataset').glob('*.csv'))
    for i, path in enumerate(inverter_files):
        raw, profile = read_table(path, inventory, quarantine, out)
        # Keep electrical columns/units exactly as supplied. No invented phase/unit corrections.
        raw['negative_total_active_power_flag'] = raw['totalActivePower(W)'] < 0
        write_csv(out / 'auxiliary/inverters' / (path.stem + '.csv.gz'), raw.reset_index())
        profile['role'] = 'auxiliary_inverter_native_resolution_not_independent_station_label'
        if i % 10 == 0:
            print(f'Inverters: {i+1}/{len(inverter_files)} normalized', flush=True)
    expected_files = sorted(p for p in SOURCE.rglob('*') if p.suffix in ('.csv', '.xlsx'))
    if sorted(x['path'] for x in inventory) != sorted(p.relative_to(SOURCE).as_posix() for p in expected_files):
        raise ValueError('Some source tables were omitted or duplicated')
    write_json(out / 'source_inventory.json', inventory)
    write_json(out / 'quarantine_manifest.json', quarantine)
    write_json(out / 'station_registry.json', sites)
    write_csv(out / 'station_registry.csv', pd.DataFrame(sites))
    write_json(out / 'station_quality.json', reports)
    write_csv(out / 'station_quality.csv', pd.json_normalize(reports, sep='_'))
    for split, example in examples.items():
        write_json(out / 'examples' / f'{split}_sample.json', example)
    report = {
        'completed': True, 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_doi': '10.5061/dryad.m37pvmd99', 'archive_release': '2024-04-02',
        'raw_source_table_count': len(inventory),
        'raw_source_csv_count': sum(x['path'].endswith('.csv') for x in inventory),
        'raw_source_xlsx_count': sum(x['path'].endswith('.xlsx') for x in inventory), 'source_rows_by_role': {
            'meteorology': sum(x['raw_rows'] for x in inventory if 'Meteorological dataset/' in x['path']),
            'station_pv': sum(x['raw_rows'] for x in inventory if 'Site level dataset/' in x['path']),
            'inverter': sum(x['raw_rows'] for x in inventory if 'Inverter level dataset/' in x['path'])},
        'station_count': len(sites), 'stations_15min': 37, 'stations_hourly': 23,
        'stations_with_indexed_samples': sum(s['selected_for_15min_samples'] for s in sites),
        'inverter_count': len(inverter_files),
        'cleaned_station_rows': sum(x['timeline_rows'] for x in reports),
        'samples': {s: sum(x['samples'][s] for x in reports) for s in ['train', 'val', 'test']},
        'power_range_rejected': sum(x['power_range_rejected'] for x in reports),
        'weather_range_rejected': sum(x['range_rejected'] for x in weather_reports),
        'training_imputed_values': 0, 'metadata_ttl_sha256': sha256(ttl_path),
        'full_fuxi_training_ready': False, 'hardware_deployment_ready': False,
        'remaining_requirements': ['Verify source timestamp timezone and power interval labeling',
                                   'Reconcile station TTL coordinates with publisher campus reference',
                                   'Resolve sea-level vs hardware station-pressure definition',
                                   'Pair contemporaneous full global FuXi weather inputs'],
        'quality_notes': ['Nighttime irradiance offsets and year-to-year drift retained for review; no invented zero correction',
                          'Only 37 native 15-minute station sources yield primary samples; hourly sources not upsampled',
                          'Common calendar splits; all 16 history observations and target in the same split',
                          'SLP pressure is retained and explicitly marked; it is not measured station pressure'],
    }
    write_json(out / 'data_report.json', report)
    write_json(out / 'build_status.json', {'completed': True, 'report': 'data_report.json'})
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.output)
