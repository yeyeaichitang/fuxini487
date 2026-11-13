"""Rebuild all historical PV datasets from immutable source workbooks.

Run with output/python312-runtime/python.exe. No model training or downloads.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ['temperature_c', 'pressure_hpa', 'humidity_percent', 'irradiance_w_m2']
FIELDS = WEATHER + ['power_mw']
MODEL_FIELDS = WEATHER + ['power_pu']
BOUNDS = dict(zip(WEATHER, [(-50, 70), (300, 1100), (0, 100), (0, 1500)]))
ALIASES = {
    'Time(year-month-day h:m:s)': 'timestamp',
    'Air temperature  (°C)': 'temperature_c',
    'Atmosphere (hpa)': 'pressure_hpa',
    'Relative humidity (%)': 'humidity_percent',
    'Global horizontal irradiance (W/m2)': 'irradiance_w_m2',
    'Global horicontal irradiance (W/m2)': 'irradiance_w_m2',
    'Total solar irradiance (W/m2)': 'total_solar_irradiance_w_m2',
    'Direct normal irradiance (W/m2)': 'direct_normal_irradiance_w_m2',
    'Power (MW)': 'power_mw',
}
HISTORY = 16
STEP_NS = 15 * 60 * 1_000_000_000
# Same calendar boundaries at every station prevent pooled temporal leakage.
TRAIN_END = pd.Timestamp('2020-01-01')
VAL_END = pd.Timestamp('2020-04-01')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, df):
    df.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.12g')


def load_sources(raw_dir, out):
    sources, sheets = [], []
    paths = sorted(raw_dir.glob('*.xlsm'))
    if not paths:
        raise ValueError('No source workbooks')
    for path in paths:
        match = re.search(r'site (\d+) \(Nominal capacity-(\d+)MW\)', path.name)
        if not match:
            raise ValueError(f'Unknown source naming: {path.name}')
        number, capacity = map(int, match.groups())
        site = ('site_1_a' if path.stem.endswith('(2)') else 'site_1_b') if number == 1 else f'site_{number}'
        print(f'EXTRACT {site}: {path.name}', flush=True)
        matches = []
        with pd.ExcelFile(path, engine='openpyxl') as book:
            for sheet in book.sheet_names:
                headers = [str(c).strip() for c in pd.read_excel(book, sheet_name=sheet, nrows=0).columns]
                pv = any(c.startswith(('Global horizontal irradiance', 'Global horicontal irradiance')) for c in headers)
                sheets.append({'source_file': path.name, 'sheet': sheet, 'columns': headers,
                               'classification': 'pv' if pv else ('empty' if not headers else 'non_pv_excluded')})
                if pv:
                    matches.append(sheet)
            if len(matches) != 1:
                raise ValueError(f'{path.name}: expected one PV sheet, found {matches}')
            df = pd.read_excel(book, sheet_name=matches[0])
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=ALIASES)
        if not set(['timestamp'] + FIELDS).issubset(df.columns):
            raise ValueError(f'Missing required fields: {path.name}')
        # Hash every original PV column, not just selected model features.
        signature = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
        df.insert(0, 'source_row', np.arange(2, len(df) + 2))
        save_csv(out / 'extracted' / f'{site}.csv', df)
        sources.append({'site_id': site, 'source_file': path.name, 'source_sheet': matches[0],
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'pv_content_sha256': signature, 'capacity_mw': capacity,
                        'capacity_status': 'filename_declared_not_independently_verified',
                        'raw_rows': len(df), 'data': df})
    save_json(out / 'sheet_inventory.json', sheets)
    # Prefer the explicitly numbered non-copy source. Retain duplicate provenance.
    for source in sources:
        peers = [s for s in sources if s['pv_content_sha256'] == source['pv_content_sha256']]
        canonical = sorted(peers, key=lambda s: (s['site_id'] == 'site_1_a', s['site_id']))[0]
        source['canonical_site_id'] = canonical['site_id']
        source['duplicate_excluded'] = source is not canonical
        source['duplicate_capacity_conflict'] = len({s['capacity_mw'] for s in peers}) > 1
        # Hash equality is backed by actual cell equality before excluding a file.
        if source['duplicate_excluded']:
            pd.testing.assert_frame_equal(source['data'], canonical['data'])
            print(f'DUPLICATE {source["site_id"]} == {canonical["site_id"]}; exclude from training', flush=True)
    return sources


def clean_source(source, out):
    original = source['data']
    df = original.copy()
    df['timestamp'] = pd.to_datetime(df.timestamp, errors='coerce').astype('datetime64[ns]')
    # Reject every row of a timestamp collision: do not choose an arbitrary observation.
    invalid_time = df.timestamp.isna()
    duplicate_time = df.timestamp.notna() & df.timestamp.duplicated(keep=False)
    off_grid = df.timestamp.notna() & (df.timestamp.astype('int64') % STEP_NS != 0)
    reject = invalid_time | duplicate_time | off_grid
    rejected = original.loc[reject].copy()
    rejected['invalid_timestamp'] = invalid_time[reject]
    rejected['duplicate_timestamp'] = duplicate_time[reject]
    rejected['off_15min_grid'] = off_grid[reject]
    save_csv(out / 'quarantine' / f'{source["site_id"]}_timestamps.csv', rejected)
    df = df.loc[~reject].sort_values('timestamp').reset_index(drop=True)
    limits = {**BOUNDS, 'power_mw': (0, source['capacity_mw'])}
    counts = {}
    for field, (low, high) in limits.items():
        numeric = pd.to_numeric(df[field], errors='coerce')
        finite = np.isfinite(numeric)
        invalid = ~finite | ~numeric.between(low, high)
        df[f'{field}_raw'] = df[field]
        df[f'{field}_invalid'] = invalid
        counts[field] = {'missing_or_nonnumeric': int(numeric.isna().sum()),
                         'infinite': int((numeric.notna() & ~finite).sum()),
                         'out_of_bounds': int((finite & ~numeric.between(low, high)).sum()),
                         'invalid_total': int(invalid.sum())}
        df[field] = numeric.mask(invalid)
    df['power_pu'] = df.power_mw / source['capacity_mw']
    df['site_id'] = source['site_id']
    df['source_file'] = source['source_file']
    df['source_sheet'] = source['source_sheet']
    df['capacity_mw'] = source['capacity_mw']
    df['all_features_valid'] = df[FIELDS].notna().all(axis=1)
    df['split'] = np.where(df.timestamp < TRAIN_END, 'train', np.where(df.timestamp < VAL_END, 'val', 'test'))
    delta = df.timestamp.diff()
    gaps = pd.DataFrame({'previous_timestamp': df.timestamp.shift(), 'next_timestamp': df.timestamp,
                         'missing_intervals': (delta / pd.Timedelta(minutes=15) - 1)})
    gaps = gaps.loc[delta > pd.Timedelta(minutes=15)].copy()
    gaps['missing_intervals'] = gaps.missing_intervals.astype(int)
    save_csv(out / 'quality' / f'{source["site_id"]}_gaps.csv', gaps)
    save_csv(out / 'cleaned' / f'{source["site_id"]}.csv', df)
    report = {k: v for k, v in source.items() if k != 'data'}
    report.update({'retained_rows': len(df), 'quarantined_timestamp_rows': int(reject.sum()),
                   'start': str(df.timestamp.min()), 'end': str(df.timestamp.max()),
                   'invalid_cells': counts, 'valid_feature_rows': int(df.all_features_valid.sum()),
                   'negative_power_rows': int((pd.to_numeric(df.power_mw_raw, errors='coerce') < 0).sum()),
                   'gap_count': len(gaps), 'missing_intervals': int(gaps.missing_intervals.sum()),
                   'splits': {}})
    return df, report


def build_samples(df, source, out, report):
    site = source['site_id']
    train = df.loc[df.split == 'train', WEATHER]
    mean = train.mean().to_numpy(dtype=np.float32)
    std = train.std(ddof=0).to_numpy(dtype=np.float32)
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError(f'{site}: invalid train-only scaling statistics')
    values = df[MODEL_FIELDS].to_numpy(dtype=np.float32)
    values[:, :4] = (values[:, :4] - mean) / std
    timestamps = df.timestamp.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    payload = {'weather_mean': mean, 'weather_std': std, 'capacity_mw': np.float32(source['capacity_mw']),
               'site_id': np.array(site), 'model_fields': np.array(MODEL_FIELDS)}
    all_indices = []
    for split in ('train', 'val', 'test'):
        positions = np.flatnonzero(df.split.to_numpy() == split)
        targets = positions[HISTORY:]
        kept = []
        for target in targets:
            if (np.isfinite(values[target-HISTORY:target]).all()
                    and np.isfinite(values[target, 4])
                    and (np.diff(timestamps[target-HISTORY:target+1]) == STEP_NS).all()):
                kept.append(target)
        ids = np.asarray(kept, dtype=np.int64)
        x = (np.stack([values[i-HISTORY:i] for i in ids]) if len(ids)
             else np.empty((0, HISTORY, len(MODEL_FIELDS)), dtype=np.float32))
        y = values[ids, 4]
        payload['X_' + split], payload['y_' + split] = x, y
        payload['time_' + split] = timestamps[ids]
        payload['target_row_' + split] = ids
        index = pd.DataFrame({'site_id': site, 'split': split, 'target_row': ids,
                              'history_start': df.timestamp.iloc[ids-HISTORY].to_numpy(),
                              'history_end': df.timestamp.iloc[ids-1].to_numpy(),
                              'target_timestamp': df.timestamp.iloc[ids].to_numpy(),
                              'target_power_mw': df.power_mw.iloc[ids].to_numpy()})
        all_indices.append(index)
        report['splits'][split] = {'rows': len(positions), 'candidate_windows': len(targets),
                                   'samples': len(ids), 'excluded_invalid_or_gap': len(targets)-len(ids)}
    np.savez_compressed(out / 'datasets' / f'{site}.npz', **payload)
    schema = {'site_id': site, 'capacity_mw': source['capacity_mw'], 'capacity_status': source['capacity_status'],
              'raw_fields': FIELDS, 'model_fields': MODEL_FIELDS, 'history_len': HISTORY,
              'interval_minutes': 15, 'horizon_minutes': 15, 'weather_mean': mean.tolist(),
              'weather_std': std.tolist(), 'scaler_fit': 'finite training-period weather values only',
              'bounds': {**BOUNDS, 'power_mw': [0, source['capacity_mw']]},
              'timezone': 'unknown; original wall-clock times retained, no UTC conversion',
              'missing_policy': 'no imputation; reject invalid 16-point history or invalid target power',
              'split_boundaries': {'train_end_exclusive': str(TRAIN_END), 'val_end_exclusive': str(VAL_END)},
              'source_sha256': source['source_sha256'], 'duplicate_capacity_conflict': source['duplicate_capacity_conflict'],
              'scope': 'station PV historical inputs; not full FuXi atmospheric inputs'}
    save_json(out / 'datasets' / f'{site}_schema.json', schema)
    return pd.concat(all_indices, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'data/processed/pv_all_sites_v2')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a new directory to preserve previous results')
    for folder in ['extracted', 'cleaned', 'datasets', 'quarantine', 'quality']:
        (args.output/folder).mkdir(parents=True, exist_ok=True)
    sources = load_sources(args.raw_dir, args.output)
    reports, merged, indices = [], [], []
    for source in sources:
        df, report = clean_source(source, args.output)
        if not source['duplicate_excluded']:
            indices.append(build_samples(df, source, args.output, report))
            merged.append(df)
        reports.append(report)
        print(f'PREPARED {source["site_id"]}: {report["splits"]}', flush=True)
    save_csv(args.output/'pv_all_sites_cleaned.csv', pd.concat(merged, ignore_index=True))
    index = pd.concat(indices, ignore_index=True)
    for split in ('train', 'val', 'test'):
        save_csv(args.output/f'pv_{split}_samples.csv', index.loc[index.split == split])
    registry = [{k: v for k, v in r.items() if k in ['site_id', 'source_file', 'source_sheet', 'capacity_mw',
                 'capacity_status', 'canonical_site_id', 'duplicate_excluded', 'duplicate_capacity_conflict',
                 'raw_rows', 'retained_rows', 'valid_feature_rows', 'negative_power_rows', 'gap_count', 'missing_intervals']} for r in reports]
    save_csv(args.output/'station_registry.csv', pd.DataFrame(registry))
    summary = {'source_workbooks': len(sources), 'raw_pv_rows': sum(r['raw_rows'] for r in reports),
               'canonical_datasets': len(merged), 'canonical_cleaned_rows': sum(len(d) for d in merged),
               'samples': index.groupby('split').size().to_dict(), 'stations': reports,
               'data_preparation_complete': False, 'validation_pending': True,
               'independent_station_metadata_verification_complete': False}
    save_json(args.output/'data_report.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'stations'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
