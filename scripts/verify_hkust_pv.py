"""Independently verify prepared HKUST values, sample windows and raw provenance."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import zlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/processed/hkust_rooftop_pv'
RAW = ROOT / 'data/raw/public_datasets/hkust_rooftop_pv_2021_2023'
SOURCE = RAW / 'extracted/Dataset'
sys.path.insert(0, str(ROOT / 'full_fuxi_pv'))
from contract import prepare_history

FIELDS = ['temperature_c', 'pressure_hpa', 'humidity_percent', 'irradiance_w_m2', 'power_mw']


def main():
    report = json.loads((OUT / 'data_report.json').read_text(encoding='utf-8'))
    assert report['completed'] and not report['full_fuxi_training_ready']
    inventory = json.loads((OUT / 'source_inventory.json').read_text(encoding='utf-8'))
    assert len(inventory) == 125
    for item in inventory:
        with (SOURCE / item['path']).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == item['sha256']
    # Compare all extracted bytes with the original downloaded archive.
    with zipfile.ZipFile(RAW / 'dataset.zip') as archive:
        verified_raw = 0
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            path = RAW / 'extracted' / entry.filename
            crc = 0
            with path.open('rb') as handle:
                while chunk := handle.read(1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            assert crc == entry.CRC and path.stat().st_size == entry.file_size
            verified_raw += 1
    print(f'Original files preserved and CRC verified: {verified_raw}', flush=True)
    weather = pd.read_csv(OUT / 'weather/weather_15min_trailing.csv', parse_dates=['timestamp']).set_index('timestamp')
    weather_counts = pd.read_csv(OUT / 'weather/weather_15min_counts.csv', parse_dates=['timestamp']).set_index('timestamp')
    # Recompute all four required weather fields directly from original observations.
    specs = [('Temperature', FIELDS[0], -50, 70), ('Sea Level Pressure', FIELDS[1], 300, 1100),
             ('Relative Humidity', FIELDS[2], 0, 100), ('Irradiance', FIELDS[3], 0, 1500)]
    checked_weather = 0
    for folder, field, low, high in specs:
        blocks = []
        for path in sorted((SOURCE / 'Time series dataset/Meteorological dataset' / folder).glob('*.csv')):
            raw = pd.read_csv(path)
            times = pd.to_datetime(raw.iloc[:, 0], format='mixed')
            values = pd.to_numeric(raw.iloc[:, 1], errors='coerce')
            values = values.where(np.isfinite(values) & (values >= low) & (values <= high))
            # Ceiling-minute grouping independently matches (T-15min,T].
            groups = pd.DataFrame({'end': times.dt.ceil('15min'), 'value': values})
            blocks.append(groups)
        combined = pd.concat(blocks, ignore_index=True)
        grouped = combined.groupby('end').value.agg(['mean', 'count'])
        expected = grouped['mean'].where(grouped['count'] == 15).reindex(weather.index)
        np.testing.assert_allclose(weather[field], expected, rtol=1e-10, atol=1e-8, equal_nan=True)
        np.testing.assert_array_equal(weather_counts[field + '_valid_count'], grouped['count'].reindex(weather.index).fillna(0))
        checked_weather += len(expected)
    print('All required weather aggregations match original minute data', flush=True)
    registry = json.loads((OUT / 'station_registry.json').read_text(encoding='utf-8'))
    assert len(registry) == 60
    totals = {'train': 0, 'val': 0, 'test': 0}
    checked_npz, checked_station_rows = 0, 0
    for site in registry:
        frame = pd.read_csv(OUT / site['normalized_file'], parse_dates=['timestamp'])
        assert not frame.timestamp.duplicated().any()
        assert frame.timestamp.is_monotonic_increasing
        assert (frame.timestamp.diff().dropna() == pd.Timedelta(minutes=site['interval_minutes'])).all()
        raw = pd.read_csv(SOURCE / site['source_file'])
        dates = pd.to_datetime(raw.Time, format='mixed').astype('datetime64[ns]')
        power = pd.to_numeric(raw['power(W)'], errors='coerce') / 1e6
        power = power.where(np.isfinite(power) & power.between(0, site['capacity_mw']))
        unique = ~dates.duplicated(keep=False)
        expected_power = pd.Series(power[unique].to_numpy(), index=dates[unique]).reindex(frame.timestamp)
        np.testing.assert_allclose(frame.power_mw, expected_power, rtol=1e-10, atol=1e-12, equal_nan=True)
        joined = weather[FIELDS[:4]].reindex(frame.timestamp)
        np.testing.assert_allclose(frame[FIELDS[:4]], joined, rtol=1e-10, atol=1e-8, equal_nan=True)
        checked_station_rows += len(frame)
        if not site['selected_for_15min_samples']:
            continue
        schema = json.loads((OUT / 'datasets' / (site['site_id'] + '_schema.json')).read_text(encoding='utf-8'))
        with np.load(OUT / 'datasets' / (site['site_id'] + '.npz'), allow_pickle=False) as arrays:
            raw_values = frame[FIELDS].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(arrays['raw_features'], raw_values, rtol=1e-6, atol=1e-6, equal_nan=True)
            np.testing.assert_array_equal(arrays['timestamp_local_ns'], pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8)
            good = np.isfinite(raw_values).all(axis=1)
            np.testing.assert_array_equal(arrays['row_valid'], good)
            fit = good & (frame.timestamp < pd.Timestamp('2022-07-01')).to_numpy()
            mean = raw_values[fit, :4].mean(axis=0)
            std = raw_values[fit, :4].std(axis=0)
            std = np.where(std < 1e-8, 1., std)
            np.testing.assert_allclose(schema['weather_mean'], mean, rtol=1e-10, atol=1e-8)
            np.testing.assert_allclose(schema['weather_std'], std, rtol=1e-10, atol=1e-8)
            scaled = raw_values.copy()
            scaled[:, :4] = (scaled[:, :4] - mean) / std
            scaled[:, 4] /= site['capacity_mw']
            np.testing.assert_allclose(arrays['model_features'], scaled, rtol=1e-6, atol=1e-6, equal_nan=True)
            # Rolling sums are independent of the builder's prefix-sum implementation.
            history_complete = pd.Series(good).rolling(16, min_periods=16).sum().shift(1).eq(16).to_numpy()
            target_good = np.isfinite(raw_values[:, 4])
            eligible = history_complete & target_good
            codes = np.where(frame.timestamp < pd.Timestamp('2022-07-01'), 0,
                             np.where(frame.timestamp < pd.Timestamp('2023-01-01'), 1, 2))
            for code, split in enumerate(totals):
                same_period = np.zeros(len(frame), dtype=bool)
                same_period[16:] = (codes[16:] == codes[:-16])
                expected_ids = np.flatnonzero(eligible & (codes == code) & same_period)
                indices = arrays['target_row_' + split]
                np.testing.assert_array_equal(indices, expected_ids)
                np.testing.assert_allclose(arrays['y_' + split], raw_values[indices, 4] / site['capacity_mw'], rtol=1e-6)
                totals[split] += len(indices)
                if len(indices):
                    # Exercise the actual model input contract at both ends of every split.
                    for j in [int(indices[0]), int(indices[-1])]:
                        history = []
                        for i in range(j - 16, j):
                            row = {f: float(frame.iloc[i][f]) for f in FIELDS}
                            row['timestamp'] = frame.iloc[i].timestamp.isoformat() + '+08:00'
                            history.append(row)
                        request = {'station_id': site['site_id'], 'capacity_mw': site['capacity_mw'],
                                   'interval_minutes': 15, 'as_of': history[-1]['timestamp'], 'history': history}
                        features, quality, _ = prepare_history(request, schema, training=True)
                        assert quality['power_imputed_count'] == 0
                        np.testing.assert_allclose(features, arrays['model_features'][j-16:j], rtol=1e-5, atol=1e-5)
        checked_npz += 1
        print(f'Verified {site["site_id"]}', flush=True)
    assert totals == report['samples']
    assert checked_npz == report['stations_with_indexed_samples']
    assert len(list((OUT / 'auxiliary/inverters').glob('*.csv.gz'))) == 44
    checked_inverter_rows = 0
    for item in inventory:
        if 'Inverter level dataset/' not in item['path']:
            continue
        source_path = SOURCE / item['path']
        raw = pd.read_csv(source_path)
        times = pd.to_datetime(raw.pop('Time'), format='mixed', errors='coerce')
        keep = times.notna() & ~times.duplicated(keep=False)
        expected = raw.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        expected.index = pd.DatetimeIndex(times)
        expected = expected.loc[keep.to_numpy()].sort_index()
        actual = pd.read_csv(OUT / 'auxiliary/inverters' / (source_path.stem + '.csv.gz'), parse_dates=['timestamp'])
        np.testing.assert_array_equal(pd.DatetimeIndex(actual.timestamp).as_unit('ns').asi8,
                                      expected.index.as_unit('ns').asi8)
        np.testing.assert_allclose(actual[list(expected.columns)], expected, rtol=1e-10, atol=1e-8, equal_nan=True)
        np.testing.assert_array_equal(actual.negative_total_active_power_flag, expected['totalActivePower(W)'] < 0)
        checked_inverter_rows += len(actual)
    print('All 44 inverter tables match their original numeric observations', flush=True)
    result = {'passed': True, 'raw_files_crc_verified': verified_raw, 'source_tables_sha256_verified': len(inventory),
              'weather_aggregated_values_checked': checked_weather, 'station_rows_checked': checked_station_rows,
              'indexed_station_datasets_checked': checked_npz, 'sample_counts_verified': totals,
              'actual_model_input_contract_checked': True, 'imputed_training_values': 0,
              'auxiliary_inverter_rows_checked': checked_inverter_rows,
              'full_fuxi_training_ready': False, 'hardware_deployment_ready': False,
              'scope': 'numeric/provenance verification; does not verify unknown timezone, coordinates or pressure equivalence'}
    (OUT / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
