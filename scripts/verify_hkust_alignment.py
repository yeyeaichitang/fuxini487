"""Audit HKUST metadata and construct explicitly provisional ERA5/PV pairs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/hkust_rooftop_pv'
OUT = DATA / 'verification'
EVIDENCE = ROOT / 'output/hkust_verification'
GLOBAL = ROOT / 'data/weather/era5_hkust_pilot_20211201/input.nc'
FIELDS = ['temperature_c', 'pressure_hpa', 'humidity_percent', 'irradiance_w_m2', 'power_mw']
sys.path.insert(0, str(ROOT / 'full_fuxi_pv'))
from data import load_weather, station_grid, validate_example


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def distance_km(lat, lon, ref_lat=22.3363, ref_lon=114.2634):
    a, b = math.radians(lat), math.radians(ref_lat)
    h = math.sin((a-b)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(lon-ref_lon)/2)**2
    return 6371 * 2 * math.asin(math.sqrt(h))


def main():
    OUT.mkdir(exist_ok=True)
    registry = json.loads((DATA / 'station_registry.json').read_text(encoding='utf-8'))
    latest_text = (EVIDENCE / 'author_metadata_latest.ttl').read_text(encoding='utf-8')
    latest_blocks = re.findall(r'^pvsystem:(\S+) a brick:PV_Generation_System\s*;(.*?)(?=\n\npvsystem:|\Z)', latest_text, re.M | re.S)
    latest_coords = {name: {key: float(value) for key, value in re.findall(r'brick:(latitude|longitude)\s+([\d.]+)', block)} for name, block in latest_blocks}
    author_coordinates_unchanged = all(
        latest_coords[s['metadata_entity']] == {'latitude': s['source_latitude'], 'longitude': s['source_longitude']}
        for s in registry)
    ref_lat, ref_lon = 22.3363, 114.2634
    grid = station_grid(ref_lat, ref_lon)
    coordinate_records, correlations, candidates = [], [], []
    for site in registry:
        original_grid = station_grid(site['source_latitude'], site['source_longitude'])
        coordinate_records.append({
            'site_id': site['site_id'], 'metadata_latitude': site['source_latitude'],
            'metadata_longitude': site['source_longitude'],
            'distance_to_published_campus_km': distance_km(site['source_latitude'], site['source_longitude']),
            'source_grid_row': original_grid[0], 'source_grid_column': original_grid[1],
            'campus_grid_row': grid[0], 'campus_grid_column': grid[1],
            'source_changes_nearest_grid': original_grid != grid,
            'exact_rooftop_coordinates_verified': False,
        })
        candidates.append({
            'site_id': site['site_id'], 'station_name': site['station_name'],
            'fuxi_context_latitude': ref_lat, 'fuxi_context_longitude': ref_lon,
            'location_role': 'published campus representative, not surveyed rooftop coordinates',
            'location_source': 'https://doi.org/10.1038/s41597-025-04397-y',
            'timezone_candidate': 'Asia/Hong_Kong',
            'timezone_status': 'strongly_supported_not_explicitly_confirmed_for_export',
            'pressure_reference': 'sea_level', 'metadata_verified': False,
        })
        if site['selected_for_15min_samples']:
            frame = pd.read_csv(DATA / site['normalized_file'], parse_dates=['timestamp'])
            selected = frame[(frame.timestamp >= '2022-01-01') & (frame.timestamp < '2023-01-01')]
            values = {}
            for shift in [-32, 0, 32]:
                corr = selected.power_mw.corr(selected.irradiance_w_m2.shift(shift))
                values[f'weather_shift_{shift*15}_minutes'] = None if pd.isna(corr) else float(corr)
            correlations.append({'site_id': site['site_id'], **values})
    pd.DataFrame(coordinate_records).to_csv(OUT / 'coordinate_audit.csv', index=False, encoding='utf-8-sig')
    write_json(OUT / 'station_context_candidates.json', candidates)
    write_json(OUT / 'relative_time_alignment.json', correlations)
    valid_corr = [c['weather_shift_0_minutes'] for c in correlations if c['weather_shift_0_minutes'] is not None]
    pressure_contract = {
        'model_pressure_field': 'pressure_hpa', 'model_pressure_unit': 'hPa',
        'model_pressure_reference_for_hkust_training': 'sea_level',
        'raw_hardware_pressure_field': 'station_pressure_hpa',
        'raw_hardware_pressure_reference': 'station',
        'unit_conversions': {'Pa_to_hPa': 'divide by 100', 'kPa_to_hPa': 'multiply by 10', 'mbar_to_hPa': 'unchanged'},
        'reference_conversion_is_not_unit_conversion': True,
        'hardware_fields_required': ['sensor_model', 'sensor_altitude_m_above_mean_sea_level',
                                     'firmware_pressure_reference', 'sea_level_correction_method_and_parameters'],
        'hardware_configuration_available': False,
        'policy': 'Preserve raw station pressure. Supply pressure_hpa only after a documented sea-level conversion, or pass an already corrected sea-level reading with provenance. Reject unknown reference; never silently substitute ERA5 for the hardware observation.',
        'campbell_example': 'CS106 manual section 8.5 uses an elevation-dependent standard-atmosphere correction; this does not prove which correction the dataset logger used.',
        'source': 'https://s.campbellsci.com/documents/us/manuals/cs106.pdf',
    }
    write_json(OUT / 'pressure_input_contract.json', pressure_contract)
    sources = {
        'dataset_paper': 'https://doi.org/10.1038/s41597-025-04397-y',
        'paper_xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11733121/fullTextXML',
        'official_weather_time': 'https://envf.ust.hk/dataview/aws/current/index.py?station=PHSUP2',
        'official_weather_text': 'https://envf.ust.hk/dataview/aws/current/text/PHSUP2_text.txt',
        'hkt_definition': 'https://www.hko.gov.hk/en/gts/time/History_of_HK_Time_Service.htm',
        'author_metadata': 'https://github.com/ZinanLin-Oscar/SPARQL-Example-for-PV-Brick-Model',
        'pressure_sensor_manual': pressure_contract['source'],
        'era5_mirror': 'https://github.com/google-research/arco-era5',
    }
    pairing = {'global_input_available': GLOBAL.exists(), 'candidate_pair_count': 0,
               'formally_verified_pair_count': 0, 'ready_for_training': False}
    if GLOBAL.exists():
        # Load the exact original input schema, normalization and frame spacing.
        values, init = load_weather(GLOBAL)
        del values
        weather_table = pd.read_csv(DATA / 'weather/weather_15min_trailing.csv', parse_dates=['timestamp']).set_index('timestamp')
        with xr.open_dataset(GLOBAL) as ds, xr.open_dataarray(ROOT / 'fuxi-2.1/mean.nc') as means, xr.open_dataarray(ROOT / 'fuxi-2.1/std.nc') as stds:
            idx = list(ds.channel.values).index('msl')
            mean, std = float(means.values[idx]), float(stds.values[idx])
            pressure_checks = []
            for i, valid_time in enumerate(ds.time.values):
                utc = pd.Timestamp(valid_time).tz_localize('UTC')
                local = utc.tz_convert('Asia/Hong_Kong').tz_localize(None)
                era_pressure = (float(ds['input'][i, idx, grid[0], grid[1]]) * (std + 1e-8) + mean) / 100
                observed = float(weather_table.loc[local, 'pressure_hpa'])
                pressure_checks.append({'time_utc': utc.isoformat(), 'time_hkt_hypothesis': local.isoformat() + '+08:00',
                                        'era5_msl_hpa': era_pressure, 'dataset_slp_hpa': observed,
                                        'dataset_minus_era5_hpa': observed - era_pressure})
        count = 0
        with (OUT / 'candidate_pairs.jsonl').open('w', encoding='utf-8') as output:
            for site in registry:
                if not site['selected_for_15min_samples']:
                    continue
                schema = json.loads((DATA / 'datasets' / (site['site_id'] + '_schema.json')).read_text())
                frame = pd.read_csv(DATA / site['normalized_file'], parse_dates=['timestamp'])
                with np.load(DATA / 'datasets' / (site['site_id'] + '.npz')) as arrays:
                    ids = arrays['target_row_train']
                as_of_utc = pd.DatetimeIndex(frame.timestamp).tz_localize('Asia/Hong_Kong').tz_convert('UTC')
                for j in ids:
                    as_of = as_of_utc[int(j)-1]
                    age = (as_of - pd.Timestamp(init)).total_seconds()
                    if not 0 <= age <= 6*3600:
                        continue
                    history = []
                    for i in range(int(j)-16, int(j)):
                        row = {f: float(frame.iloc[i][f]) for f in FIELDS}
                        row['timestamp'] = frame.iloc[i].timestamp.isoformat() + '+08:00'
                        history.append(row)
                    request = {'station_id': site['site_id'], 'capacity_mw': site['capacity_mw'],
                               'interval_minutes': 15, 'as_of': history[-1]['timestamp'], 'history': history,
                               'pressure_reference': 'sea_level'}
                    candidate = {
                        'request': request, 'schema': schema, 'split': 'train',
                        'target_timestamp': frame.iloc[int(j)].timestamp.isoformat() + '+08:00',
                        'target_power_mw': float(frame.iloc[int(j)].power_mw),
                        'target_source': 'observed historical power, no imputation',
                        'weather_path': str(GLOBAL), 'weather_init_utc': init.isoformat(),
                        'latitude': ref_lat, 'longitude': ref_lon,
                        'pair_status': 'provisional_not_approved_for_training',
                        'timezone_hypothesis': 'Asia/Hong_Kong',
                        'location_role': 'published campus representative',
                        'metadata_verified': False, 'pressure_reference': 'sea_level',
                    }
                    validate_example(candidate)
                    output.write(json.dumps(candidate, ensure_ascii=False, allow_nan=False) + '\n')
                    count += 1
        pairing.update({'candidate_pair_count': count, 'input_times_utc': ['2021-12-01T12:00:00+00:00', init.isoformat()],
                        'weather_path': str(GLOBAL.relative_to(ROOT)), 'fuXi_shape_channels_normalization_checked': True,
                        'candidate_input_contract_checked': True, 'pressure_comparison': pressure_checks,
                        'comparison_scope': 'Two contemporaneous points on a coarse reanalysis grid; detects potential bias, not sensor calibration truth',
                        'download_manifest': str(GLOBAL.with_name('download_manifest.json').relative_to(ROOT))})
    evidence = []
    for path in EVIDENCE.iterdir():
        if path.is_file():
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            evidence.append({'path': str(path.relative_to(ROOT)), 'sha256': digest, 'bytes': path.stat().st_size})
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'sources': sources,
        'timezone': {'candidate': 'Asia/Hong_Kong', 'utc_offset_hours': 8,
                     'official_aws_display_uses_hkt': True, 'dataset_export_explicitly_documents_timezone': False,
                     'status': 'strongly_supported_not_export_author_confirmed',
                     'relative_power_weather_zero_shift_median_correlation': float(np.median(valid_corr)),
                     'relative_alignment_is_not_absolute_timezone_proof': True},
        'coordinates': {'published_campus_latitude': ref_lat, 'published_campus_longitude': ref_lon,
                        'author_latest_coordinates_same_as_archive': author_coordinates_unchanged,
                        'nearest_fuxi_grid': list(grid), 'nearest_grid_latitude': 90-grid[0]/4, 'nearest_grid_longitude': grid[1]/4,
                        'exact_rooftop_coordinates_verified': False,
                        'source_coordinates_change_grid_for_stations': sum(x['source_changes_nearest_grid'] for x in coordinate_records),
                        'maximum_source_distance_from_published_campus_km': max(x['distance_to_published_campus_km'] for x in coordinate_records),
                        'policy': 'Use the documented campus representative only for an explicitly labeled research pilot. Do not treat it as independently verified rooftop coordinates.'},
        'pressure': {'dataset_declared_reference_verified': 'sea_level', 'dataset_unit_verified': 'hPa',
                     'physical_sea_level_reduction_verified': False,
                     'possible_offset_against_era5_requires_further_investigation': True,
                     'sensor_identified_in_paper': 'Campbell Scientific CS106',
                     'sensor_model_alone_does_not_prove_logger_sea_level_correction': True,
                     'hardware_configuration_available': False, 'automatic_hardware_equivalence_verified': False,
                     'historical_values_changed': False, 'input_contract': 'pressure_input_contract.json'},
        'era5_pairing': pairing, 'evidence': evidence,
        'full_fuxi_training_ready': False, 'hardware_deployment_ready': False,
        'unresolved': ['Explicit timezone and interval-label confirmation for the exported PV/weather dataset',
                       'Exact rooftop locations; campus representative is explicitly provisional for the pilot',
                       'Actual hardware pressure reference, sensor altitude and correction configuration'],
    }
    write_json(OUT / 'verification_report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'evidence'}, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
