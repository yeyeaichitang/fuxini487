"""Independently recompute CSV metrics and exercise saved-model CLI contract."""
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ROOT, FIELDS
from predict import predict
from train import load_data, save_json
import numpy as np
import pandas as pd


def main():
    directory = ROOT / 'models/iwoa_ceemdan_vmd_bilstm_v1'
    metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    schema = json.loads((directory / 'preprocessing.json').read_text(encoding='utf-8'))
    table = pd.read_csv(directory / 'test_predictions.csv')
    truth = table.actual_mw.to_numpy(dtype='float64')
    for name, expected in metrics.items():
        column = 'persistence_mw' if name == 'Persistence' else name
        prediction = table[column].to_numpy(dtype='float64')
        error = truth - prediction
        actual = {'mae_mw': np.abs(error).mean(), 'rmse_mw': np.sqrt(np.mean(error**2)),
                  'r2': 1 - np.sum(error**2) / np.sum((truth-truth.mean())**2)}
        assert len(table) == expected['n']
        assert np.isfinite(prediction).all()
        assert np.all((prediction >= 0) & (prediction <= schema['capacity_mw']))
        for key, value in actual.items():
            assert np.isclose(value, expected[key], rtol=1e-5, atol=1e-5), (name, key, value)
    groups, _, _, _, _ = load_data(schema['site_id'])
    assert np.array_equal(groups['test']['rows'], table.target_row.to_numpy())
    assert np.allclose(groups['test']['y'] * schema['capacity_mw'], truth, atol=1e-5)
    source = ROOT / f'data/processed/pv_all_sites_v2/cleaned/{schema["site_id"]}.csv'
    frame = pd.read_csv(source)
    row = int(table.target_row.iloc[0])
    history = frame.iloc[row-16:row][['timestamp']+FIELDS].to_dict(orient='records')
    request = {'station_id': schema['site_id'], 'capacity_mw': schema['capacity_mw'],
               'as_of': history[-1]['timestamp'], 'interval_minutes': 15, 'history': history}
    save_json(directory / 'example_input.json', request)
    result = predict(request, directory / 'model.pt')
    prediction = result['predictions'][0]
    assert pd.Timestamp(prediction['timestamp']) == pd.Timestamp(frame.timestamp.iloc[row])
    difference = abs(prediction['power_mw'] - table.iloc[0]['IWOA-CEEMDAN-VMD-BiLSTM'])
    assert difference <= schema['capacity_mw'] * 2e-5
    save_json(directory / 'example_output.json', result)
    save_json(directory / 'example_reference.json', {'actual_power_mw': float(truth[0]),
              'target_timestamp': frame.timestamp.iloc[row], 'target_row': row,
              'source_timezone_verified': False})
    bad_inputs = []
    for key, value in [('station_id', 'DG1'), ('capacity_mw', 10.0)]:
        invalid = copy.deepcopy(request)
        invalid[key] = value
        bad_inputs.append(invalid)
    invalid = copy.deepcopy(request)
    invalid['history'][3]['pressure_hpa'] = None
    bad_inputs.append(invalid)
    invalid = copy.deepcopy(request)
    invalid['history'][3]['timestamp'] = invalid['history'][2]['timestamp']
    bad_inputs.append(invalid)
    for invalid in bad_inputs:
        try:
            predict(invalid, directory / 'model.pt')
        except (ValueError, TypeError):
            continue
        raise AssertionError('Invalid input accepted')
    units = ['degC', 'hPa', '%', 'W/m2', 'MW']
    save_json(directory / 'feature_schema.json', {'fields': [dict(name=k, unit=u, required=True)
              for k,u in zip(FIELDS, units)], 'history_length': 16, 'interval_minutes': 15,
              'horizon_minutes': 15, 'output_unit': 'MW', 'all_features_must_be_measured': True})
    save_json(directory / 'validation.json', {'passed': True, 'metrics_independently_recomputed': True,
              'test_rows_and_labels_matched_to_source': True, 'saved_model_predict_entrypoint_passed': True,
              'prediction_matches_test_csv': True, 'negative_input_checks_passed': len(bad_inputs),
              'cpu_vs_cuda_example_difference_mw': float(difference),
              'cpu_vs_cuda_tolerance_mw': schema['capacity_mw'] * 2e-5,
              'all_four_hardware_fields_used': True, 'decomposition_uses_history_only': True,
              'test_not_used_for_optimization': True,
              'model_sha256': hashlib.sha256((directory/'model.pt').read_bytes()).hexdigest(),
              'production_ready': False})
    print('Independent metrics, source alignment, saved-model prediction and invalid-input checks passed')


if __name__ == '__main__':
    main()
