"""Load the complete BiLSTM checkpoint and forecast from 16 observed records."""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ROOT, FIELDS, PowerBiLSTM, ceemdan_window, vmd_window
import numpy as np
import torch


def predict(request, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    schema = saved['preprocessing']
    if request['station_id'] != schema['site_id']:
        raise ValueError('Station does not match trained station')
    if not np.isclose(request['capacity_mw'], schema['capacity_mw'], rtol=0, atol=1e-8):
        raise ValueError('Capacity does not match training metadata')
    if request.get('interval_minutes') != 15 or len(request['history']) != 16:
        raise ValueError('Exactly 16 records at 15-minute intervals required')
    times = [datetime.fromisoformat(row['timestamp']) for row in request['history']]
    if any(b-a != timedelta(minutes=15) for a,b in zip(times, times[1:])):
        raise ValueError('History has discontinuous timestamps')
    if datetime.fromisoformat(request['as_of']) != times[-1]:
        raise ValueError('as_of must match latest history record')
    raw = np.asarray([[row[field] for field in FIELDS] for row in request['history']], dtype='float32')
    if not np.isfinite(raw).all():
        raise ValueError('Every historical measurement must be finite')
    lower = np.array([-50, 300, 0, 0, 0])
    upper = np.array([70, 1100, 100, 1500, schema['capacity_mw']])
    if np.any(raw < lower) or np.any(raw > upper):
        raise ValueError('Measurements outside supported physical bounds')
    if raw[-1, 3] <= 20:
        raise ValueError('Research model evaluated only when latest irradiance exceeds 20 W/m2')
    raw[:, 4] /= schema['capacity_mw']
    mean = np.asarray(schema['weather_mean'], dtype='float32')
    std = np.asarray(schema['weather_std'], dtype='float32')
    raw[:, :4] = (raw[:, :4] - mean) / std
    parameters = saved['parameters']
    if saved['kind'] == 'ceemdan_vmd':
        components = ceemdan_window(raw[:, 4])
        modes = vmd_window((components[:, 0], parameters['alpha'], parameters['modes']))
        values = np.concatenate([raw, components[:, 1:2], modes], axis=-1)
    elif saved['kind'] == 'vmd':
        values = np.concatenate([raw, vmd_window((raw[:, 4], parameters['alpha'], parameters['modes']))], axis=-1)
    else:
        values = raw
    model = PowerBiLSTM(saved['input_size'], saved['hidden'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    with torch.no_grad():
        power = float(model(torch.from_numpy(values[None].astype('float32'))).clamp(0, 1).item())
    return {'status': 'ok', 'station_id': request['station_id'], 'based_on': request['as_of'],
            'interval_minutes': 15, 'model_name': 'pv-forecast',
            'model_version': 'iwoa-ceemdan-vmd-bilstm-v1',
            'predictions': [{'timestamp': (times[-1] + timedelta(minutes=15)).isoformat(),
                             'power_mw': power * schema['capacity_mw']}],
            'research_only': True, 'source_timezone_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'models/iwoa_ceemdan_vmd_bilstm_v1/model.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = predict(json.loads(args.input.read_text(encoding='utf-8')), args.checkpoint)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
