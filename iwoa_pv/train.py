"""Train the documented research variant and four controlled ablations."""
import argparse
import copy
import hashlib
import json
import math
import random
import time
import os
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import ROOT, FIELDS, PowerBiLSTM, ceemdan_window, vmd_window
import numpy as np
import pandas as pd
import torch


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def scores(truth, prediction, capacity):
    error = (truth - prediction) * capacity
    denominator = np.sum((truth - truth.mean()) ** 2)
    return {'n': len(truth), 'mae_mw': float(np.abs(error).mean()),
            'rmse_mw': float(np.sqrt(np.mean(error ** 2))),
            'r2': float(1 - np.sum((truth - prediction) ** 2) / denominator),
            'mae_capacity_percent': float(np.abs(error).mean() / capacity * 100)}


def load_data(site):
    source = ROOT / f'data/processed/pv_all_sites_v2/cleaned/{site}.csv'
    frame = pd.read_csv(source)
    capacity = float(frame.capacity_mw.iloc[0])
    raw = frame[FIELDS].to_numpy(dtype='float32')
    raw[:, 4] /= capacity
    timestamps = pd.to_datetime(frame.timestamp).to_numpy(dtype='datetime64[ns]')
    valid = frame.all_features_valid.to_numpy(dtype=bool)
    splits = frame.split.to_numpy()
    groups = {}
    for split in ['train', 'val', 'test']:
        rows = []
        for j in range(16, len(raw)):
            # Daylight selection uses the last available irradiance, never target power.
            if splits[j] != split or not np.all(splits[j-16:j] == split):
                continue
            if not valid[j-16:j+1].all() or raw[j-1, 3] <= 20:
                continue
            if not np.all(np.diff(timestamps[j-16:j+1]) == np.timedelta64(15, 'm')):
                continue
            rows.append(j)
        indices = np.asarray(rows, dtype=int)
        if len(indices) == 0:
            raise ValueError(f'No samples for {split}')
        groups[split] = {'x': np.stack([raw[j-16:j] for j in indices]),
                         'y': raw[indices, 4], 'times': frame.timestamp.iloc[indices].tolist(),
                         'rows': indices}
    mean = groups['train']['x'][:, :, :4].mean(axis=(0, 1))
    std = groups['train']['x'][:, :, :4].std(axis=(0, 1)).clip(1e-6)
    for data in groups.values():
        data['x'][:, :, :4] = (data['x'][:, :, :4] - mean) / std
    return groups, capacity, mean, std, hashlib.sha256(source.read_bytes()).hexdigest()


def parallel_map(function, tasks, workers, label):
    directory = ROOT / 'output/iwoa_worker_temp'
    directory.mkdir(parents=True, exist_ok=True)
    kind = 'ceemdan' if function is ceemdan_window else 'vmd'
    if kind == 'ceemdan':
        power, alpha, modes = np.asarray(tasks), 2000, 2
    else:
        power = np.stack([task[0] for task in tasks])
        alpha, modes = tasks[0][1:]
    input_path = directory / 'power.npy'
    np.save(input_path, power)
    start = time.time()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    processes = []
    for worker in range(workers):
        output = directory / f'worker_{worker}.npz'
        command = [sys.executable, '-u', str(Path(__file__).with_name('decomposition.py')),
                   '--input', str(input_path), '--output', str(output), '--kind', kind,
                   '--alpha', str(alpha), '--modes', str(modes), '--worker', str(worker), '--workers', str(workers)]
        processes.append(subprocess.Popen(command, env=environment))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise RuntimeError(f'Decomposition workers failed: {failures}')
    parts = [np.load(directory / f'worker_{worker}.npz') for worker in range(workers)]
    result = np.empty((len(power),) + parts[0]['values'].shape[1:], dtype='float32')
    for part in parts:
        result[part['indices']] = part['values']
        part.close()
    print(f'{label}: {len(power)} windows, {time.time()-start:.1f}s', flush=True)
    return result


class FeatureCache:
    def __init__(self, groups, directory, workers):
        self.groups, self.directory, self.workers = groups, directory, workers
        self.memory = {}

    def get(self, kind, alpha=1000, modes=2):
        key = f'{kind}_{alpha}_{modes}' if 'vmd' in kind else kind
        if key in self.memory:
            return self.memory[key]
        features = {}
        for split, data in self.groups.items():
            path = self.directory / f'{key}_{split}.npy'
            if path.exists():
                features[split] = np.load(path)
                continue
            base = data['x']
            extra = []
            if kind.startswith('ceemdan'):
                ce_path = self.directory / f'ce_components_{split}.npy'
                if ce_path.exists():
                    components = np.load(ce_path)
                else:
                    components = parallel_map(ceemdan_window, base[:, :, 4], self.workers, f'CEEMDAN {split}')
                    np.save(ce_path, components)
                if kind == 'ceemdan':
                    extra.append(components)
                else:
                    extra.append(components[:, :, 1:2])
                    high = components[:, :, 0]
            else:
                high = base[:, :, 4]
            if 'vmd' in kind:
                tasks = [(values, alpha, modes) for values in high]
                extra.append(parallel_map(vmd_window, tasks, self.workers, f'VMD {key} {split}'))
            values = np.concatenate([base] + extra, axis=-1) if extra else base.copy()
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite decomposition {key} {split}')
            np.save(path, values)
            features[split] = values
        self.memory[key] = features
        return features


def fit(features, groups, parameters, epochs, device):
    seed_all()
    model = PowerBiLSTM(features['train'].shape[-1], parameters['hidden']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    arrays = {key: torch.as_tensor(value, device=device) for key, value in features.items()}
    targets = {key: torch.as_tensor(data['y'], device=device) for key, data in groups.items()}
    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(targets['train']), device=device)
        total = 0.0
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(arrays['train'][indices]) - targets['train'][indices]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.no_grad():
            val = model(arrays['val']).clamp(0, 1)
            val_loss = torch.mean((val - targets['val']) ** 2).item()
        history.append({'epoch': epoch + 1, 'train_mse_pu': total / len(order), 'val_mse_pu': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 12:
            break
    model.load_state_dict(best_state)
    return model, best_state, best_loss, best_epoch, history


def decode(position):
    return {'alpha': [500, 2000][int(position[0] >= .5)],
            'modes': [2, 3][int(position[1] >= .5)],
            'hidden': [16, 32, 64][min(2, int(position[2] * 3))],
            'lr': float(10 ** (-3.5 + position[3] * 1.3)),
            'weight_decay': float(10 ** (-6 + position[4] * 3))}


def optimize(cache, groups, device, out, kind='ceemdan_vmd'):
    """Explicit IWOA variant: opposition initialization, nonlinear a, elite mutation."""
    rng = np.random.default_rng(42)
    log_path = out / f'{kind}_optimization_log.json'
    if log_path.exists():
        previous = json.loads(log_path.read_text(encoding='utf-8'))
        if len(previous) == 16:
            return min(previous, key=lambda record: record['val_rmse_pu'])['parameters']
    population = rng.random((4, 5))
    population[2:] = 1 - population[:2]
    best_score, best_position = float('inf'), population[0].copy()
    records = []
    for generation in range(4):
        for position in population:
            params = decode(position)
            features = cache.get(kind, params['alpha'], params['modes'])
            _, _, loss, _, _ = fit(features, groups, params, 6, device)
            records.append({'generation': generation, 'parameters': params, 'val_rmse_pu': math.sqrt(loss)})
            if loss < best_score:
                best_score, best_position = loss, position.copy()
            print(f'IWOA evaluation {len(records)} validation RMSE {math.sqrt(loss):.6f}', flush=True)
            save_json(out / f'{kind}_optimization_log.json', records)
        a = 2 * (1 - (generation + 1) / 4) ** 2
        updated = [best_position.copy()]
        for i in range(1, len(population)):
            r = rng.random(5)
            A, C = 2 * a * r - a, 2 * rng.random(5)
            if rng.random() < .5:
                leader = best_position if np.max(np.abs(A)) < 1 else population[rng.integers(len(population))]
                candidate = leader - A * np.abs(C * leader - population[i])
            else:
                l = rng.uniform(-1, 1)
                candidate = best_position + np.abs(best_position - population[i]) * np.exp(l) * np.cos(2 * np.pi * l)
            if i == 1:
                candidate += rng.normal(0, .1 * (1 - generation / 4), 5)
            updated.append(np.clip(candidate, 0, 1))
        population = np.asarray(updated)
    return decode(best_position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='site_1_b')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--cache-directory', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/iwoa_ceemdan_vmd_bilstm_v1')
    args = parser.parse_args()
    if (args.output / 'metrics.json').exists():
        raise FileExistsError('Completed output exists; use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_directory or ROOT / f'output/iwoa_cache_{args.site}'
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    groups, capacity, mean, std, source_hash = load_data(args.site)
    preprocessing = {'site_id': args.site, 'capacity_mw': capacity, 'fields': FIELDS,
                     'history_len': 16, 'interval_minutes': 15, 'horizon_minutes': 15,
                     'weather_mean': mean.tolist(), 'weather_std': std.tolist(),
                     'timezone': 'unknown; source wall-clock timestamps retained',
                     'source_sha256': source_hash, 'split_counts': {k: len(v['y']) for k,v in groups.items()},
                     'daylight_rule': 'last observed irradiance > 20 W/m2',
                     'split_rule': 'original chronological split; history and target entirely within split',
                     'missing_policy': 'reject invalid or discontinuous history or target; no imputation',
                     'ceemdan': {'trials': 16, 'epsilon': .005, 'max_imf': 3, 'seed': 42},
                     'iwoa': {'population': 4, 'generations': 4, 'fitness_epochs': 6,
                              'improvements': ['opposition initialization', 'quadratic convergence schedule', 'elite Gaussian mutation']}}
    preprocessing_path = args.output / 'preprocessing.json'
    if preprocessing_path.exists():
        previous = json.loads(preprocessing_path.read_text(encoding='utf-8'))
        for key in ['source_sha256', 'site_id', 'split_counts', 'ceemdan']:
            if previous[key] != preprocessing[key]:
                raise ValueError('Existing output uses different training data or decomposition settings')
    save_json(preprocessing_path, preprocessing)
    signature = {'source_sha256': source_hash, 'site_id': args.site,
                 'decomposition_sha256': hashlib.sha256(Path(__file__).with_name('decomposition.py').read_bytes()).hexdigest(),
                 'split_counts': preprocessing['split_counts'], 'ceemdan': preprocessing['ceemdan']}
    signature_path = cache_dir / 'signature.json'
    if signature_path.exists() and json.loads(signature_path.read_text(encoding='utf-8')) != signature:
        raise ValueError('Cache signature differs; use a new --cache-directory')
    save_json(signature_path, signature)
    print(f'Data {preprocessing["split_counts"]}; device {device}', flush=True)
    cache = FeatureCache(groups, cache_dir, args.workers)
    parameters = optimize(cache, groups, device, args.output)
    vmd_parameters = optimize(cache, groups, device, args.output, kind='vmd')
    save_json(args.output / 'selected_parameters.json', parameters)
    defaults = {'alpha': 2000, 'modes': 2, 'hidden': 32, 'lr': .001, 'weight_decay': .00001}
    experiments = [('IWOA-CEEMDAN-VMD-BiLSTM', 'ceemdan_vmd', parameters),
                   ('CEEMDAN-VMD-BiLSTM', 'ceemdan_vmd', defaults),
                   ('IWOA-VMD-BiLSTM', 'vmd', vmd_parameters),
                   ('BiLSTM', 'raw', defaults)]
    results = {}
    predictions = pd.DataFrame({'timestamp': groups['test']['times'], 'target_row': groups['test']['rows'],
                                'actual_mw': groups['test']['y'] * capacity})
    for name, kind, params in experiments:
        print(f'Final training {name}', flush=True)
        features = cache.get(kind, params['alpha'], params['modes'])
        model, state, val_loss, epoch, history = fit(features, groups, params, args.epochs, device)
        with torch.no_grad():
            prediction = model(torch.as_tensor(features['test'], device=device)).clamp(0, 1).cpu().numpy()
        results[name] = dict(scores(groups['test']['y'], prediction, capacity), selected_epoch=epoch,
                             val_rmse_mw=math.sqrt(val_loss)*capacity, parameters=params)
        checkpoint = {'state_dict': state, 'input_size': features['train'].shape[-1],
                      'hidden': params['hidden'], 'kind': kind, 'parameters': params, 'preprocessing': preprocessing}
        path = args.output / ('model.pt' if name == experiments[0][0] else f'{name}.pt')
        torch.save(checkpoint, path)
        reloaded = PowerBiLSTM(checkpoint['input_size'], checkpoint['hidden']).to(device)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        reloaded.load_state_dict(saved['state_dict'])
        reloaded.eval()
        with torch.no_grad():
            check = reloaded(torch.as_tensor(features['test'][:8], device=device)).clamp(0,1).cpu().numpy()
            same_batch = model(torch.as_tensor(features['test'][:8], device=device)).clamp(0,1).cpu().numpy()
        if not np.allclose(check, same_batch, rtol=0, atol=1e-6):
            raise AssertionError('Reload prediction mismatch')
        save_json(args.output / f'{name}_training_log.json', history)
        predictions[name] = prediction * capacity
        print(f'{name} {results[name]}', flush=True)
    persistence = groups['test']['x'][:, -1, 4]
    predictions['persistence_mw'] = persistence * capacity
    results['Persistence'] = scores(groups['test']['y'], persistence, capacity)
    predictions.to_csv(args.output / 'test_predictions.csv', index=False)
    save_json(args.output / 'metrics.json', results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['mae_mw', 'rmse_mw']):
        ax.bar(range(len(results)), [r[metric] for r in results.values()])
        ax.set_xticks(range(len(results)), list(results), rotation=25, ha='right', fontsize=7)
        ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(args.output / 'comparison_metrics.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4))
    subset = predictions.iloc[:160]
    ax.plot(subset.actual_mw.to_numpy(), label='Actual', color='black')
    for name, _, _ in experiments:
        ax.plot(subset[name].to_numpy(), label=name, alpha=.8)
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Chronological daylight test samples (nighttime gaps omitted)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / 'comparison_curves.png', dpi=160)
    plt.close(fig)
    save_json(args.output / 'validation.json', {'passed': True, 'saved_weights_reloaded': True,
              'all_four_hardware_fields_used': True, 'decomposition_uses_history_only': True,
              'test_not_used_for_optimization': True, 'production_ready': False})


if __name__ == '__main__':
    main()
