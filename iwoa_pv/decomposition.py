"""CPU decomposition worker with no CUDA runtime imports."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'output/iwoa_dependencies'))
import numpy as np
from PyEMD import CEEMDAN
from vmdpy import VMD


def ceemdan_window(power):
    if np.std(power) < 1e-10:
        return np.stack([np.zeros_like(power), power], axis=-1).astype('float32')
    decomposition = CEEMDAN(trials=16, epsilon=0.005, parallel=False)
    decomposition.noise_seed(42)
    components = decomposition(power.astype('float64'), max_imf=3)
    high = components[0]
    return np.stack([high, power - high], axis=-1).astype('float32')


def vmd_window(task):
    power, alpha, modes = task
    if np.std(power) < 1e-10:
        return np.column_stack([np.zeros((len(power), modes)), power]).astype('float32')
    values, _, _ = VMD(power.astype('float64'), alpha, 0, modes, 0, 1, 1e-6)
    return np.column_stack([values.T, power - values.sum(axis=0)]).astype('float32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=['ceemdan', 'vmd'], required=True)
    parser.add_argument('--alpha', type=int, default=2000)
    parser.add_argument('--modes', type=int, default=2)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    power = np.load(args.input, mmap_mode='r')
    indices = np.arange(len(power))[args.worker::args.workers]
    results = []
    for i, index in enumerate(indices):
        values = power[index]
        results.append(ceemdan_window(values) if args.kind == 'ceemdan' else vmd_window((values, args.alpha, args.modes)))
        if (i + 1) % 1000 == 0:
            print(f'{args.kind} worker {args.worker}: {i+1}/{len(indices)}', flush=True)
    np.savez(args.output, indices=indices, values=np.stack(results))
