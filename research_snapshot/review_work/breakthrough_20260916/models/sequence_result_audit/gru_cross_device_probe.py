"""Prepared parent-scheduled GPU replay of exact bounded GRU input tensors."""
import torch
from pathlib import Path
import sys
import argparse
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'review_work/breakthrough_20260916/sequence_context'))
import adapter
import cache

OUT = ROOT / 'private_runs/breakthrough_20260916/models/sequence_result_audit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    target = OUT / f'gru_cross_device_probe_{args.device}.json'
    if target.exists():
        raise ValueError('Preserve previous probe')
    results = {}
    for fold in ('F1', 'F3'):
        path = OUT / f'gru_{fold}_cross_device_fixture.joblib'
        data = joblib.load(path)
        network = adapter.Network(**data['architecture']).to(args.device)
        network.load_state_dict(data['state'])
        network.eval()
        with torch.no_grad():
            value = network(*adapter.tensors(data['query'], data['event'], args.device), contextual=True)
            residual = value.cpu().numpy() * data['target_scale'] + data['target_mean']
        prediction = residual.astype('float64') + data['proxy']
        saved_delta = np.abs(prediction - data['saved_gpu_prediction'])
        cpu_delta = np.abs(prediction - data['cpu_prediction'])
        results[fold] = {'rows': len(prediction), 'fixture_sha256': cache.sha(path),
            'max_delta_from_original_saved_GPU_sec': float(saved_delta.max()),
            'max_delta_from_independent_CPU_sec': float(cpu_delta.max()),
            'saved_GPU_delta_percentiles_sec': np.quantile(saved_delta, [.5, .99, 1]).tolist(),
            'CPU_delta_percentiles_sec': np.quantile(cpu_delta, [.5, .99, 1]).tolist()}
        print('GRU_DEVICE_PROBE', args.device, fold, results[fold], flush=True)
    cache.write_json(target, results)


if __name__ == '__main__':
    main()
