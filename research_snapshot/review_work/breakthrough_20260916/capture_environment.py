"""Record reproducibility metadata without environment variables or credentials."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = external_path(args.output)
    if path.exists():
        raise ValueError('Preserve existing environment receipt')
    packages = {}
    for name in ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost',
                 'catboost', 'torch', 'tabm', 'rtdl_num_embeddings', 'pytabkit', 'tabicl',
                 'tabdpt', 'faiss-cpu', 'hmmlearn', 'joblib', 'psutil']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                          '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'packages': packages, 'host_total_bytes': psutil.virtual_memory().total,
              'logical_cpus': psutil.cpu_count(), 'gpu_inventory': gpu.stdout.strip(),
              'gpu_inventory_exit_code': gpu.returncode,
              'scope': 'Version snapshot only. Run manifests and source/checkpoint hashes bind actual experiments.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('ENVIRONMENT_CAPTURED', path, flush=True)


if __name__ == '__main__':
    main()
