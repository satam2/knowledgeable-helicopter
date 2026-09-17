"""Sequential approved CPU experiments following an existing owned fit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path
OUT = external_path(ROOT / 'private_runs/tail240_20260916/models/declared_cpu_queue_v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='ascii')


def main(pid):
    owned = psutil.Process(pid)
    assert 'ordinary_state_tune.py' in ' '.join(owned.cmdline())
    created = owned.create_time()
    OUT.mkdir(parents=True, exist_ok=False)
    model_root = ROOT / 'review_work/tail240_20260916/models'
    jobs = [
        ('ordinary_state_F3', model_root / 'ordinary_state_tune.py', ['--fold', 'F3']),
        ('scaled_finite_F1', model_root / 'scaled_finite_tune.py', ['--fold', 'F1']),
        ('scaled_finite_F3', model_root / 'scaled_finite_tune.py', ['--fold', 'F3']),
        ('scaled_finite_assessment', model_root / 'assess_scaled_finite.py', []),
    ]
    frozen = {str(path): digest(path) for _, path, _ in jobs}
    save(OUT / 'protocol.json', dict(source_sha256=digest(Path(__file__)), source_hashes=frozen,
        predecessor_pid=pid, predecessor_create_time=created,
        jobs=[dict(name=name, source=str(path), arguments=args) for name,path,args in jobs],
        policy='One existing allocated 2CPU/10GiB main lane; sequential jobs, 18GiB startup,8GiB reserve. No automatic score or refit advancement. Preserve failure and stop queue on failed child.'))
    print('WAIT_PREDECESSOR', pid, flush=True)
    while psutil.pid_exists(pid):
        try:
            if psutil.Process(pid).create_time() != created:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(5)
    predecessor = ROOT / 'private_runs/tail240_20260916/models/ordinary_state_tune_v1/F1/manifest.json'
    assert json.loads(predecessor.read_text())['status'] == 'complete'
    results = []
    for name, path, args in jobs:
        assert digest(path) == frozen[str(path)]
        while psutil.virtual_memory().available < 18 * 1024**3:
            print('WAIT_MEMORY', name, psutil.virtual_memory().available, flush=True)
            time.sleep(15)
        started = time.time()
        print('START', name, flush=True)
        command = [str(ROOT / '.review-venv/Scripts/python.exe'), '-B', '-u', str(path), *args]
        result = subprocess.run(command, cwd=ROOT)
        record = dict(name=name, command=command, start_unix=started,
            end_unix=time.time(), returncode=result.returncode)
        results.append(record)
        save(OUT / 'results.json', results)
        print('FINISH', record, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    save(OUT / 'complete.json', dict(status='complete', results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predecessor-pid', type=int, required=True)
    main(parser.parse_args().predecessor_pid)
