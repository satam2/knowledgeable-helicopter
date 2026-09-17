"""Single-child GPU scheduler with explicit experiment budgets and durable receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledgeable-helicopter-screening/src'))
from taxiout.paths import external_path


def now():
    return datetime.now(timezone.utc)


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def stop_tree(process, known_descendants=()):
    captured = set(known_descendants)
    try:
        owned = psutil.Process(process.pid)
        captured.update(owned.children(recursive=True))
        captured.add(owned)
    except psutil.NoSuchProcess:
        pass
    for child in captured:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(captured, timeout=10)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=10)
    if alive:
        raise RuntimeError('Owned child tree did not exit; refuse another GPU launch')
    process.wait(timeout=20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    destination = external_path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    deadline = datetime.fromisoformat(plan['deadline_utc'])
    python = ROOT / '.breakthrough-venv/Scripts/python.exe'
    for source in [Path(__file__), args.plan]:
        shutil.copyfile(source, destination / source.name)
    results = {}
    identity = {'created_utc': now().isoformat(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                'gpu_policy': 'Parent confirmed exclusive GPU ownership before launch. At most one owned child command at a time.',
                'deadline_utc': plan['deadline_utc']}
    write(destination / 'protocol.json', identity)
    for job in plan['jobs']:
        name = job['name']
        if any(results.get(dep, {}).get('status') != 'complete' for dep in job.get('requires', [])):
            results[name] = {'status': 'skipped_dependency'}
            write(destination / 'results.json', results)
            continue
        if (deadline - now()).total_seconds() < job.get('minimum_remaining_seconds', 120):
            results[name] = {'status': 'skipped_deadline'}
            write(destination / 'results.json', results)
            continue
        wait_start = time.monotonic()
        while psutil.virtual_memory().available < job.get('minimum_available_gib', 12) * 1024 ** 3:
            if (deadline - now()).total_seconds() < job.get('minimum_remaining_seconds', 120) or time.monotonic() - wait_start > 600:
                break
            print('WAIT_RAM', name, round(psutil.virtual_memory().available / 2**30, 2), flush=True)
            time.sleep(20)
        if psutil.virtual_memory().available < job.get('minimum_available_gib', 12) * 1024 ** 3:
            results[name] = {'status': 'skipped_ram_budget'}
            write(destination / 'results.json', results)
            continue
        if (deadline - now()).total_seconds() < job.get('minimum_remaining_seconds', 120):
            results[name] = {'status': 'skipped_deadline_after_ram_wait'}
            write(destination / 'results.json', results)
            continue
        command = [str(python), '-B', '-u', *job['arguments']]
        started = time.monotonic()
        record = {'status': 'running', 'started_utc': now().isoformat(), 'command': command,
                  'max_seconds': job['max_seconds'], 'initial_available_gib': psutil.virtual_memory().available / 2**30}
        results[name] = record
        write(destination / 'results.json', results)
        print('START', name, now().isoformat(), flush=True)
        with (destination / (name + '.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            known_descendants = set()
            try:
                record['pid'] = process.pid
                write(destination / 'results.json', results)
                last_report = started
                reserve_low_since = None
                while process.poll() is None:
                    try:
                        known_descendants.update(psutil.Process(process.pid).children(recursive=True))
                    except psutil.NoSuchProcess:
                        pass
                    elapsed = time.monotonic() - started
                    available = psutil.virtual_memory().available / 2**30
                    reserve_low_since = time.monotonic() if available < 8 and reserve_low_since is None else reserve_low_since
                    if available >= 8:
                        reserve_low_since = None
                    reason = ('deadline' if now() >= deadline else 'time_budget' if elapsed > job['max_seconds'] else
                              'ram_reserve' if reserve_low_since is not None and time.monotonic() - reserve_low_since > 10 else None)
                    if reason:
                        record['stop_reason'] = reason
                        stop_tree(process, known_descendants)
                        break
                    if time.monotonic() - last_report >= 30:
                        print('RUNNING', name, round(elapsed, 1), 'seconds', round(available, 2), 'GiB free', flush=True)
                        last_report = time.monotonic()
                    time.sleep(2)
                code = process.wait()
            finally:
                alive = [child for child in known_descendants if child.is_running()]
                if process.poll() is None or alive:
                    stop_tree(process, alive)
        record.update(status='complete' if code == 0 and 'stop_reason' not in record else 'interrupted' if 'stop_reason' in record else 'failed',
                      exit_code=code, completed_utc=now().isoformat(), runtime_sec=time.monotonic() - started,
                      log_sha256=hashlib.sha256((destination / (name + '.log')).read_bytes()).hexdigest())
        write(destination / 'results.json', results)
        print('FINISH', name, record['status'], round(record['runtime_sec'], 1), flush=True)
    write(destination / 'complete.json', {'completed_utc': now().isoformat(), 'jobs': results})
    print('QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
