"""Queue broader residual checks behind the runway screening lane."""

import argparse
import subprocess
import sys
import psutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int, required=True)
    args = parser.parse_args()
    try:
        process = psutil.Process(args.after_pid)
        if not any('next230_gpu_queue.py' in part for part in process.cmdline()):
            raise ValueError('Wait target is not the runway queue')
        print(f'Broader GPU validation waits for runway queue PID {process.pid}', flush=True)
        while process.is_running():
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired:
                continue
            break
    except psutil.NoSuchProcess:
        pass
    subprocess.run([sys.executable, '-u', 'review_work/next230_train.py', '--candidates',
                    'capacity_d8_5000', '--folds', 'F2', 'G1'], check=True)
