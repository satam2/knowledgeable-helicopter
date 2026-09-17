"""Serialize independent GPU jobs after an identified existing training process."""

import argparse
import subprocess
import sys

import psutil


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--after-pid', type=int, required=True)
    args = p.parse_args()
    try:
        process = psutil.Process(args.after_pid)
        command = process.cmdline()
        if not any('next230_train.py' in part for part in command):
            raise ValueError('Wait target is not the expected GPU training process')
        print(f'GPU queue waits for training PID {process.pid}', flush=True)
        while process.is_running():
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired:
                continue
            break
    except psutil.NoSuchProcess:
        pass
    print('GPU lane available: starting runway screens', flush=True)
    subprocess.run([sys.executable, '-u', 'review_work/next230_train.py',
                    '--candidates', 'runway_d8_2500', '--folds', 'F1', 'F3'], check=True)
