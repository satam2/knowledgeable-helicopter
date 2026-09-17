"""Run GPU seed repeats after the independent validation queue completes."""

import argparse
import subprocess
import sys
import psutil


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--after-pid',type=int,required=True)
    args=p.parse_args()
    try:
        process=psutil.Process(args.after_pid)
        if not any('next230_validation_queue.py' in part for part in process.cmdline()):
            raise ValueError('Wait target is not the validation queue')
        print(f'Main seed queue waits for validation PID {process.pid}',flush=True)
        while process.is_running():
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired:
                continue
            break
    except psutil.NoSuchProcess:
        pass
    subprocess.run([sys.executable,'-u','review_work/next230_residual_seeds.py'],check=True)
