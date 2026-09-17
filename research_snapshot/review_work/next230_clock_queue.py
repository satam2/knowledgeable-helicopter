"""Run each clock calibration as soon as its two independent experts complete."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def complete(path):
    return path.exists() and json.loads(path.read_text())['status']=='complete'


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'private_runs/next_230'
    pending=['F1','F3','F2','G1']
    while pending:
        progressed=False
        for fold in list(pending):
            if not complete(root/'clock_cpu_experts'/fold/'expert.json'):
                continue
            if not complete(root/'models'/f'capacity_d8_5000_{fold}_s20260910'/'manifest.json'):
                continue
            print(f'CLOCK DEPENDENCIES READY {fold}',flush=True)
            subprocess.run([sys.executable,'-u','review_work/next230_clock_cpu.py','calibrate','--folds',fold],check=True)
            subprocess.run([sys.executable,'-u','review_work/next230_clock_gate.py','--folds',fold],check=True)
            pending.remove(fold)
            progressed=True
        if pending and not progressed:
            time.sleep(30)
    print('ALL FOUR CLOCK FOLDS COMPLETE',flush=True)
