"""Windows retry: load torch DLLs before numpy/pandas; retain first attempt."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import torch
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name('supplement.py')), run_name='__main__')
