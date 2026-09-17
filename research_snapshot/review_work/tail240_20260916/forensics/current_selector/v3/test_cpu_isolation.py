import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v1'))
import selector
import numpy as np
import pandas as pd
import torch
import unittest


class CpuIsolationTests(unittest.TestCase):
    def test_training_does_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(torch.cuda.is_available())
        rng = np.random.default_rng(20260916)
        features = pd.DataFrame({'x':rng.normal(size=32)})
        predictions = rng.normal(1000, 100, (32,9))
        selector.fit(features, predictions, predictions[:,0]+3, epochs=2, batch_size=16)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
