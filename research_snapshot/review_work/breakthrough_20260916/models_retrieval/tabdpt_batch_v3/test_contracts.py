"""CPU mock tests of public batch knob; no pretrained estimator or CUDA calls."""
import torch
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter


class Processor:
    def transform(self, x):
        return x.to_numpy(np.float32)


class Estimator:
    use_flash = False
    def __init__(self):
        self.model = self
        self.calls = []
    def predict(self, x, **kwargs):
        self.calls.append(kwargs)
        return x[:, 0] + 3 * x[:, 1]


def main():
    x = pd.DataFrame({'a': np.arange(4099), 'b': np.arange(4099) - 3000})
    estimator = Estimator()
    model = {'estimator': estimator, 'processor': Processor(), 'threads': 2,
        'seed': 20260916, 'context_size': 256, 'query_batch_size': 64}
    results = [adapter.predict_batch(model, x, b) for b in [16, 64, 128]]
    for result in results:
        np.testing.assert_array_equal(result, x.a + 3*x.b)
    assert len(estimator.calls) == 9
    for i, batch in enumerate([16, 64, 128]):
        for call in estimator.calls[i*3:(i+1)*3]:
            assert call == {'context_size': 256, 'batch_size': batch, 'n_ensembles': 1, 'seed': 20260916}
    np.testing.assert_array_equal(adapter.predict(model, x), results[1])
    assert len(adapter.predict_batch(model, x.iloc[:0], 16)) == 0
    try:
        adapter.predict_batch(model, x, 32)
        raise AssertionError('Undeclaredbatch accepted')
    except ValueError:
        pass
    assert adapter.REFERENCE_LIMIT == 32000 and not torch.cuda.is_initialized()
    print('TESTS_PASS unchangedcontexts/ensembles/seed;2048outerchunks;batch16/64/128;finite/fullrowoutput;GPUuninitialized', flush=True)


if __name__ == '__main__':
    main()
