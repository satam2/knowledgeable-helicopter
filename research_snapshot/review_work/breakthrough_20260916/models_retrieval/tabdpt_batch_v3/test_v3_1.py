"""Exact context-hook restoration on success and failure, CPU mock only."""
import torch
import numpy as np
import pandas as pd
import run_v3_1 as runner


class Estimator:
    def _get_context_indices(self, x, context_size, seed=None):
        return np.tile(np.arange(context_size), (len(x), 1))


def main():
    query = pd.DataFrame({'value': [1., 2., 3.]})
    original_predict = runner.adapter.predict_batch
    for overridden in [False, True]:
        for failing in [False, True]:
            estimator = Estimator()
            if overridden:
                estimator._get_context_indices = lambda x, context_size, seed=None: np.zeros((len(x), context_size), dtype=int)
            previous = estimator.__dict__.copy()
            def mock_predict(model, x, batch):
                model['estimator']._get_context_indices(x, context_size=4, seed=20260916)
                if failing:
                    raise RuntimeError('Injected prediction failure')
                return np.arange(len(x), dtype=float)
            runner.adapter.predict_batch = mock_predict
            try:
                result, contexts = runner.capture({'estimator': estimator}, query, 16)
                assert not failing
                assert contexts.shape == (3, 4)
                np.testing.assert_array_equal(result, np.arange(3))
            except RuntimeError:
                assert failing
            assert estimator.__dict__ == previous
    runner.adapter.predict_batch = original_predict
    assert not torch.cuda.is_initialized()
    print('TESTS_PASS context-hook exactrestoration with/withoutprioroverride andonpredictionfailure; CUDAuninitialized', flush=True)


if __name__ == '__main__':
    main()
