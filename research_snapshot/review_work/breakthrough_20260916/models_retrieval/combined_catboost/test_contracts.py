"""Small CPU-only contracts; no private data or CUDA initialization."""
import lightgbm
from pathlib import Path
import tempfile
import sys
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adapter


def main():
    rng = np.random.default_rng(20260916)
    frame = pd.DataFrame({'category': pd.Categorical(['a', 'b', None, 'a'] * 100),
        'number': rng.normal(size=400), 'negative_long_proxy': np.linspace(-9000, 90000, 400)})
    frame.loc[0, 'number'] = -999999
    frame.loc[1, 'number'] = np.inf
    y = rng.normal(size=400) * 200 + np.arange(400) * 2
    y[2] = -1e5
    y[3] = 1e5
    tune = pd.DataFrame({'category': pd.Categorical(['new', None, 'a'] * 30),
        'number': np.linspace(-1, 1, 90), 'negative_long_proxy': np.linspace(-5000, 100000, 90)})
    ty = np.linspace(-300, 300, 90)
    model, evidence = adapter.fit(frame, y, (tune, ty), device='CPU', threads=2, max_iterations=24)
    assert evidence['rows'] == 400 and evidence['tune_rows'] == 90
    assert 'new' not in model['encoder'].categories['category']
    transformed = model['encoder'].transform(tune)
    assert transformed.category.iloc[0] == 1 and transformed.category.iloc[1] == 0
    train = model['encoder'].transform(frame)
    assert np.isnan(train.number.iloc[0]) and np.isnan(train.number.iloc[1])
    assert train.negative_long_proxy.min() == -9000 and train.negative_long_proxy.max() == 90000
    refit_frame = pd.concat([frame, tune], ignore_index=True)
    refit_frame['category'] = refit_frame['category'].astype('category')
    final, final_evidence = adapter.fit(refit_frame, np.concatenate([y, ty]), steps=evidence['steps'], device='CPU', threads=2)
    assert final_evidence['rows'] == 490 and final_evidence['steps'] == evidence['steps']
    assert 'new' in final['encoder'].categories['category']
    expected = adapter.predict(final, tune)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'model.joblib'
        joblib.dump(final, path)
        np.testing.assert_array_equal(expected, adapter.predict(joblib.load(path), tune))
    for kwargs in ({'steps': 0}, {'steps': 5, 'tuning': (tune, ty)}, {}):
        try:
            adapter.fit(frame, y, device='CPU', **kwargs)
            raise AssertionError('Invalid contract accepted')
        except ValueError:
            pass
    try:
        adapter.predict(final, tune[['number', 'category', 'negative_long_proxy']])
        raise AssertionError('Wrong schema accepted')
    except ValueError:
        pass
    assert adapter.parameters()['iterations'] == 5000 and adapter.parameters()['task_type'] == 'GPU'
    print('TESTS_PASS fit-only vocabulary; missing/unknown; finite extreme labels retained; numeric missing conversion; freshrefit fixedcount; exact savedreplay; schema/input rejection; frozen GPUparams', flush=True)


if __name__ == '__main__':
    main()
