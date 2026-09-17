"""One-tree refit runtime probe; no scoring predictions or model selection."""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--order', choices=['prior_first', 'numpy_first'], required=True)
args = parser.parse_args()
if args.order == 'numpy_first':
    import joblib
    import numpy
    import pandas
import normalized_missing_tune as prior
import psutil
import numpy as np

x, meta = prior.shared.load_missing()
idx, split, _ = prior.common.fold_data(meta, 'F1', full=True)
rows = idx['refit'][~np.isfinite(meta.proxy_sec.to_numpy(float))[idx['refit']]]
frame = x.loc[meta.iloc[rows][prior.ID]]
encoder = prior.FrameEncoder().fit(frame)
matrix = encoder.transform(frame)
target = prior.transformed(meta.iloc[rows][prior.TARGET].to_numpy(float), np.ones(len(rows)))
assert matrix.index.equals(frame.index) and len(target) == len(matrix)
assert target.dtype == np.float64 and target.flags.c_contiguous and np.isfinite(target).all()
print('INPUT', args.order, matrix.shape, target.dtype, float(target.min()), float(target.max()),
      'availableGiB', psutil.virtual_memory().available / 1024**3, flush=True)
dataset = prior.lgb.Dataset(matrix, label=target, weight=np.ones(len(rows)), categorical_feature=list(encoder.categories))
model = prior.lgb.train(prior.PARAMS, dataset, num_boost_round=1)
print('PASS', args.order, model.current_iteration(), flush=True)
