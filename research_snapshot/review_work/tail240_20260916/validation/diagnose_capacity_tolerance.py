"""CPU-only native tensor identity after a preserved small-batch replay failure."""
import verify_capacity_replay as frozen
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def main():
    out = frozen.OUT / 'tolerance_diagnosis.json'
    assert not out.exists()
    failure = frozen.read(frozen.OUT / 'gpu_receipt.json')
    assert failure['status'] == 'numerical_tolerance_failed'
    marker = frozen.read(frozen.MODEL / 'manifest.json')
    model = joblib.load(frozen.MODEL / 'fit_model.joblib')
    assert frozen.sha(frozen.MODEL / 'fit_model.joblib') == failure['model_sha256']
    encoder = model['encoder']
    vocab = frozen.read(frozen.ROOT / 'private_runs/tail240_20260916/models/linear_finite_tune_v1/F1/encoder.json')['vocab']
    nfit, ntune = marker['fit_rows'], marker['tune_rows']
    matrix = np.memmap(frozen.MODEL / 'matrix.float32', mode='r', dtype='float32', shape=(nfit+ntune, 387), order='F')
    rebuilt = np.load(frozen.OUT / 'numbers.npy', mmap_mode='r')
    categories = np.load(frozen.OUT / 'categories.npy', mmap_mode='r')
    for start in range(0, ntune, 8192):
        stop = min(start+8192, ntune)
        frame = frozen.producer.context.decode_frame(matrix[nfit+start:nfit+stop], vocab, encoder.columns)
        numbers, cats = encoder.transform(frame)
        np.testing.assert_array_equal(numbers, rebuilt[start:stop])
        np.testing.assert_array_equal(cats, categories[start:stop])
        frozen.guard()
    prediction = np.load(frozen.OUT / 'independent_predictions.npy')
    saved = pd.read_parquet(frozen.MODEL / 'tune_predictions.parquet')
    delta = np.abs(prediction-saved.prediction_sec.to_numpy(float))
    normalized = delta/model['y_scale']
    affected = delta > failure['atol_sec']
    result = dict(status='cpu_tensor_identity_passed_replay_tolerance_still_failed',
                  source_sha256=frozen.sha(Path(__file__)), failure_receipt_sha256=frozen.sha(frozen.OUT / 'gpu_receipt.json'),
                  native_encoder_numeric_and_category_tensors_exact_all_rows=True, rows=ntune,
                  rows_over_declared_tolerance=int(affected.sum()), max_abs_sec=float(delta.max()),
                  max_abs_normalized_network_output=float(normalized.max()),
                  abs_sec_quantiles={str(q):float(np.quantile(delta,q)) for q in [.5,.9,.99,.999,1.]},
                  saved_rmse=float(np.sqrt(np.mean((saved.prediction_sec-saved[frozen.TARGET])**2))),
                  independent_rmse=float(np.sqrt(np.mean((prediction-saved[frozen.TARGET])**2))),
                  peak_bytes=frozen.guard(), no_GPU=True,
                  interpretation='Inputs and network/state identity are verified. Difference is consistent with batch-dependent float32 operations but original declared tolerance remains failed. No larger batch or widened tolerance was run.')
    frozen.write(out, result)
    print(result, flush=True)


if __name__ == '__main__':
    frozen.torch.set_num_threads(1)
    with frozen.threadpool_limits(1):
        main()
