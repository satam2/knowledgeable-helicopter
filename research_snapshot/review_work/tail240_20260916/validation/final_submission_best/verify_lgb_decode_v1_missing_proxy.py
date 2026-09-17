"""Native full-June LGB225 replay through final-release matrix decoding."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import lightgbm
from preflight import ROOT, OUT, ID, read, sha
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits


def main():
    dest = OUT / 'lgb225_decoder_receipt.json'
    assert not dest.exists()
    path = ROOT / 'review_work/tail240_20260916/final_ordinary/run.py'
    spec = importlib.util.spec_from_file_location('final_ordinary_review', path)
    subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)
    assert subject.ROOT == ROOT
    matrix_folder = ROOT / 'private_runs/tail240_20260916/forensics/source_contract/v2'
    matrix_receipt = read(ROOT / 'private_runs/tail240_20260916/validation/source_contract_v2/receipt.json')
    assert matrix_receipt['status'] == 'passed'
    assert sha(matrix_folder / 'matrix.float32') == matrix_receipt['unchanged_matrix_byte_hash']
    encoding_folder = ROOT / 'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    marker = read(encoding_folder / 'manifest.json')
    assert sha(encoding_folder / 'encoder.json') == marker['outputs']['encoder.json']
    encoding = read(encoding_folder / 'encoder.json')
    columns = read(matrix_folder / 'protocol.json')['columns']
    assert encoding['columns'] == columns
    matrix = np.memmap(matrix_folder / 'matrix.float32', dtype='float32', mode='r', shape=(996700,387), order='F')
    folder = subject.folder('lgb225', 'F1')
    manifest = read(folder / 'manifest.json')
    assert manifest['feature_columns'] == columns[:225]
    for name in ('fit_model.joblib', 'tune_predictions.parquet'):
        assert sha(folder / name) == manifest['outputs'][name]
    model = joblib.load(folder / 'fit_model.joblib')
    expected = pd.read_parquet(folder / 'tune_predictions.parquet')
    assert len(expected) == 180640
    prediction = np.empty(len(expected))
    for start in range(0,len(expected),8192):
        stop = min(start+8192,len(expected))
        frame = subject.decode(matrix[816060+start:816060+stop,:225],encoding['vocab'],columns[:225])
        prediction[start:stop] = model['estimator'].predict(model['encoder'].transform(frame), num_iteration=model['steps'], num_threads=1) + expected.proxy_sec.iloc[start:stop].to_numpy(float)
    np.testing.assert_array_equal(prediction, expected.prediction_sec)
    # Missing base categories remain null; added categories retain explicit MISSING.
    artificial = np.array([[0,0,-999999],[1,1,np.nan],[2,2,4]],dtype='float32')
    decoded = subject.decode(artificial, {'ADEP_mvt':['A'], 'conv_clock_rank_signature':['K']}, ['ADEP_mvt','conv_clock_rank_signature','n'])
    assert pd.isna(decoded.ADEP_mvt.iloc[0]) and decoded.conv_clock_rank_signature.iloc[0] == 'MISSING'
    assert decoded.ADEP_mvt.iloc[1] == '__UNSEEN_CONTEXT_VALUE__'
    encoded = subject.adapter_for('lgb225').NativeFrameEncoder().fit(decoded.iloc[[0,2]]).transform(decoded)
    assert encoded.n.iloc[0] == -999999 and np.isnan(encoded.n.iloc[1])
    assert encoded.ADEP_mvt.astype(int).tolist() == [0,1,2]
    w = read(OUT / 'preflight_receipt.json')['proposed_weights']
    shift = max(abs(w['original'][key] - w['final'][key]) for key in w['final'])
    receipt = dict(status='passed', rows=180640, features=225, native_max_abs_delta_sec=0.,
        decoder_source_sha256=sha(path), verifier_sha256=sha(__file__), inherited_matrix_sha256=matrix_receipt['unchanged_matrix_byte_hash'],
        model_manifest_sha256=sha(folder / 'manifest.json'),missing_unknown_sentinel_checks=True,
        final_coefficient_max_abs_shift=shift, fitting_used=False, gpu_used=False,
        peak_bytes=psutil.Process().memory_info().peak_wset,
        scope='Exact original LGB225 fit-model replay through final decode; 449 numeric extension uses same decode with no new categorical fields.')
    dest.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
