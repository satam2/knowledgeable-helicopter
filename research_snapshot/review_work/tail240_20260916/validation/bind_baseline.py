"""Bind the continuation baseline, original contracts, and initial system state."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'review_work/campaign_20260916'))
import common
from taxiout.artifacts import object_hash, read_json, write_json, sha256, utc_now
from taxiout.schema import ID, FLIGHT_ID, MOVEMENT, TARGET

OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/validation')
BASE = ROOT / 'private_runs/breakthrough_20260916/missing/route_composition_v4'
WEIGHTS = {'F1': 192122 / 344841, 'F3': 152719 / 344841}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'baseline_binding.json').exists()
    prior = read_json(ROOT / 'private_runs/breakthrough_20260916/preservation_final_1402/receipt.json')
    now = read_json(OUT / 'preservation_start/receipt.json')
    assert now['status'] == 'passed' and now['raw_hashes'] == prior['raw_hashes']
    assert now['checkouts'] == prior['checkouts'], 'Record changed checkout state before proceeding'
    for prefix in ('review_work', 'private_runs', 'output'):
        common.external_path(ROOT / prefix / 'tail240_20260916')
    meta_path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit = read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')
    assert sha256(meta_path) == audit['artifacts'][meta_path.name]
    meta = pd.read_parquet(meta_path, columns=[ID, FLIGHT_ID, MOVEMENT, TARGET, 'proxy_sec'])
    verification = read_json(BASE / 'verification.json')
    assert verification['status'] == 'passed'
    folds = {}
    for fold in WEIGHTS:
        idx, split, _ = common.fold_data(meta, fold, full=True)
        folder = BASE / 'global9' / fold
        manifest = read_json(folder / 'manifest.json')
        prediction = folder / 'candidate.parquet'
        assert manifest['status'] == 'complete'
        receipt = verification['folds'][fold]['global9']
        assert sha256(folder / 'manifest.json') == receipt['manifest_sha256']
        assert sha256(prediction) == manifest['outputs']['candidate.parquet'] == receipt['candidate_sha256']
        ref, _ = common.reference(fold)
        frame = pd.read_parquet(prediction)
        np.testing.assert_array_equal(frame[ID], ref[ID])
        np.testing.assert_array_equal(frame[TARGET], ref[TARGET])
        np.testing.assert_array_equal(frame[ID], meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(frame[TARGET], meta.iloc[idx['score']][TARGET])
        assert object_hash(manifest['split']) == object_hash(split)
        y = frame[TARGET].to_numpy(float)
        p = frame.prediction_sec.to_numpy(float)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        rmse = float(np.sqrt(np.mean((p-y)**2)))
        scopes = {'all': np.ones(len(meta), bool), 'missing_nm': ~np.isfinite(meta.proxy_sec),
            'finite_nm': np.isfinite(meta.proxy_sec), 'ordinary_nm': np.isfinite(meta.proxy_sec) & meta.proxy_sec.between(0,7200)}
        cohorts = {}
        for scope, eligible in scopes.items():
            eligible = np.asarray(eligible)
            cohorts[scope] = {stage: {'n': int(eligible[rows].sum()), 'hash': object_hash(meta.iloc[rows[eligible[rows]]][ID].tolist())}
                              for stage, rows in idx.items()}
        folds[fold] = {'prediction_path': str(prediction), 'prediction_sha256': sha256(prediction),
            'manifest_path': str(folder / 'manifest.json'), 'manifest_sha256': sha256(folder / 'manifest.json'),
            'score_rows': len(frame), 'score_id_hash': object_hash(frame[ID].tolist()),
            'score_target_hash': object_hash(frame[TARGET].tolist()), 'rmse': rmse,
            'split': split, 'split_hash': object_hash(split), 'cohorts': cohorts}
    seasonal = float(np.sqrt(sum(WEIGHTS[f] * folds[f]['rmse']**2 for f in WEIGHTS)))
    assert abs(seasonal - 272.2639669175017) < 1e-9
    observers = {os.getpid(), os.getppid()}
    processes, inaccessible = [], []
    for process in psutil.process_iter(['pid', 'ppid', 'name', 'cmdline']):
        try:
            if 'python' in (process.info['name'] or '').lower() and process.pid not in observers:
                processes.append(process.info)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            inaccessible.append(process.pid)
    result = {'status': 'passed', 'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'baseline_name': 'Frozen V4 global9 complete composition', 'seasonal_rmse': seasonal,
        'weights': WEIGHTS, 'folds': folds, 'metadata_sha256': sha256(meta_path),
        'baseline_verification_sha256': sha256(BASE / 'verification.json'),
        'preservation_receipt_sha256': sha256(OUT / 'preservation_start/receipt.json'),
        'prior_preservation_receipt_sha256': sha256(ROOT / 'private_runs/breakthrough_20260916/preservation_final_1402/receipt.json'),
        'git_state_matches_prior': True, 'python_processes_other_than_observer': processes,
        'process_query_inaccessible_pids': inaccessible,
        'training_job_observation': 'No other Python processes' if not processes else 'Listed processes require coordinator review; no process was stopped',
        'limits': 'Original folds are development-exposed; baseline is local only. New scores cannot establish ranking transfer or a 240-second guarantee.'}
    write_json(OUT / 'baseline_binding.json', result)
    print('BASELINE_BOUND', seasonal, 'other_python_processes', len(processes), flush=True)


if __name__ == '__main__':
    main()
