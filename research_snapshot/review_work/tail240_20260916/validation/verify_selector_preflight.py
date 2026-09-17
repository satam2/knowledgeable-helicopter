"""CPU-only selector provenance, label isolation and deterministic contract checks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import torch
import lightgbm
import sys
from pathlib import Path
import inspect
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT, read, sha, write, guard, object_hash
sys.path.insert(0, str(ROOT / 'review_work/tail240_20260916/forensics/current_selector/v2'))
import run_selector as subject
from verify_current_calibration import solve_simplex


def main():
    protocol = subject.declare()
    assert protocol['training']['epochs'] == 20 and protocol['training']['no_eval_set'] and protocol['training']['fixed_final_epoch']
    assert inspect.signature(subject.selector.fit).parameters['epochs'].default == 20
    assert 'late_labels' not in inspect.signature(subject.selector.fit).parameters
    assert protocol['experts'] == ['tabm_source','tabm_combined','lgb63','lgb63_sequence','tabm_ple8','lgb63_sequence8','lgb63_union','catboost_combined','xgb_combined']
    assert protocol['resources']['process_cap_bytes'] == 4*1024**3 and protocol['resources']['device'] == 'cpu'
    for relative, expected in protocol['source_hashes'].items():
        assert sha(ROOT / relative) == expected
    folds = {}
    for fold in ('F1','F3'):
        frame, saved = subject.inputs(fold, protocol)
        prior = read(ROOT / f'private_runs/tail240_20260916/validation/current_calibration_{fold}_v1/receipt.json')
        assert prior['status'] == 'passed'
        assert prior['producer_manifest_sha256'] == protocol['folds'][fold]['manifest.json']
        assert subject.TARGET not in frame.columns
        ordered = frame.sort_values([subject.old.TIME, subject.ID], kind='stable')
        cutoff = ordered.iloc[len(frame)*6//10][subject.old.TIME]
        early = frame[subject.old.TIME].lt(cutoff).to_numpy()
        late = ~early
        later_flights = set(frame.loc[late, subject.old.FID].dropna())
        purged = early & frame[subject.old.FID].notna().to_numpy() & frame[subject.old.FID].isin(later_flights).to_numpy()
        fit = early & ~purged
        for name, values in [('fit',fit),('early',early),('late',late),('purged',purged)]:
            np.testing.assert_array_equal(frame[name], values)
        pre = read(subject.OUT / fold / 'preflight.json')
        assert pre['protocol_sha256'] == sha(subject.OUT / 'protocol.json')
        assert pre['fit_ids_hash'] == object_hash(frame.loc[fit,subject.ID].tolist())
        assert pre['late_ids_hash'] == object_hash(frame.loc[late,subject.ID].tolist())
        actual_read = pd.read_parquet
        reads = []
        def checked_read(path, **kwargs):
            reads.append(kwargs)
            assert kwargs['columns'] == [subject.ID, subject.TARGET]
            assert kwargs['filters'] == [(subject.ID, 'in', frame.loc[fit,subject.ID].tolist())]
            return actual_read(path, **kwargs)
        with patch.object(subject.pd, 'read_parquet', checked_read):
            labels = subject.labels_for(fold, frame.loc[fit,subject.ID])
        assert len(reads) == 1
        p = frame[protocol['experts']].to_numpy(float)
        independent = solve_simplex(labels, p[fit])
        expected = np.asarray(read(subject.old.OUT / fold / 'weights.json')['weights'])
        np.testing.assert_allclose(independent, expected, rtol=0, atol=1e-10)
        neural = subject.old.NEURAL[fold]
        recorded_ple = pd.read_parquet(neural / 'tune_predictions.parquet', columns=[subject.ID,'prediction_sec']).set_index(subject.ID)
        np.testing.assert_array_equal(p[:,4], recorded_ple.loc[frame[subject.ID],'prediction_sec'])
        folds[fold] = dict(fit_rows=int(fit.sum()), late_rows=int(late.sum()), purged_rows=int(purged.sum()),
                           early_labels_only_filtered=True, independent_simplex_max_delta=float(np.max(np.abs(independent-expected))),
                           currentPLE387_prediction_order_exact=True, prior_verified_input_manifest_sha256=prior['producer_manifest_sha256'])
        guard()
    # Alter withheld labels while the only permitted label accessor receives early IDs.
    rng = np.random.default_rng(20260916)
    x = pd.DataFrame({'n':rng.normal(size=64), 'c':pd.Categorical(['a','b']*32)})
    p = rng.normal(1000,100,(64,9)); y = .6*p[:,0]+.4*p[:,1]
    full = pd.DataFrame({subject.ID:np.arange(80), subject.TARGET:np.r_[y,np.zeros(16)]})
    def synthetic_read(path, columns, filters):
        assert columns == [subject.ID,subject.TARGET]
        selected = filters[0][2]
        return full.loc[full[subject.ID].isin(selected), columns]
    with patch.object(subject.pd, 'read_parquet', synthetic_read):
        ids = pd.Series(np.arange(64))
        labels_a = subject.labels_for('F1',ids)
        full.loc[64:,subject.TARGET] = np.arange(16)*1e12
        labels_b = subject.labels_for('F1',ids)
    np.testing.assert_array_equal(labels_a, labels_b)
    first = subject.selector.fit(x,p,labels_a,epochs=2,batch_size=32)
    second = subject.selector.fit(x,p,labels_b,epochs=2,batch_size=32)
    for name, value in first['network'].state_dict().items():
        assert torch.equal(value, second['network'].state_dict()[name])
    prediction, weights = subject.selector.predict(first,x,p)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(1),1.,rtol=0,atol=1e-12)
    assert (prediction >= p.min(1)).all() and (prediction <= p.max(1)).all()
    assert not torch.cuda.is_initialized()
    out = ROOT / 'private_runs/tail240_20260916/validation/current_selector_preflight_v2'
    out.mkdir(parents=True,exist_ok=False)
    result = dict(status='passed_static_and_bounded_preflight',source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(subject.OUT/'protocol.json'),folds=folds,synthetic_withheld_label_mutation_same_model_state=True,
        convex_weights_and_hull_checked=True,producer_tests_passed=3,recursive_watchdog_tests_passed=2,
        no_production_model_fit=True,no_GPU_initialized=True,peak_bytes=guard(),
        scope='Productionsource/cohort/expertorder/simplexpreflight plus two small synthetic2epochfits; no new real-data selector fit.',
        limitation='Late labels are excluded from this selector optimizer, but upstream models used the same tune period for stopping. Exposed-developmentpilot only; runtime process-tree limits and saved native predictions remain to verify.')
    write(out/'receipt.json',result)
    print(result,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):
        main()
