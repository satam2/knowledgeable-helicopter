"""Bind complete matched experiments, replay saved models, then independently score."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import validate_candidate as evaluation

ROOT = evaluation.ROOT
read_json, write_json, sha256, object_hash = evaluation.read_json, evaluation.write_json, evaluation.sha256, evaluation.object_hash
ID = evaluation.ID


def load_module(name, folder, filename):
    sys.path.insert(0, str(ROOT/folder))
    spec = importlib.util.spec_from_file_location(name, ROOT/folder/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['state', 'lexical', 'opdi'], required=True)
    args = parser.parse_args()
    family = args.experiment
    out = ROOT/f'private_runs/tail240_20260916/validation/{family}_postfit_v1'
    out.mkdir(parents=True, exist_ok=False)
    binding = read_json(evaluation.BINDING)
    if family == 'state':
        runner = load_module('state_validation_runner', 'review_work/tail240_20260916/state', 'run_missing.py')
        root = ROOT/'private_runs/tail240_20260916/state/models_v2'
        arms = ['control', 'both12']
        x, meta = runner.load_missing()
        base = runner.base
        declaration = read_json(root/'protocol.json')
        sources = declaration['source_hashes']
        wrapper = ROOT/'review_work/tail240_20260916/state/run_missing_v2.py'
        assert sha256(wrapper) == read_json(root/'execution_wrapper.json')['wrapper_sha256']
        sources[str(wrapper.relative_to(ROOT))] = sha256(wrapper)
        oldroot = ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models'
        path_for = lambda arm, fold: root/arm/fold
    elif family == 'opdi':
        runner = load_module('opdi_validation_runner', 'review_work/tail240_20260916/models', 'run_opdi_dates_missing.py')
        root = runner.OUT
        arms = ['control', 'opdi45']
        x, meta = runner.shared.load_missing()
        cache = read_json(runner.CACHE/'manifest.json')
        assert sha256(runner.CACHE/'features.parquet') == cache['outputs']['features.parquet']
        oracle = read_json(ROOT/'private_runs/tail240_20260916/validation/opdi_dates_oracle_v3/receipt.json')
        assert oracle['producer_manifest_sha256'] == sha256(runner.CACHE/'manifest.json')
        opdi_extra = pd.read_parquet(runner.CACHE/'features.parquet').set_index(ID)
        assert x.index.isin(opdi_extra.index).all() and opdi_extra.index.is_unique
        base = runner.base
        declaration = read_json(root/'protocol.json')
        sources = declaration['source_hashes']
        oldroot = ROOT/'private_runs/breakthrough_20260916/missing/id_context_v1/models'
        path_for = lambda arm, fold: runner.CONTROL/fold if arm == 'control' else root/fold
    else:
        runner = load_module('lexical_validation_runner', 'review_work/tail240_20260916/forensics', 'run_lexical.py')
        root = ROOT/'private_runs/tail240_20260916/forensics/models/lexical_v1'
        arms = ['control', 'lexical']
        completion = read_json(root/'completion.json')
        assert completion['status'] == 'complete'
        assert sha256(root/'all_missing_features.parquet') == completion['feature_sha256']
        x, meta = runner.prior.load_data()
        extra = runner.lexical_features(runner.decode_flight_tokens(x.flight))
        cached = pd.read_parquet(root/'all_missing_features.parquet').set_index(ID)
        pd.testing.assert_frame_equal(pd.concat([x, extra], axis=1), cached)
        base = runner.prior
        declaration = read_json(root/'protocol.json')['declaration']
        sources = declaration['source_hashes']
        oldroot = ROOT/'private_runs/breakthrough_20260916/missing/models'
        path_for = lambda arm, fold: root/arm/'models'/f'historical_template_{fold}_s20260916'
    for filename, expected in sources.items():
        assert sha256(ROOT/filename) == expected
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    records, predictions, replay = {}, {}, {}
    for arm in arms:
        for fold in ('F1','F3'):
            evaluation.guard()
            dest = path_for(arm, fold)
            record = read_json(dest/'manifest.json')
            assert record['status'] == 'complete'
            for name, digest in record['outputs'].items():
                assert sha256(dest/name) == digest
            assert record['fit_ids'] == binding['folds'][fold]['cohorts']['missing_nm']
            assert object_hash(record['split']) == binding['folds'][fold]['split_hash']
            indices, _, _ = base.common.fold_data(meta, fold, full=True)
            selected = indices['score'][missing[indices['score']]]
            ids = meta.iloc[selected][ID]
            features = x.loc[ids].copy()
            if family == 'lexical' and arm == 'lexical':
                features = pd.concat([features, extra.loc[ids]], axis=1)
            if family == 'opdi' and arm == 'opdi45':
                features = pd.concat([features, opdi_extra.loc[ids]], axis=1)
            if family == 'state' and arm == 'both12':
                cache = read_json(runner.CACHE/'manifest.json')
                part = cache['folds'][fold]['stages']['score']
                assert sha256(ROOT/part['path']) == part['sha256']
                additional = pd.read_parquet(ROOT/part['path']).set_index(ID)
                np.testing.assert_array_equal(additional.index, meta.iloc[indices['score']][ID])
                features = pd.concat([features, additional.loc[ids]], axis=1)
            expected_columns = record.get('features_used', record.get('feature_columns'))
            assert list(features) == expected_columns
            times = pd.Series(pd.to_datetime(meta.iloc[selected]['MVT_TIME_UTC_mvt'], utc=True).to_numpy(), index=features.index)
            model = joblib.load(dest/'model.joblib')
            values = base.predict_arm(model, features, times)
            frame = pd.read_parquet(dest/'candidate.parquet')
            np.testing.assert_array_equal(frame[ID], meta.iloc[indices['score']][ID])
            np.testing.assert_array_equal(frame[evaluation.TARGET], meta.iloc[indices['score']][evaluation.TARGET])
            active = missing[indices['score']]
            delta = float(np.max(np.abs(values-frame.prediction_sec.to_numpy(float)[active])))
            assert delta <= 1e-9
            evidence = {'native_saved_model_replay_max_abs_delta': delta, 'rows': len(values), 'manifest_sha256':sha256(dest/'manifest.json')}
            if arm == 'control':
                old = oldroot/f'historical_template_{fold}_s20260916'
                oldrecord = read_json(old/'manifest.json')
                for name in ('candidate.parquet','tune_predictions.parquet'):
                    assert sha256(old/name) == oldrecord['outputs'][name]
                prior = pd.read_parquet(old/'candidate.parquet')
                np.testing.assert_array_equal(prior[ID], frame[ID])
                previous_delta = float(np.max(np.abs(prior.prediction_sec.to_numpy(float)[active]-values)))
                assert previous_delta <= 1e-9
                before, now = pd.read_parquet(old/'tune_predictions.parquet'), pd.read_parquet(dest/'tune_predictions.parquet')
                pd.testing.assert_frame_equal(before, now, check_exact=True)
                evidence.update(original_control_score_max_abs_delta=previous_delta, original_control_tune_exact=True)
            replay[f'{arm}_{fold}'] = evidence
            records[arm,fold] = record
            for variant in ('candidate','blend25'):
                pred = pd.read_parquet(dest/f'{variant}.parquet')
                np.testing.assert_array_equal(pred[ID], frame[ID])
                np.testing.assert_array_equal(pred[evaluation.TARGET], frame[evaluation.TARGET])
                predictions[arm,fold,variant] = pred[[ID,'prediction_sec']]
            del model, features
    write_json(out/'replay.json', {'status':'passed','source_sha256':sha256(__file__),'models':replay,'peak_rss_bytes':evaluation.guard()})
    for arm in arms:
        for variant in ('candidate','blend25'):
            package = out/f'{arm}_{variant}_exchange'
            package.mkdir()
            protocol = {'name':f'{family}_{arm}_{variant}','baseline_binding_sha256':sha256(evaluation.BINDING),
                'score_labels_used_for_selection':False,'routing_uses_score_targets':False,'selection_periods':['fit','tune'],
                'inference_columns':records[arm,'F1'].get('features_used',records[arm,'F1'].get('feature_columns')),
                'prediction_transform':'raw_unclipped','tail_definition':'None; original full missing-source cohort',
                'routing_definition':'Missing observable NM proxy only; finite-source predictions exact frozen global9',
                'selection_rule':'Predeclared600/depth5 CatBoost HistoricalTemplate; tune early stopping; fixed100 or25 percent replacement; report every variant',
                'development_status':'Exposed adaptive development; no independent temporal generalization claim',
                'producer_protocol_sha256':sha256(root/'protocol.json')}
            write_json(package/'protocol.json',protocol)
            manifest = {'status':'complete','protocol_sha256':sha256(package/'protocol.json'),'source_hashes':sources,'folds':{}}
            for fold in ('F1','F3'):
                predpath = package/f'{fold}.parquet'
                controlpath = package/f'{fold}_control.parquet'
                predictions[arm,fold,variant].to_parquet(predpath,index=False)
                predictions['control',fold,variant].to_parquet(controlpath,index=False)
                manifest['folds'][fold] = {'prediction':{'path':predpath.name,'sha256':sha256(predpath)},
                    'matched_control':{'path':controlpath.name,'sha256':sha256(controlpath)},
                    'split_hash':binding['folds'][fold]['split_hash'],'training_scope':'missing_nm',
                    'cohorts':records[arm,fold]['fit_ids'],'prediction_scope':'missing_nm'}
            write_json(package/'manifest.json',manifest)
            subprocess.run([sys.executable,'-B','-u',str(Path(evaluation.__file__)), '--candidate',str(package),
                '--output',str(out/f'{arm}_{variant}_evaluation')],check=True)
    print('EXPERIMENT_VERIFIED',family,flush=True)


if __name__ == '__main__':
    main()
