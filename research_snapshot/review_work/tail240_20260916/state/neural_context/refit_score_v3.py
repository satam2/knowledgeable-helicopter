"""Declared full PLE387 refits and frozen ordinary-only score composition."""
import torch
import argparse
import gc
from pathlib import Path
import run_f3_v3 as memory
import joblib

frozen = memory.frozen
common, np, pd = frozen.common, frozen.np, frozen.pd
ROOT, ID, TARGET = frozen.ROOT, frozen.ID, frozen.TARGET
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/state/neural_context/refit_score_v3')
ENSEMBLE = ROOT / 'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
BASELINE = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
EPOCHS = {'F1': 20, 'F3': 19}
RESOURCES = {'F1': {'process_gib': 16, 'startup_available_gib': 24},
             'F3': {'process_gib': 20, 'startup_available_gib': 28}}
TUNE = {'F1': ROOT / 'private_runs/tail240_20260916/state/neural_context/v1/F1',
        'F3': ROOT / 'private_runs/tail240_20260916/state/neural_context/v3/F3'}
from taxiout.metrics import scores, season_score


def declare():
    protocol = memory.declare()
    baseline = common.read_json(BASELINE)
    records = {}
    for fold, folder in TUNE.items():
        marker = common.read_json(folder / 'manifest.json')
        assert marker['selected_epochs'] == EPOCHS[fold]
        assert marker['native_reload_max_delta_sec'] == 0
        assert marker['feature_columns'] == protocol['feature_columns']
        weights = common.read_json(ENSEMBLE / f'{fold}_weights.json')
        version = 'v1' if fold == 'F1' else 'v3'
        verification = ROOT / f'private_runs/tail240_20260916/validation/neural_context_{fold}_{version}_verifier_v4/receipt.json'
        verified = common.read_json(verification)
        assert verified['candidate_native_max_abs_delta_sec'] == 0
        assert verified['control_native_max_abs_delta_sec'] == 0
        records[fold] = dict(tune_manifest_sha256=common.sha256(folder / 'manifest.json'),
            baseline=baseline['folds'][fold],
            tune_independent_receipt_sha256=common.sha256(verification),
            tune_metrics_sha256=common.sha256(folder / 'metrics.json'),
            original225_manifest_sha256=common.sha256(frozen.control_folder(fold) / 'manifest.json'),
            weights_sha256=common.sha256(ENSEMBLE / f'{fold}_weights.json'),
            global9_score_manifest_sha256=common.sha256(ENSEMBLE / f'{fold}_score/manifest.json'),
            epochs=EPOCHS[fold], experts=weights['experts'], global_weights=weights['global'])
    payload = dict(created_from='ExposedJune/Octtune matchedfamilyinformationqualification',
        baseline_binding_sha256=common.sha256(BASELINE),
        prior_unexecuted_source_sha256=common.sha256(Path(__file__).with_name('refit_score.py')),
        prior_unexecuted_protocol_sha256=common.sha256(ROOT / 'private_runs/tail240_20260916/state/neural_context/refit_score_v1/protocol.json'),
        corrections='Canonicalsplit hashcomparison; fullscorebaseline is boundcurrent272 V4global9 composition, notolderglobal9fullcomposition. Ordinarypredictionssame, preserveboundV4nonordinaryroutes exactly.',
        source_sha256=common.sha256(__file__), feature_columns=protocol['feature_columns'],
        source_chain=dict(memory_wrapper=common.sha256(memory.__file__),
            grouped_bins=common.sha256(memory.prior.__file__), schema_helper=common.sha256(memory.schema_discovery.__file__),
            tune_runner=common.sha256(frozen.__file__), adapter=common.sha256(frozen.adapter.__file__),
            trainer=common.sha256(frozen.tabm_gpu.__file__), encoder=common.sha256(frozen.encoders.__file__),
            loader=common.sha256(frozen.risk.__file__)), folds=records,
        qualification='MatchedPLE225to387 seasonalrawMSEweightedRMSEgain5.882630allfinite/5.860511ordinary withpositivebothfolds/dayremovals/bootstrap. Global9replacement1.746089andblend1.805710FAIL2secmateriality; not anensemblegatepass.',
        training='OriginalfullpurgedfiniteNMrefit Jan-Jun forF1/Jan-Oct forF3, rawY-P, newrefitonlyFrameEncoder/vocab/median/standardization andgroupedofficial48bins. Fixed20/19epochs, no tuning, samearchitecture/seed/batch/optimizer. No row/label filtering beyondfiniteNM.',
        scoring='OriginalfullJuly/Novscorecohorts, allrawlabels. Nativeold225scoremodel replaybeforetraining; savednew387nativeGPUscore reloadexact. Predictallfinitescore,composeonlyordinary0<=proxy<=7200.',
        variants=dict(replacement='boundV4global9 + frozenPLEweight*(new387-old225) onordinaryonly',
            blend25='.75boundV4global9+.25new387 onordinaryonly'),
        protected='Allnonordinary rows includingmissingNM andfinitetail exactlyboundcurrentV4global9; allotherexpertweightsandpredictions unchanged. No scoreselectedweights/variants.',
        evaluation='Reportallfinite matchedfamily,ordinarymatchedfamily,andfullscoreglobal9composition; fixed seasonalMSEweights192122summer/152719winter; dayclusterbootstrap/dayremovals/topbeneficialrowdeletions diagnostic only.',
        exposure='July/November are repeatedly exposeddevelopmentmonths, not freshholdout or leaderboard. Separate refitmodels, no rankingpredictions or submission.',
        integration='Parent separatelypredeclares compositionwithapprovednormalizedmissing25 before newscorepredictionaccess; this runner reportspureordinary changes first.',
        resource_budgets=RESOURCES,
        resources='ExclusiveGPU,2CPU,8GiBhostreserve; F1process16GiB/start24GiB,F3process20GiB/start28GiB; sampledRSS+historicalpeakchecks, no OShardcap. Sequentialfolds andparentcoordination; F3awaitsmainCPUcompletion.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == payload, 'Preserve frozen refit declaration'
    else:
        common.write_json(path, payload)
    return payload


def compose(base, old, new, ordinary, weight):
    replacement, blend = base.copy(), base.copy()
    replacement[ordinary] += weight * (new[ordinary] - old[ordinary])
    blend[ordinary] = .75 * base[ordinary] + .25 * new[ordinary]
    assert np.array_equal(replacement[~ordinary], base[~ordinary])
    assert np.array_equal(blend[~ordinary], base[~ordinary])
    return replacement, blend


def run(fold):
    protocol = declare()
    budget = RESOURCES[fold]
    assert frozen.psutil.virtual_memory().available >= budget['startup_available_gib'] * 1024**3
    def guard():
        info = frozen.psutil.Process().memory_info()
        assert max(info.rss, info.peak_wset) < budget['process_gib'] * 1024**3
        assert frozen.psutil.virtual_memory().available >= 8 * 1024**3
        return int(info.rss)
    folder = OUT / fold
    folder.mkdir(exist_ok=False)
    common.write_json(folder / 'launch.json', dict(created_utc=common.utc_now(),
        protocol_sha256=common.sha256(OUT / 'protocol.json'), available_bytes=frozen.psutil.virtual_memory().available))
    guard()
    frozen.guard = guard
    frozen.risk.guard = guard
    frozen.adapter.fit_bins = memory.prior.grouped_fit_bins
    original = frozen.risk.feature_sources

    def discover(columns):
        sources, receipt = memory.schema_discovery.cached_discovery(original, columns)
        receipt['tuples'] = [[str(p), digest, fill, names] for p, digest, fill, names in sources]
        common.write_json(folder / 'discovery.json', receipt)
        return sources

    frozen.risk.feature_sources = discover
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(path) == common.read_json(ROOT / 'private_runs/screening_230/reports/data_audit.json')['artifacts'][path.name]
    meta = pd.read_parquet(path, columns=[ID, 'FLIGHT_ID_mvt', common.MOVEMENT, 'ADEP_mvt', TARGET, 'proxy_sec'])
    idx, split, _ = common.fold_data(meta, fold, full=True)
    refit = meta.iloc[idx['refit']].copy()
    refit = refit.loc[np.isfinite(refit.proxy_sec)]
    full = meta.iloc[idx['score']].copy()
    finite = np.isfinite(full.proxy_sec).to_numpy()
    score = full.loc[finite].copy()
    del meta
    gc.collect()
    original_folder = frozen.control_folder(fold)
    original_marker = common.read_json(original_folder / 'manifest.json')
    assert common.sha256(original_folder / 'manifest.json') == protocol['folds'][fold]['original225_manifest_sha256']
    assert common.object_hash(split) == common.object_hash(original_marker['split'])
    assert common.sha256(BASELINE) == protocol['baseline_binding_sha256']
    baseline = protocol['folds'][fold]['baseline']
    assert common.object_hash(split) == common.object_hash(baseline['split'])
    assert common.object_hash(full[ID].tolist()) == baseline['score_id_hash']
    assert common.object_hash(full[TARGET].tolist()) == baseline['score_target_hash']
    assert len(refit) == original_marker['refit']['rows']
    assert common.object_hash(refit[ID].tolist()) == original_marker['fit_ids']['refit']['hash']
    assert common.object_hash(score[ID].tolist()) == original_marker['fit_ids']['score']['hash']
    desired = protocol['feature_columns']
    ids = pd.Index(pd.concat([refit[ID], score[ID]], ignore_index=True))
    matrix, vocab, receipts = frozen.risk.load_matrix(ids, len(refit), desired, folder)
    x = frozen.decode_frame(matrix, vocab, desired)
    x.index = ids
    xr, xs = x.iloc[:len(refit)], x.iloc[len(refit):]
    for name in ['model.joblib', 'candidate.parquet']:
        assert common.sha256(original_folder / name) == original_marker['outputs'][name]
    old_full = pd.read_parquet(original_folder / 'candidate.parquet').set_index(ID)
    assert np.array_equal(old_full.index, full[ID])
    assert np.array_equal(old_full[TARGET], full[TARGET])
    old_model = joblib.load(original_folder / 'model.joblib')
    old_pred = frozen.adapter.predict(old_model, xs[desired[:225]]) + score.proxy_sec.to_numpy(float)
    assert np.array_equal(old_pred, old_full.loc[score[ID], 'prediction_sec'].to_numpy())
    del old_model
    gc.collect()
    common.write_json(folder / 'control_replay.json', dict(rows=len(score), native_max_abs_delta_sec=0.))
    guard()
    model, evidence = frozen.adapter.fit(xr, (refit[TARGET] - refit.proxy_sec).to_numpy(float),
        steps=EPOCHS[fold], seed=20260916, threads=2)
    guard()
    assert evidence['steps'] == EPOCHS[fold] and evidence['rows'] == len(refit)
    joblib.dump(model, folder / 'model.joblib')
    pred = frozen.adapter.predict(model, xs) + score.proxy_sec.to_numpy(float)
    native = frozen.adapter.predict(joblib.load(folder / 'model.joblib'), xs) + score.proxy_sec.to_numpy(float)
    assert np.array_equal(pred, native) and np.isfinite(pred).all()
    common.write_json(folder / 'refit_evidence.json', evidence)
    score.assign(prediction_sec=pred, control225_prediction_sec=old_pred).to_parquet(folder / 'finite_score_predictions.parquet', index=False)
    ensemble_marker_path = ENSEMBLE / f'{fold}_score/manifest.json'
    assert common.sha256(ensemble_marker_path) == protocol['folds'][fold]['global9_score_manifest_sha256']
    ensemble_marker = common.read_json(ensemble_marker_path)
    old_base_path = ENSEMBLE / f'{fold}_score/global9.parquet'
    assert common.sha256(old_base_path) == ensemble_marker['outputs']['global9.parquet']
    assert common.sha256(baseline['manifest_path']) == baseline['manifest_sha256']
    base_path = Path(baseline['prediction_path'])
    assert common.sha256(base_path) == baseline['prediction_sha256']
    base = pd.read_parquet(base_path)
    assert np.array_equal(base[ID], full[ID]) and np.array_equal(base[TARGET], full[TARGET])
    old_base = pd.read_parquet(old_base_path)
    assert np.array_equal(old_base[ID], full[ID])
    ordinary = full.proxy_sec.between(0, 7200).to_numpy()
    assert np.array_equal(base.prediction_sec.to_numpy()[ordinary], old_base.prediction_sec.to_numpy()[ordinary])
    weights = common.read_json(ENSEMBLE / f'{fold}_weights.json')
    assert common.sha256(ENSEMBLE / f'{fold}_weights.json') == protocol['folds'][fold]['weights_sha256']
    weight = weights['global'][weights['experts'].index('tabm_ple8')]
    raw = np.full(len(full), np.nan)
    raw[finite] = pred
    variants = compose(base.prediction_sec.to_numpy(), old_full.prediction_sec.to_numpy(), raw, ordinary, weight)
    reports = {}
    for name, values in zip(['replacement', 'blend25'], variants):
        result = full.copy()
        result['prediction_sec'] = values
        result.to_parquet(folder / f'{name}.parquet', index=False)
        assert np.array_equal(pd.read_parquet(folder / f'{name}.parquet').prediction_sec, values)
        reports[name] = dict(metrics=scores(full[TARGET], values),
            paired=frozen.paired(full[TARGET].to_numpy(float), values, base.prediction_sec.to_numpy(), full[common.MOVEMENT].dt.floor('D').to_numpy()),
            protected_rows=int((~ordinary).sum()), protected_exact=True)
    ordinary_finite = score.proxy_sec.between(0, 7200).to_numpy()
    for name, mask in [('matched_allfinite', np.ones(len(score), bool)), ('matched_ordinary', ordinary_finite)]:
        reports[name] = dict(metrics=scores(score[TARGET].to_numpy()[mask], pred[mask]),
            control_metrics=scores(score[TARGET].to_numpy()[mask], old_pred[mask]),
            paired=frozen.paired(score[TARGET].to_numpy(float)[mask], pred[mask], old_pred[mask], score[common.MOVEMENT].dt.floor('D').to_numpy()[mask]))
    reports['baseline'] = scores(full[TARGET], base.prediction_sec)
    common.write_json(folder / 'metrics.json', reports)
    info = frozen.psutil.Process().memory_info()
    guard()
    manifest = dict(status='complete', fold=fold, protocol_sha256=common.sha256(OUT / 'protocol.json'),
        source_sha256=common.sha256(__file__), split=split, refit_rows=len(refit), full_score_rows=len(full), finite_score_rows=len(score),
        refit_ids_hash=common.object_hash(refit[ID].tolist()), refit_label_hash=common.object_hash(refit[TARGET].tolist()),
        score_ids_hash=common.object_hash(full[ID].tolist()), score_label_hash=common.object_hash(full[TARGET].tolist()),
        feature_columns=desired, feature_receipts=receipts, fixed_epochs=EPOCHS[fold], native_replay_max_abs_delta_sec=0.,
        peak_wset_bytes=info.peak_wset, outputs={p.name: common.sha256(p) for p in folder.iterdir() if p.is_file() and p.name != 'matrix.float32'})
    common.write_json(folder / 'manifest.json', manifest)
    del xr, xs, x, matrix, model
    gc.collect()
    (folder / 'matrix.float32').unlink()
    print('COMPLETE_REFIT_SCORE', fold, {k: v['metrics']['rmse_sec'] for k, v in reports.items() if k != 'baseline'}, flush=True)


def summary():
    data = {fold: common.read_json(OUT / fold / 'metrics.json') for fold in EPOCHS}
    result = {}
    for name in ['replacement', 'blend25', 'matched_allfinite', 'matched_ordinary']:
        candidate = season_score(data['F1'][name]['metrics'], data['F3'][name]['metrics'])
        controls = [data[f][name]['control_metrics'] if name.startswith('matched_') else data[f]['baseline'] for f in EPOCHS]
        control = season_score(*controls)
        result[name] = dict(candidate_rmse=candidate, control_rmse=control, gain=control-candidate)
    common.write_json(OUT / 'summary.json', dict(status='complete', seasonal=result, exposure='Previously exposed July/November development; not fresh holdout.'))
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    elif args.summary:
        summary()
    else:
        assert args.fold
        frozen.pa.set_cpu_count(2)
        frozen.pa.set_io_thread_count(1)
        with frozen.threadpool_limits(2):
            run(args.fold)
