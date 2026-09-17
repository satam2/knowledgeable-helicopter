"""Freeze calibration and predictions before a separate evaluation-label process."""
import argparse
from pathlib import Path
import time
import train
from mixture import fit_mixture, predict_mixture
import numpy as np
import pandas as pd

common, ROOT, ID, TARGET = train.common, train.ROOT, train.ID, train.TARGET
OUT = train.OUT / 'ensemble'
VARIANTS = ['equal', 'simplex', 'shrunk']


def declare():
    training = train.declare()
    record = {'training_protocol_sha256': common.sha256(train.OUT/'protocol.json'),
              'source_sha256': common.sha256(__file__),
              'mixture_sha256': common.sha256(Path(__file__).with_name('mixture.py')),
              'evaluation': training['evaluation'], 'families': train.FAMILIES,
              'ordinary_bounds': [0, 7200], 'variants': VARIANTS,
              'historical_hybrid': 'Replace only ordinary original V3 July/November predictions; preserved routes use historical later-stopped fits, so not the strict chronological endpoint.',
              'no_promotion_from_hybrid_alone': True}
    common.external_path(OUT).mkdir(parents=True, exist_ok=True)
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def experts(fold, stage):
    arrays, reference, bindings = [], None, {}
    for family in train.FAMILIES:
        folder = train.OUT/fold/family/'refit'
        marker = common.read_json(folder/'manifest.json')
        assert marker['status'] == 'complete'
        assert marker['protocol_sha256'] == common.sha256(train.OUT/'protocol.json')
        path = folder/f'{stage}.parquet'
        assert common.sha256(path) == marker['predictions'][stage]['sha256']
        frame = pd.read_parquet(path)
        assert TARGET not in frame
        if reference is None:
            reference = frame.drop(columns='prediction_sec')
        else:
            pd.testing.assert_frame_equal(reference, frame.drop(columns='prediction_sec'))
        arrays.append(frame.prediction_sec.to_numpy(float))
        bindings[family] = {'manifest_sha256': common.sha256(folder/'manifest.json'),
                            'prediction_sha256': common.sha256(path)}
    return reference, np.column_stack(arrays), bindings


def calibrate(fold):
    declare()
    folder = OUT/fold/'calibration'
    folder.mkdir(parents=True, exist_ok=False)
    meta, p, bindings = experts(fold, 'calibration')
    ordinary = meta.proxy_sec.between(0, 7200).to_numpy()
    y = train.read_labels(meta.loc[ordinary])
    state = fit_mixture(p[ordinary], y)
    state.update(families=train.FAMILIES, label_hash=common.object_hash(y.tolist()),
                 id_hash=common.object_hash(meta.loc[ordinary, ID].tolist()),
                 bindings=bindings, fitted_utc=common.utc_now())
    common.write_json(folder/'weights.json', state)
    before = common.sha256(folder/'weights.json')
    stage_names = ['calibration'] + ['evaluation_'+m for m in declare()['evaluation']['panels'][fold]]
    output = {}
    for stage in stage_names:
        meta, p, binding = experts(fold, stage)
        for i, family in enumerate(train.FAMILIES):
            meta[family] = p[:, i]
        for variant in VARIANTS:
            meta[variant] = predict_mixture(state, p, variant)
        meta.to_parquet(folder/f'{stage}.parquet', index=False)
        output[stage] = {'sha256': common.sha256(folder/f'{stage}.parquet'), 'experts': binding}
    assert common.sha256(folder/'weights.json') == before
    common.write_json(folder/'manifest.json', {'status': 'frozen_before_evaluation',
        'protocol_sha256': common.sha256(OUT/'protocol.json'), 'weights_sha256': before,
        'predictions': output, 'evaluation_labels_read': False, 'completed_utc': common.utc_now()})
    print('CALIBRATED', fold, {v: state[v] for v in VARIANTS}, flush=True)


def metrics(y, p):
    error = np.asarray(p, float)-np.asarray(y, float)
    return {'rows': len(y), 'mse': float(np.mean(error**2)),
            'rmse': float(np.sqrt(np.mean(error**2))), 'mae': float(np.abs(error).mean()),
            'bias': float(error.mean())}


def paired(y, candidate, control, dates):
    a, b = (y-candidate)**2, (y-control)**2
    codes, days = pd.factorize(dates, sort=True)
    counts = np.bincount(codes)
    sums_a, sums_b = np.bincount(codes, weights=a), np.bincount(codes, weights=b)
    weights = np.random.default_rng(20260916).multinomial(len(days), np.full(len(days), 1/len(days)), size=2000)
    interval = np.quantile((weights@(sums_b-sums_a))/(weights@counts), [.025, .975])
    removals = [float(np.sqrt((b.sum()-bb)/(len(y)-nn))-
                      np.sqrt((a.sum()-aa)/(len(y)-nn))) for aa, bb, nn in zip(sums_a, sums_b, counts)]
    order = np.argsort(-(b-a), kind='stable')
    deleted = {}
    for k in [1, 2, 5, 10]:
        keep = np.ones(len(y), bool)
        keep[order[:k]] = False
        deleted[str(k)] = float(np.sqrt(b[keep].mean())-np.sqrt(a[keep].mean()))
    return {'gain_sec': float(np.sqrt(b.mean())-np.sqrt(a.mean())),
            'mse_gain': float((b-a).mean()), 'mse_gain_ci95': interval.tolist(),
            'day_removal_gain_min': min(removals), 'all_day_removals_positive': min(removals)>0,
            'top_beneficial_deletion_gain': deleted,
            'interpretation': 'Conditional exposed-development paired-day sensitivity; not a fresh or selection-adjusted confidence interval.'}


def score(fold):
    protocol = declare()
    folder = OUT/fold/'evaluation'
    folder.mkdir(parents=True, exist_ok=False)
    calibration = OUT/fold/'calibration'
    marker = common.read_json(calibration/'manifest.json')
    assert marker['status'] == 'frozen_before_evaluation'
    assert marker['protocol_sha256'] == common.sha256(OUT/'protocol.json')
    assert common.sha256(calibration/'weights.json') == marker['weights_sha256']
    common.write_json(folder/'evaluation_opened.json', {'utc': common.utc_now(),
        'calibration_manifest_sha256': common.sha256(calibration/'manifest.json')})
    reports = {}
    for month in protocol['evaluation']['panels'][fold]:
        stage = 'evaluation_'+month
        path = calibration/f'{stage}.parquet'
        assert common.sha256(path) == marker['predictions'][stage]['sha256']
        frame = pd.read_parquet(path)
        y = train.read_labels(frame)
        ordinary = frame.proxy_sec.between(0, 7200).to_numpy()
        dates = frame[train.MOVEMENT].dt.floor('D').to_numpy()
        report = {}
        for slice_name, keep in [('ordinary', ordinary), ('all_finite', np.ones(len(frame), bool))]:
            report[slice_name] = {name: metrics(y[keep], frame[name].to_numpy()[keep])
                                  for name in train.FAMILIES+VARIANTS}
            report[slice_name]['shrunk_vs_equal'] = paired(y[keep], frame.shrunk.to_numpy()[keep], frame.equal.to_numpy()[keep], dates[keep])
            report[slice_name]['simplex_vs_equal'] = paired(y[keep], frame.simplex.to_numpy()[keep], frame.equal.to_numpy()[keep], dates[keep])
        frame[TARGET] = y
        frame.to_parquet(folder/f'{stage}.parquet', index=False)
        report['label_hash'] = common.object_hash(y.tolist())
        report['rows'] = len(frame)
        if (fold, month) in [('F1', '2025-07'), ('F3', '2025-11')]:
            old = ROOT/'private_runs/tail240_20260916/models/neural_missing_integration_v1/replacement'
            old_marker = common.read_json(old/'manifest.json')
            old_path = old/f'{fold}.parquet'
            assert common.sha256(old_path) == old_marker['folds'][fold]['prediction']['sha256']
            v3 = pd.read_parquet(old_path, columns=[ID, 'prediction_sec'])
            full = train.read_stage(fold, stage)
            assert np.array_equal(full[ID], v3[ID])
            full_y = train.read_labels(full)
            values = v3.prediction_sec.to_numpy().copy()
            positions = pd.Index(full[ID]).get_indexer(frame.loc[ordinary, ID])
            assert (positions >= 0).all()
            values[positions] = frame.loc[ordinary, 'shrunk']
            keep_other = np.ones(len(full), bool)
            keep_other[positions] = False
            assert np.array_equal(values[keep_other], v3.prediction_sec.to_numpy()[keep_other])
            full['candidate'] = values
            full['v3'] = v3.prediction_sec.to_numpy()
            full[TARGET] = full_y
            full.to_parquet(folder/'historical_hybrid.parquet', index=False)
            report['historical_hybrid'] = {'candidate': metrics(full_y, values),
                'v3': metrics(full_y, full.v3), 'nonordinary_exact': True,
                'paired': paired(full_y, values, full.v3.to_numpy(), full[train.MOVEMENT].dt.floor('D').to_numpy()),
                'caveat': protocol['historical_hybrid']}
        reports[month] = report
    common.write_json(folder/'metrics.json', reports)
    common.write_json(folder/'manifest.json', {'status': 'complete',
        'protocol_sha256': common.sha256(OUT/'protocol.json'),
        'calibration_manifest_sha256': common.sha256(calibration/'manifest.json'),
        'outputs': {p.name: common.sha256(p) for p in folder.iterdir() if p.is_file()}})
    print('EVALUATED', fold, {m: r['ordinary']['shrunk_vs_equal']['gain_sec'] for m, r in reports.items()}, flush=True)


def summary():
    declare()
    data = {f: common.read_json(OUT/f/'evaluation/metrics.json') for f in ['F1', 'F3']}
    checks = {}
    for fold, month in [('F1', '2025-07'), ('F3', '2025-11')]:
        p = data[fold][month]['ordinary']['shrunk_vs_equal']
        checks[fold+'_primary'] = p['gain_sec']>0 and p['mse_gain_ci95'][0]>0 and p['all_day_removals_positive'] and p['top_beneficial_deletion_gain']['10']>0
    for fold in ['F1', 'F3']:
        checks[fold+'_December'] = data[fold]['2025-12']['ordinary']['shrunk_vs_equal']['mse_gain_ci95'][0]>=0
    weights = {'F1': 192122/344841, 'F3': 152719/344841}
    hybrid = {}
    for key in ['candidate', 'v3']:
        hybrid[key] = float(np.sqrt(sum(weights[f]*data[f][m]['historical_hybrid'][key]['mse']
            for f, m in [('F1', '2025-07'), ('F3', '2025-11')])))
    common.write_json(OUT/'summary.json', {'status': 'complete', 'checks': checks,
        'shrunk_vs_equal_gate_passed': all(checks.values()), 'historical_hybrid_rmse': hybrid,
        'historical_hybrid_gain_sec': hybrid['v3']-hybrid['candidate'],
        'promotion': 'No automatic replacement; strict experiment covers three ordinary experts, not original nine-expert full pipeline.',
        'exposure': 'All months previously exposed; December shared across origins is not independent replication.'})
    print(common.read_json(OUT/'summary.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['declare', 'calibrate', 'score', 'summary'])
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    train.pa.set_cpu_count(2)
    train.pa.set_io_thread_count(1)
    with train.threadpool_limits(2):
        if args.action in ('calibrate', 'score'):
            assert args.fold
            globals()[args.action](args.fold)
        else:
            globals()[args.action]()
