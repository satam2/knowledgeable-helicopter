"""Matched CatBoost test of the already declared observed-scale raw-MSE target."""
import normalized_missing_tune as prior
from catboost import CatBoostRegressor, Pool
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

common, shared, ROOT, ID = prior.common, prior.shared, prior.ROOT, prior.ID
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/normalized_catboost_tune_v1')
PARAMS = dict(shared.base.PARAMS, random_seed=20260916, thread_count=2)
ARMS = ['unscaled', 'observed_schedule_scale']


def declare():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(source_sha256=common.sha256(__file__),
        imported_sources={str(Path(p).relative_to(ROOT)): common.sha256(p) for p in
            [prior.__file__, shared.__file__, shared.base.__file__, shared.identity.__file__]},
        params=PARAMS, arms=ARMS, stopping='Original80patience on weightedRMSE; fixed600cap, depth5,L2=30',
        target='Exact prior.transform/reconstruct/scale_of; unscaled1 or schedule-derivedscale, s^2/mean_fit(s^2)weights. Raw loss retained.',
        features='Same76airport+ID fields, native categorical CatBoost; no extra priors or labels.',
        scope='All original missing fit/tune rows, original IDpurges. No score/refit.',
        gate='Both months normalized must beat unscaled and saved scaledLightGBM; each day removal must preserve gain before score advancement.',
        comparison='Separately fit family capacities inherited from original missingCatBoost; not claiming equalcapacity toLGB.',
        references={f: common.sha256(prior.OUT / f / 'observed_schedule_scale/manifest.json') for f in ('F1','F3')})
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return record


def comparison(y, prediction, control, dates):
    gain = (y-control)**2 - (y-prediction)**2
    unique, codes = np.unique(dates, return_inverse=True)
    totals = np.bincount(codes, weights=gain)
    counts = np.bincount(codes)
    removal = (gain.sum()-totals)/(len(gain)-counts)
    return dict(mse_gain=float(gain.mean()), day_removal_min_gain=float(removal.min()),
        all_day_removals_improve=bool(np.all(removal > 0)), days=len(unique))


def run():
    protocol = declare()
    x, meta = shared.load_missing()
    missing = ~np.isfinite(meta.proxy_sec.to_numpy(float))
    y = meta[prior.TARGET].to_numpy(float)
    summary = {}
    for fold in ('F1','F3'):
        idx, split, _ = common.fold_data(meta, fold, full=True)
        rows = {s: p[missing[p]] for s,p in idx.items() if s in ('fit','tune')}
        frames = {s:x.loc[meta.iloc[p][ID]] for s,p in rows.items()}
        cats = shared.base.cats(frames['fit'])
        predictions = {}
        summary[fold] = {}
        for arm in ARMS:
            dest = OUT / fold / arm
            dest.mkdir(parents=True, exist_ok=False)
            scales = {s:np.ones(len(f)) if arm=='unscaled' else prior.scale_of(f) for s,f in frames.items()}
            normalizer = float(np.mean(scales['fit']**2))
            pools = {s:Pool(f, label=prior.transformed(y[rows[s]],scales[s]),
                weight=scales[s]**2/normalizer, cat_features=cats) for s,f in frames.items()}
            model = CatBoostRegressor(**PARAMS)
            model.fit(pools['fit'], eval_set=pools['tune'], early_stopping_rounds=80, use_best_model=True)
            pred = prior.reconstruct(model.predict(frames['tune'],thread_count=2),scales['tune'])
            assert np.isfinite(pred).all()
            model.save_model(str(dest/'model.cbm'))
            saved = CatBoostRegressor().load_model(str(dest/'model.cbm'))
            replay = prior.reconstruct(saved.predict(frames['tune'],thread_count=2),scales['tune'])
            np.testing.assert_array_equal(pred,replay)
            pd.DataFrame({ID:frames['tune'].index, 'prediction_sec':pred, 'scale':scales['tune'],
                'raw_target_sec':y[rows['tune']]}).to_parquet(dest/'tune.parquet',index=False)
            record = dict(status='complete', fold=fold, arm=arm, steps=model.tree_count_,
                rmse=float(np.sqrt(np.mean((pred-y[rows['tune']])**2))), split=split,
                ids={s:dict(n=len(p),hash=common.object_hash(meta.iloc[p][ID].tolist())) for s,p in rows.items()},
                source_sha256=common.sha256(__file__), protocol_sha256=common.sha256(OUT/'protocol.json'),
                params=PARAMS, normalizer=normalizer, feature_columns=list(x), replay_max_abs_delta=0.,
                outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file()})
            common.write_json(dest/'manifest.json',record)
            predictions[arm] = pred
            summary[fold][arm] = record['rmse']
            print('RESULT', fold, arm, record['rmse'], 'steps',model.tree_count_,flush=True)
        reference = pd.read_parquet(prior.OUT/fold/'observed_schedule_scale/tune.parquet')
        marker = common.read_json(prior.OUT/fold/'observed_schedule_scale/manifest.json')
        assert common.sha256(prior.OUT/fold/'observed_schedule_scale/manifest.json') == protocol['references'][fold]
        assert common.sha256(prior.OUT/fold/'observed_schedule_scale/tune.parquet') == marker['outputs']['tune.parquet']
        np.testing.assert_array_equal(reference[ID],frames['tune'].index)
        dates = meta.iloc[rows['tune']][prior.TIME].dt.floor('D').to_numpy()
        summary[fold]['comparisons'] = {name:comparison(y[rows['tune']],predictions['observed_schedule_scale'],control,dates)
            for name,control in [('unscaled',predictions['unscaled']),('scaled_lightgbm',reference.prediction_sec.to_numpy())]}
    advance = all(r['mse_gain']>0 and r['all_day_removals_improve'] for f in summary.values() for r in f['comparisons'].values())
    common.write_json(OUT/'summary.json',dict(status='complete',results=summary,advance=advance,no_score_prediction=True))
    print('SUMMARY',summary,'advance',advance,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args = parser.parse_args()
    if args.declare_only:
        declare()
    else:
        run()
