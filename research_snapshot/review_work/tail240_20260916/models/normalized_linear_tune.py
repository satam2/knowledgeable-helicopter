"""Matched constant versus linear leaf test on normalized missing-clock targets."""
import normalized_missing_tune as prior
import normalized_catboost_tune as metrics
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

common, ROOT, ID = prior.common, prior.ROOT, prior.ID
OUT = common.external_path(ROOT / 'private_runs/tail240_20260916/models/normalized_linear_tune_v1')
ARMS = ['constant', 'linear']


def standardize(frames, encoder):
    fit = frames['fit'][encoder.numeric].to_numpy(float)
    means = np.array([np.mean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in fit.T])
    stds = np.array([np.std(col[np.isfinite(col)]) if np.isfinite(col).any() else 1. for col in fit.T])
    stds = np.where(stds > 1e-6, stds, 1.)
    result = {}
    for stage, frame in frames.items():
        result[stage] = frame.copy()
        result[stage][encoder.numeric] = (frame[encoder.numeric].to_numpy(float)-means)/stds
    return result, dict(means=means, stds=stds)


def declare():
    OUT.mkdir(parents=True,exist_ok=True)
    record = dict(source_sha256=common.sha256(__file__),
        source_hashes={str(Path(p).relative_to(ROOT)):common.sha256(p) for p in
            [prior.__file__,metrics.__file__,prior.shared.__file__,prior.shared.base.__file__,prior.shared.identity.__file__,
             ROOT/'review_work/breakthrough_20260916/models/encoders.py']},
        params=prior.PARAMS, linear_lambda=5., arms=ARMS, rounds=600, stopping=60,
        features='Exact76airport+ID, fit-onlyFrameEncoder, numeric fit-onlymean/std, missingNaNpreserved; categoriesunchanged.',
        target='Same schedule scale and s^2/mean_fit(s^2) weights; rawMSE exact dataobjective; no label clipping.',
        contrast='Same standardized frames, parameters and targets; only linear_tree flag differs. linear_lambda=5fixed inbotharms.',
        rationale='Leaf-local numeric response may estimate correction magnitude better than piecewiseconstant Z; prior ordinary rawlinear experiment does not test this sparse normalized population.',
        gate='Both June/October linear beats constant and frozen normalizedLightGBM, allsingleUTCdayremovals must preserve gain. No score/refit before independent review.',
        scope='Full original missingfit/tune only with originalflightIDpurges; no sampling or scorelabelselection.')
    path = OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


def run():
    declare()
    x,meta=prior.shared.load_missing()
    y=meta[prior.TARGET].to_numpy(float)
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    summary={}
    for fold in ('F1','F3'):
        idx,split,_=common.fold_data(meta,fold,full=True)
        rows={s:p[missing[p]] for s,p in idx.items() if s in ('fit','tune')}
        frames={s:x.loc[meta.iloc[p][ID]] for s,p in rows.items()}
        encoder=prior.FrameEncoder().fit(frames['fit'])
        encoded={s:encoder.transform(f) for s,f in frames.items()}
        matrices,scaler=standardize(encoded,encoder)
        scales={s:prior.scale_of(f) for s,f in frames.items()}
        normalizer=float(np.mean(scales['fit']**2))
        targets={s:prior.transformed(y[p],scales[s]) for s,p in rows.items()}
        weights={s:scales[s]**2/normalizer for s in rows}
        predictions={}
        summary[fold]={}
        for arm in ARMS:
            dest=OUT/fold/arm
            dest.mkdir(parents=True,exist_ok=False)
            params=dict(prior.PARAMS,linear_tree=arm=='linear',linear_lambda=5.)
            train=prior.lgb.Dataset(matrices['fit'],label=targets['fit'],weight=weights['fit'],categorical_feature=list(encoder.categories),params=params)
            tune=prior.lgb.Dataset(matrices['tune'],label=targets['tune'],weight=weights['tune'],reference=train,categorical_feature=list(encoder.categories),params=params)
            model=prior.lgb.train(params,train,num_boost_round=600,valid_sets=[tune],callbacks=[prior.lgb.early_stopping(60,verbose=False)])
            pred=prior.reconstruct(model.predict(matrices['tune']),scales['tune'])
            assert np.isfinite(pred).all()
            joblib.dump(dict(model=model,encoder=encoder,scaler=scaler,arm=arm),dest/'model.joblib')
            saved=joblib.load(dest/'model.joblib')
            replay_frame=saved['encoder'].transform(frames['tune'])
            replay_frame[encoder.numeric]=(replay_frame[encoder.numeric].to_numpy(float)-saved['scaler']['means'])/saved['scaler']['stds']
            replay=prior.reconstruct(saved['model'].predict(replay_frame),scales['tune'])
            np.testing.assert_array_equal(pred,replay)
            pd.DataFrame({ID:frames['tune'].index,'prediction_sec':pred,'raw_target_sec':y[rows['tune']],'scale':scales['tune']}).to_parquet(dest/'tune.parquet',index=False)
            record=dict(status='complete',fold=fold,arm=arm,params=params,steps=model.best_iteration,
                rmse=float(np.sqrt(np.mean((pred-y[rows['tune']])**2))),split=split,normalizer=normalizer,
                ids={s:dict(n=len(p),hash=common.object_hash(meta.iloc[p][ID].tolist())) for s,p in rows.items()},
                feature_columns=list(x),replay_max_abs_delta=0.,source_sha256=common.sha256(__file__),
                protocol_sha256=common.sha256(OUT/'protocol.json'),
                outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.is_file()})
            common.write_json(dest/'manifest.json',record)
            predictions[arm]=pred
            summary[fold][arm]=record['rmse']
            print('RESULT',fold,arm,record['rmse'],'steps',model.best_iteration,flush=True)
        reference=pd.read_parquet(prior.OUT/fold/'observed_schedule_scale/tune.parquet')
        np.testing.assert_array_equal(reference[ID],frames['tune'].index)
        marker=common.read_json(prior.OUT/fold/'observed_schedule_scale/manifest.json')
        assert common.sha256(prior.OUT/fold/'observed_schedule_scale/tune.parquet')==marker['outputs']['tune.parquet']
        dates=meta.iloc[rows['tune']][prior.TIME].dt.floor('D').to_numpy()
        summary[fold]['comparisons']={name:metrics.comparison(y[rows['tune']],predictions['linear'],control,dates)
            for name,control in [('constant',predictions['constant']),('frozen_scaled_lgb',reference.prediction_sec.to_numpy())]}
    advance=all(r['mse_gain']>0 and r['all_day_removals_improve'] for f in summary.values() for r in f['comparisons'].values())
    common.write_json(OUT/'summary.json',dict(status='complete',results=summary,advance=advance,no_score_prediction=True))
    print('SUMMARY',summary,'advance',advance,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    if args.declare_only:
        declare()
    else:
        run()
