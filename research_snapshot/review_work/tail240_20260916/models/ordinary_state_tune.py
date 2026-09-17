"""Chronological historical state on the ordinary/full finite population."""
import following_groups_tune as shared
import schema_discovery
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

ROOT,common,risk,ID,TARGET=shared.ROOT,shared.common,shared.risk,shared.ID,shared.TARGET
CACHE=ROOT/'private_runs/tail240_20260916/state/cache_v1'
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/models/ordinary_state_tune_v1')
ORIGINAL=risk.feature_sources
FOLD=None


def state_columns():
    import pyarrow.parquet as pq
    return [c for c in pq.read_schema(CACHE/'F1_fit.parquet').names if c!=ID]


def sources(columns):
    result,receipt=schema_discovery.cached_discovery(ORIGINAL,columns)
    common.write_json(OUT/FOLD/'discovery.json',receipt)
    marker=common.read_json(CACHE/'manifest.json')
    chosen=[c for c in columns if c in state_columns()]
    for stage in ('fit','tune'):
        record=marker['folds'][FOLD]['stages'][stage]
        path=ROOT/record['path']
        assert common.sha256(path)==record['sha256']
        result.append((path,record['sha256'],True,chosen))
    return result


def declaration():
    marker=common.read_json(CACHE/'manifest.json')
    control=common.read_json(risk.union_folder('F1')/'manifest.json')
    columns=state_columns()
    assert marker['status']=='complete' and len(columns)==12
    OUT.mkdir(parents=True,exist_ok=True)
    record=dict(source_sha256=common.sha256(__file__),
        sources={str(Path(p).relative_to(ROOT)):common.sha256(p) for p in [risk.__file__,shared.__file__,schema_discovery.__file__]},
        cache_manifest_sha256=common.sha256(CACHE/'manifest.json'),
        columns=control['feature_columns']+columns,params=control['fit']['params'],
        model='Frozenleaf63 rawY-minusNM, fulloriginalfinitefit/tune,2500cap150patience,2CPU. Exactsame387control withadditional12chronologicalstatefields.',
        state='Original independentlyverified wholemonthcrossfit taxi/source airport-hour priors and28daystate; originalpurges beforehistoryaggregation; no sourcecache edits.',
        difference='Previousstate12experiment restrictedmissingNM; this tests ordinary/fullfinite population with matchedstrongunion387.',
        availability='State earlierlabels only, inherited387finalbatchretrospective. Adjacent2025validation only; July2026sixmonthlabelblackout NOTvalidated.',
        gate='Bothmonths state399 beatsunion387 andordinaryfixed25global9 improves, everydayremovalpositive and>=2seasonalensembleRMSEgain before fullscore consideration. No score/refitpath inthisrunner.',
        transport='Evenpositiveadjacenttune result requiresseparateblackout/prior-year availability investigation before ranking recommendation.',
        resources='2CPU; sampledRSS10GiB and8GiBhostreserve; startsat18GiBfree, schema-cachedsource discovery.')
    path=OUT/'protocol.json'
    if path.exists():
        assert common.read_json(path)==record
    else:
        common.write_json(path,record)
    return record


def run(fold):
    global FOLD
    FOLD=fold
    protocol=declaration()
    assert psutil.virtual_memory().available>=18*1024**3
    dest=OUT/fold
    dest.mkdir(exist_ok=False)
    risk.feature_sources=sources
    risk.guard=shared.guard
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path)==audit['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    cache=common.read_json(CACHE/'manifest.json')
    assert common.object_hash(split)==common.object_hash(cache['folds'][fold]['split'])
    parts={s:meta.iloc[p[np.isfinite(meta.iloc[p].proxy_sec.to_numpy())]].copy() for s,p in idx.items() if s in ('fit','tune')}
    ids=pd.Index(pd.concat([parts['fit'][ID],parts['tune'][ID]],ignore_index=True))
    nfit=len(parts['fit'])
    matrix,vocab,receipts=risk.load_matrix(ids,nfit,protocol['columns'],dest)
    target={s:f[TARGET].to_numpy(float)-f.proxy_sec.to_numpy(float) for s,f in parts.items()}
    model=lgb.LGBMRegressor(**protocol['params'])
    model.fit(matrix[:nfit],target['fit'],categorical_feature=[protocol['columns'].index(c) for c in vocab],
        feature_name=protocol['columns'],eval_X=matrix[nfit:],eval_y=target['tune'],eval_metric='rmse',
        callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(250)])
    steps=int(model.best_iteration_ or model.n_estimators_)
    prediction=model.predict(matrix[nfit:],num_iteration=steps)+parts['tune'].proxy_sec.to_numpy(float)
    model.booster_.save_model(str(dest/'model.txt'),num_iteration=steps)
    replay=lgb.Booster(model_file=str(dest/'model.txt')).predict(matrix[nfit:])+parts['tune'].proxy_sec.to_numpy(float)
    np.testing.assert_array_equal(prediction,replay)
    assert np.isfinite(prediction).all()
    pd.DataFrame({ID:parts['tune'][ID],'prediction_sec':prediction}).to_parquet(dest/'tune.parquet',index=False)
    common.write_json(dest/'encoder.json',dict(columns=protocol['columns'],vocab=vocab))
    control_folder=risk.union_folder(fold)
    old=common.read_json(control_folder/'manifest.json')
    assert common.sha256(control_folder/'tune_predictions.parquet')==old['outputs']['tune_predictions.parquet']
    control=pd.read_parquet(control_folder/'tune_predictions.parquet')
    np.testing.assert_array_equal(control[ID],parts['tune'][ID])
    y=parts['tune'][TARGET].to_numpy(float)
    dates=parts['tune'][common.MOVEMENT].dt.floor('D').to_numpy()
    comparison=shared.metric.comparison(y,prediction,control.prediction_sec.to_numpy(),dates)
    base=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=common.read_json(base/'preparation.json')['folds'][fold]
    path=base/f'{fold}_aligned_tune.parquet'
    assert common.sha256(path)==prep['aligned_tune_sha256']
    weights=common.read_json(base/f'{fold}_weights.json')
    assert weights==prep['weights']
    aligned=pd.read_parquet(path)
    ordinary=parts['tune'].proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(aligned[ID],parts['tune'].loc[ordinary,ID])
    baseline=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    blended=.75*baseline+.25*prediction[ordinary]
    ensemble=shared.metric.comparison(y[ordinary],blended,baseline,dates[ordinary])
    ensemble['reference_rmse']=float(np.sqrt(np.mean((baseline-y[ordinary])**2)))
    ensemble['rmse']=float(np.sqrt(np.mean((blended-y[ordinary])**2)))
    ensemble['gain']=ensemble['reference_rmse']-ensemble['rmse']
    common.write_json(dest/'manifest.json',dict(status='complete',fold=fold,steps=steps,params=protocol['params'],split=split,
        source_sha256=common.sha256(__file__),protocol_sha256=common.sha256(OUT/'protocol.json'),feature_receipts=receipts,
        ids={s:dict(n=len(f),hash=common.object_hash(f[ID].tolist())) for s,f in parts.items()},
        rmse=float(np.sqrt(np.mean((prediction-y)**2))),matched=comparison,ordinary_fixed25=ensemble,
        native_replay_max_abs_delta=0.,no_score_prediction=True,
        outputs={p.name:common.sha256(p) for p in dest.iterdir() if p.name in ['model.txt','encoder.json','tune.parquet','discovery.json']}))
    del matrix
    (dest/'matrix.float32').unlink()
    print('COMPLETE',fold,comparison,ensemble,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--fold',choices=['F1','F3'])
    args=parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        assert args.fold
        run(args.fold)
