"""Predeclared matched and fixed-blend checks for following-clock tune models."""
import following_groups_tune as fit
import argparse
import numpy as np
import pandas as pd

common,ROOT,ID,TARGET=fit.common,fit.ROOT,fit.ID,fit.TARGET
OUT=common.external_path(ROOT/'private_runs/tail240_20260916/models/following_groups_assessment_v1')


def run(fold):
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/f'{fold}.json').exists()
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert common.sha256(meta_path)==audit['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,'FLIGHT_ID_mvt',common.MOVEMENT,'ADEP_mvt',TARGET,'proxy_sec'])
    idx,split,_=common.fold_data(meta,fold,full=True)
    rows=idx['tune'][np.isfinite(meta.iloc[idx['tune']].proxy_sec.to_numpy())]
    truth=meta.iloc[rows].copy()
    predictions={}
    manifests={}
    for arm in fit.ARMS:
        folder=fit.OUT/fold/arm
        manifest=common.read_json(folder/'manifest.json')
        assert manifest['status']=='complete' and common.object_hash(manifest['split'])==common.object_hash(split)
        assert common.sha256(folder/'tune.parquet')==manifest['outputs']['tune.parquet']
        values=pd.read_parquet(folder/'tune.parquet')
        np.testing.assert_array_equal(values[ID],truth[ID])
        predictions[arm]=values.prediction_sec.to_numpy()
        manifests[arm]=common.sha256(folder/'manifest.json')
    actual=truth[TARGET].to_numpy(float)
    dates=truth[common.MOVEMENT].dt.floor('D').to_numpy()
    matched=fit.metric.comparison(actual,predictions['following415'],predictions['control387'],dates)
    base=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    prep=common.read_json(base/'preparation.json')['folds'][fold]
    path=base/f'{fold}_aligned_tune.parquet'
    assert common.sha256(path)==prep['aligned_tune_sha256']
    weights=common.read_json(base/f'{fold}_weights.json')
    assert weights==prep['weights']
    aligned=pd.read_parquet(path)
    ordinary=truth.proxy_sec.between(0,7200).to_numpy()
    np.testing.assert_array_equal(truth.loc[ordinary,ID],aligned[ID])
    np.testing.assert_array_equal(truth.loc[ordinary,TARGET],aligned[TARGET])
    reference=aligned[weights['experts']].to_numpy(float)@np.asarray(weights['global'])
    y=aligned[TARGET].to_numpy(float)
    candidate=.75*reference+.25*predictions['following415'][ordinary]
    paired=fit.metric.comparison(y,candidate,reference,dates[ordinary])
    paired.update(reference_rmse=float(np.sqrt(np.mean((reference-y)**2))),rmse=float(np.sqrt(np.mean((candidate-y)**2))))
    paired['rmse_improvement']=paired['reference_rmse']-paired['rmse']
    day_codes,days=pd.factorize(dates[ordinary],sort=True)
    bootstrap=np.random.default_rng(20260916).multinomial(len(days),np.full(len(days),1/len(days)),size=300)
    gain=(reference-y)**2-(candidate-y)**2
    paired['mse_gain_ci95']=fit.risk.bootstrap_interval(gain,np.ones(len(gain)),day_codes,bootstrap)
    advance=matched['mse_gain']>0 and matched['all_day_removals_improve'] and paired['mse_gain']>0 and paired['all_day_removals_improve'] and paired['rmse_improvement']>2
    report=dict(status='complete',fold=fold,source_sha256=common.sha256(__file__),model_manifests=manifests,
        matched=matched,ordinary_fixed25=paired,fold_gate=advance,
        limitation='Global9 weights learned on same tune month; exposed adaptive development. Both fold gates and independent replay required before score. No score/ranking reads.')
    common.write_json(OUT/f'{fold}.json',report)
    print('ASSESSMENT',fold,report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',choices=['F1','F3'],required=True)
    run(parser.parse_args().fold)
