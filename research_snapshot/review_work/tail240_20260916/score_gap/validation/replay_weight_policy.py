"""Label-preserving local replay of the submitted fixed weight policy; no fitting."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import psutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'private_runs/tail240_20260916/score_gap/validation'
BASE=ROOT/'private_runs/breakthrough_20260916'
TAIL=ROOT/'private_runs/tail240_20260916'
ID,TARGET,TIME='MVT_ID_mvt','TAXITIME_SEC_mvt','MVT_TIME_UTC_mvt'
WEIGHT_ROOT=BASE/'models/context_gate/final_simplex9_v1'
BEST=TAIL/'models/neural_missing_integration_v1/replacement'
NEURAL=TAIL/'state/neural_context/refit_score_v3'
MISSING=TAIL/'models/normalized_missing_refit_v2/observed_schedule_scale_blend25'
FINAL=TAIL/'final_submission_v3_v2'
SEASONS={'F1':192122/344841,'F3':152719/344841}
ACTIVE=['tabm_combined','tabm_ple8','lgb63_sequence8','lgb63_union','catboost_combined']
BINDINGS={}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):digest.update(part)
    return digest.hexdigest()


def read(path):
    path=Path(path);BINDINGS[str(path.relative_to(ROOT))]=sha(path)
    return json.loads(path.read_text())


def table(path,expected=None):
    path=Path(path);digest=sha(path)
    if expected is not None:assert digest==expected,path
    BINDINGS[str(path.relative_to(ROOT))]=digest
    return pd.read_parquet(path)


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    assert y.shape==p.shape and np.isfinite(y).all() and np.isfinite(p).all()
    error=p-y;absolute=np.abs(error);sse=float(error@error)
    return dict(n=len(y),sse=sse,mse=sse/len(y),rmse=float(np.sqrt(sse/len(y))),
                mae=float(absolute.mean()),bias=float(error.mean()),p95=float(np.quantile(absolute,.95)),p99=float(np.quantile(absolute,.99)))


def guard():
    memory=psutil.Process().memory_info()
    assert memory.peak_wset<4*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return int(memory.peak_wset)


def compare(y,a,b,mask):
    if not mask.any():return dict(n=0)
    old,new=metrics(y[mask],a[mask]),metrics(y[mask],b[mask])
    return dict(n=old['n'],baseline=old,candidate=new,delta_rmse=new['rmse']-old['rmse'],delta_sse=new['sse']-old['sse'])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    final=read(FINAL/'protocol.json');official=read(FINAL/'organizer_result.json')
    official_v2=read(ROOT/'private_runs/submission_v2/organizer_v2_result.json')
    assert official['status']==official_v2['status']=='Succeeded'
    assert official['used_pairs']==official_v2['used_pairs']==344841
    best_manifest=read(BEST/'manifest.json');assert best_manifest['status']=='complete'
    assert best_manifest['protocol_sha256']==sha(BEST/'protocol.json');read(BEST/'protocol.json')
    preparation=read(WEIGHT_ROOT/'preparation.json')
    binding=read(TAIL/'validation/baseline_binding.json')
    audited=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    audit=read(ROOT/'private_runs/screening_230/reports/data_audit.json')
    assert sha(audited)==audit['artifacts'][audited.name]
    BINDINGS[str(audited.relative_to(ROOT))]=sha(audited)
    raw=pd.read_parquet(audited,columns=[ID,TARGET,TIME,'ADEP_mvt','proxy_sec']).set_index(ID)
    assert raw.index.is_unique
    rankpath=ROOT/'private_runs/submission_v2/ranking_meta.parquet'
    rankbinding=read(rankpath.parent/'ranking_inputs.json')
    ranking=table(rankpath,rankbinding['files'][rankpath.name])
    ranking_counts=ranking[TIME].dt.strftime('%Y-%m').value_counts().to_dict()
    assert ranking_counts=={'2026-07':192122,'2026-01':152719}
    fixed=np.array(final['weights'],float)
    fold_weights={fold:read(WEIGHT_ROOT/f'{fold}_weights.json') for fold in SEASONS}
    for fold in SEASONS:assert sha(WEIGHT_ROOT/f'{fold}_weights.json')==final['weight_sources'][fold]
    averaged=sum(SEASONS[fold]*np.array(fold_weights[fold]['global']) for fold in SEASONS)
    averaged[averaged<1e-12]=0;averaged/=averaged.sum();np.testing.assert_array_equal(fixed,averaged)
    assert list(final['experts'])==fold_weights['F1']['experts']==fold_weights['F3']['experts']
    protocol=dict(source_sha256=sha(__file__),scope='Controlled local release-weight/rounding/metric sensitivity; no model fitting, no new optimization, no final-model evaluation on2025, no network/truth access.',
        actual_final_protocol_sha256=sha(FINAL/'protocol.json'),fixed_weights=final['active_weights'],
        variants=['local_fold_weights_raw','local_fold_weights_rounded','final_fixed_weights_raw','final_fixed_weights_rounded','v2_raw','v2_rounded'],
        rows='Unchanged complete July190713/November162332 movement IDs and original unmodified labels',
        caveat='Fixed weights include October calibration when replayed on July. This is a mechanical counterfactual, not chronological fresh validation.',
        resources='CPU1,4GiBpeak,8GiBreserve')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    folds={};airports=[];routes=[];daily=[];all_y=[];all_a=[];all_b=[]
    for fold in SEASONS:
        bm=best_manifest['folds'][fold]
        best=table(BEST/bm['prediction']['path'],bm['prediction']['sha256'])
        assert len(best)==bm['cohorts']['score']['n'] and best[ID].is_unique
        local=raw.loc[best[ID]].reset_index();y=local[TARGET].to_numpy(float);proxy=local.proxy_sec.to_numpy(float)
        finite=np.isfinite(proxy);ordinary=finite&(proxy>=0)&(proxy<=7200);other=finite&~ordinary;missing=~finite
        assert set(local[TIME].dt.month)=={7 if fold=='F1' else 11}
        nm=read(NEURAL/fold/'manifest.json');assert nm['status']=='complete'
        neural=table(NEURAL/fold/'finite_score_predictions.parquet',nm['outputs']['finite_score_predictions.parquet'])
        np.testing.assert_array_equal(neural[ID],local.loc[finite,ID]);np.testing.assert_array_equal(neural[TARGET],y[finite])
        expert_values=[];expert_sources={}
        for expert in final['experts']:
            source=preparation['sources'][fold][expert];folder=Path(source['directory'])
            marker=read(folder/'manifest.json');assert sha(folder/'manifest.json')==source['manifest_sha256']
            frame=table(folder/'candidate.parquet',source['outputs']['candidate.parquet'])
            np.testing.assert_array_equal(frame[ID],best[ID]);np.testing.assert_array_equal(frame[TARGET],y)
            expert_values.append(frame.prediction_sec.to_numpy(float))
            expert_sources[expert]=str((folder/'candidate.parquet').relative_to(ROOT))
        p=np.column_stack(expert_values);ple_index=final['experts'].index('tabm_ple8')
        np.testing.assert_array_equal(neural.control225_prediction_sec,p[finite,ple_index])
        localweights=np.array(fold_weights[fold]['global'])
        original=np.sum(p[ordinary]*localweights[None,:],axis=1)
        p[finite,ple_index]=neural.prediction_sec
        reconstructed=original+localweights[ple_index]*(p[ordinary,ple_index]-expert_values[ple_index][ordinary])
        max_rebuild=float(np.max(np.abs(reconstructed-best.prediction_sec.to_numpy()[ordinary])))
        assert max_rebuild<1e-9,max_rebuild
        missing_marker=read(MISSING/'manifest.json')
        missing_file=table(MISSING/f'{fold}.parquet',missing_marker['folds'][fold]['prediction']['sha256'])
        np.testing.assert_array_equal(missing_file[ID],best[ID]);np.testing.assert_array_equal(missing_file.prediction_sec.to_numpy()[missing],best.prediction_sec.to_numpy()[missing])
        np.testing.assert_array_equal(p[other,final['experts'].index('lgb63')],best.prediction_sec.to_numpy()[other])
        baseline=best.prediction_sec.to_numpy(float)
        candidate=baseline.copy();value=np.zeros(int(ordinary.sum()))
        for expert in ACTIVE:
            j=final['experts'].index(expert);value+=fixed[j]*p[ordinary,j]
        candidate[ordinary]=value
        np.testing.assert_array_equal(candidate[~ordinary],baseline[~ordinary])
        v2folder=ROOT/f'private_runs/next_230/models/clock_and_rome_ensemble_{fold}_s20260910'
        v2marker=read(v2folder/'manifest.json');v2=table(v2folder/'score_predictions.parquet',v2marker['outputs']['score_predictions.parquet'])
        np.testing.assert_array_equal(v2[ID],best[ID]);np.testing.assert_array_equal(v2[TARGET],y)
        variants=dict(local_fold_weights_raw=baseline,local_fold_weights_rounded=np.rint(baseline),
            final_fixed_weights_raw=candidate,final_fixed_weights_rounded=np.rint(candidate),
            v2_raw=v2.prediction_sec.to_numpy(float),v2_rounded=np.rint(v2.prediction_sec.to_numpy(float)))
        scores={name:metrics(y,pred) for name,pred in variants.items()}
        labels=np.where(ordinary,'ordinary',np.where(other,'finite_nonordinary','missing'))
        days=local[TIME].dt.strftime('%Y-%m-%d').to_numpy();airport=local.ADEP_mvt.astype(str).to_numpy()
        for name in sorted(set(airport)):
            row=compare(y,baseline,candidate,airport==name);airports.append(dict(fold=fold,airport=name,**row))
        for name in ('ordinary','finite_nonordinary','missing'):
            row=compare(y,baseline,candidate,labels==name);routes.append(dict(fold=fold,route=name,**row))
        sse_a=(baseline-y)**2;sse_b=(candidate-y)**2
        for day in sorted(set(days)):
            selected=days==day
            row=compare(y,baseline,candidate,selected)
            remain=compare(y,baseline,candidate,~selected)
            daily.append(dict(fold=fold,day=day,**row,remove_day_delta_rmse=remain['delta_rmse']))
        folds[fold]=dict(rows=len(y),metrics=scores,old_weights=localweights.tolist(),fixed_weights=fixed.tolist(),
            weight_delta=(fixed-localweights).tolist(),ordinary_rows=int(ordinary.sum()),finite_nonordinary_rows=int(other.sum()),missing_rows=int(missing.sum()),
            original_ordinary_reconstruction_max_delta=max_rebuild,protected_routes_exact=True,
            prediction_shift=dict(mean=float((candidate-baseline).mean()),rms=float(np.sqrt(np.mean((candidate-baseline)**2))),maximum_absolute=float(np.max(np.abs(candidate-baseline)))),
            expert_sources=expert_sources,all_days=sum(r['fold']==fold for r in daily),
            worsened_days=sum(r['fold']==fold and r['delta_rmse']>0 for r in daily),
            improved_days=sum(r['fold']==fold and r['delta_rmse']<0 for r in daily),
            leave_one_day_delta_range=[min(r['remove_day_delta_rmse'] for r in daily if r['fold']==fold),max(r['remove_day_delta_rmse'] for r in daily if r['fold']==fold)])
        output=local[[ID,TARGET,TIME,'ADEP_mvt','proxy_sec']].copy();output['route']=labels
        for name,pred in variants.items():output[name]=pred
        output.to_parquet(OUT/f'{fold}_replay.parquet',index=False)
        all_y.append(y);all_a.append(baseline);all_b.append(candidate)
        print('FOLD',fold,{name:row['rmse'] for name,row in scores.items()},flush=True);guard()
    summaries={}
    for name in protocol['variants']:
        stats=[folds[fold]['metrics'][name] for fold in SEASONS]
        summaries[name]=dict(seasonal_rmse=float(np.sqrt(sum(SEASONS[fold]*folds[fold]['metrics'][name]['mse'] for fold in SEASONS))),
            pooled_rmse=float(np.sqrt(sum(row['sse'] for row in stats)/sum(row['n'] for row in stats))),
            equal_month_mse_rmse=float(np.sqrt(np.mean([row['mse'] for row in stats]))),
            mean_monthly_rmse=float(np.mean([row['rmse'] for row in stats])))
    score=lambda name:summaries[name]['seasonal_rmse']
    assert abs(score('local_fold_weights_raw')-268.662991126314)<1e-9
    assert abs(score('v2_raw')-292.8133464057956)<1e-9
    attribution=dict(official_v3=official['score'],official_v2=official_v2['score'],
        raw_local_gap=official['score']-score('local_fold_weights_raw'),
        local_fixed_weight_policy_delta=score('final_fixed_weights_raw')-score('local_fold_weights_raw'),
        local_rounding_after_fixed_weights_delta=score('final_fixed_weights_rounded')-score('final_fixed_weights_raw'),
        remaining_official_minus_fixed_policy_rounded_local=official['score']-score('final_fixed_weights_rounded'),
        historical_v2_official_minus_local_raw=official_v2['score']-score('v2_raw'),
        historical_v2_official_minus_local_rounded=official_v2['score']-score('v2_rounded'),
        local_recipe_gain=score('v2_raw')-score('local_fold_weights_raw'),
        local_gain_after_weight_policy_and_rounding=score('v2_rounded')-score('final_fixed_weights_rounded'),
        official_recipe_gain=official_v2['score']-official['score'],
        local_seasonal_minus_pooled=score('local_fold_weights_raw')-summaries['local_fold_weights_raw']['pooled_rmse'],
        fixed_rounded_seasonal_minus_pooled=score('final_fixed_weights_rounded')-summaries['final_fixed_weights_rounded']['pooled_rmse'])
    assert abs(attribution['raw_local_gap']-sum(attribution[k] for k in ('local_fixed_weight_policy_delta','local_rounding_after_fixed_weights_delta','remaining_official_minus_fixed_policy_rounded_local')))<1e-10
    for name,rows in [('airport_effects',airports),('route_effects',routes),('daily_effects',daily)]:
        (OUT/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),
        source_bindings=BINDINGS,folds=folds,summary=summaries,attribution=attribution,
        metric_weights=SEASONS,ranking_month_counts=ranking_counts,local_pooled_month_weights={fold:folds[fold]['rows']/353045 for fold in SEASONS},
        peak_bytes=guard(),elapsed_seconds=time.monotonic()-start,
        interpretation='Weight and rounding deltas are identified on the same local prediction vectors. Residual external gap cannot distinguish adaptive overfitting, calendar/year/airport/tail shift, larger full-year fitting, capacity transfer, or random training variation without new comparable held-out labels.',
        no_model_fitting=True,no_full2025_model_local_holdout_claim=True,no_network_or_truth_access=True,
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    for relative,digest in BINDINGS.items():assert sha(ROOT/relative)==digest
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summaries,attribution=attribution,peak_bytes=result['peak_bytes']),indent=2),flush=True)


if __name__=='__main__':main()
