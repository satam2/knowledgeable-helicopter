"""Correct diagnostic retry: guarded-reference blend changes missing route only."""
import audit as first
import numpy as np
import pandas as pd
from pathlib import Path
p=first.previous
ROOT=first.ROOT
OUT=first.OUT/'attempt_v2'
ID,TARGET,TIME=first.ID,first.TARGET,first.TIME


def main():
    OUT.mkdir(exist_ok=False)
    x,meta=p.missing.load_data()
    peers=pd.read_parquet(p.context.CACHE/'features.parquet').set_index(ID)
    assert p.sha256(p.context.CACHE/'features.parquet')==p.read_json(p.context.CACHE/'audit.json')['feature_sha256']
    np.testing.assert_array_equal(peers.index,meta[ID])
    times=meta.set_index(ID).loc[x.index,TIME]
    fx=pd.concat([x,p.context.id_context_features(x,peers.loc[x.index],times)],axis=1)
    fx[p.forest.TIME_COLUMN]=times
    folds={}
    for fold in ['F1','F3']:
        idx,split,_=p.common.fold_data(meta,fold,full=True)
        ref,rm=p.common.reference(fold)
        assert p.object_hash(split)==p.object_hash(rm['split'])
        np.testing.assert_array_equal(ref[ID],meta.iloc[idx['score']][ID])
        np.testing.assert_array_equal(ref[TARGET],meta.iloc[idx['score']][TARGET])
        y=ref[TARGET].to_numpy(float)
        day=ref.day.to_numpy()
        missing=~np.isfinite(meta.iloc[idx['score']].proxy_sec.to_numpy(float))
        pred={'V2':ref.prediction_sec.to_numpy(float)}
        candidates={}
        manifests={}
        tunes={}
        tr=idx['tune'][~np.isfinite(meta.iloc[idx['tune']].proxy_sec.to_numpy(float))]
        for name,directory in [('template',p.TEMPLATES/f'historical_template_{fold}_s20260916'),
            ('RF',p.FORESTS/f'randomforest_missing_template_idcontext_{fold}_s20260916'),
            ('ET',p.FORESTS/f'extratrees_missing_template_idcontext_{fold}_s20260916')]:
            record=p.checked(directory)
            assert p.object_hash(record['split'])==p.object_hash(split)
            for stage,pos in idx.items():
                chosen=pos[~np.isfinite(meta.iloc[pos].proxy_sec.to_numpy(float))]
                assert record['fit_ids'][stage]==dict(n=len(chosen),hash=p.object_hash(meta.iloc[chosen][ID].tolist()))
            for variant in ['candidate','blend25']:
                frame=pd.read_parquet(directory/(variant+'.parquet'))
                np.testing.assert_array_equal(frame[ID],ref[ID])
                np.testing.assert_array_equal(frame[TARGET],ref[TARGET])
                v=frame.prediction_sec.to_numpy(float)
                np.testing.assert_array_equal(v[~missing],pred['V2'][~missing])
                assert abs(p.prioraudit.metric(y,v)['rmse_sec']-record['reports'][variant]['metrics']['overall']['rmse_sec'])<1e-10
                (pred if variant=='blend25' else candidates)[name]=v
            np.testing.assert_allclose(pred[name],pred['V2']+.25*(candidates[name]-pred['V2']),rtol=0,atol=1e-9)
            tune=pd.read_parquet(directory/'tune_predictions.parquet')
            np.testing.assert_array_equal(tune[ID],meta.iloc[tr][ID])
            tunes[name]=tune.prediction_sec.to_numpy(float)
            manifests[name]=dict(path=str(directory),manifest_sha256=p.sha256(directory/'manifest.json'),fit_ids=record['fit_ids'])
            if name=='ET':
                p.check_sources(record)
                assert record['protocol_sha256']==p.sha256(p.FORESTS/'protocol_extratrees_s20260916.json')
                replay=p.replay(directory,record,fx,meta,idx)
                assert replay['fit']['estimator_class']=='ExtraTreesRegressor'
        floor=float(meta.iloc[idx['fit']][TARGET].min())
        guarddir=ROOT/'private_runs/breakthrough_20260916/models/support_guard'/fold
        gm=p.checked(guarddir)
        assert gm['fit_min_target_sec']==floor
        guard=pd.read_parquet(guarddir/'fit_min_floor.parquet').prediction_sec.to_numpy(float)
        np.testing.assert_array_equal(guard,np.maximum(pred['V2'],floor))
        post={name:np.maximum(value,floor) for name,value in pred.items()}
        before={'V2':guard}
        for name,value in candidates.items():
            before[name]=guard.copy()
            before[name][missing]+=.25*(value[missing]-guard[missing])
            np.testing.assert_array_equal(before[name][~missing],guard[~missing])
        gain=(pred['V2']-y)**2-(pred['ET']-y)**2
        diff=(pred['template']-y)**2-(pred['ET']-y)**2
        top=np.argsort(gain)[-2:]
        topdiff=np.argsort(diff)[-2:]
        tail=missing&(y>=86400)
        masks={'all':np.ones(len(y),bool),'remove_ETgain_top2':~np.isin(np.arange(len(y)),top),
            'remove_ET_vs_template_gain_top2':~np.isin(np.arange(len(y)),topdiff),
            'remove_missing_dayplus':~tail,'remove_missing_dayplus_days':~np.isin(day,np.unique(day[tail])),
            'remove_V2_below_fitfloor':pred['V2']>=floor}
        metrics={s:{name:p.prioraudit.metric(y,v,mask) for name,v in pred.items()} for s,mask in masks.items()}
        pairs=lambda values:{name+'_vs_ET':p.prioraudit.comparison(y,v,values['ET'],day) for name,v in values.items() if name!='ET'}
        sensitivity=ref[[ID,TARGET,'day']].copy()
        for name,v in pred.items():
            sensitivity[name]=v
        for name,v in candidates.items():
            sensitivity[name+'_candidate']=v
        sensitivity['ET_gain_vs_V2']=gain
        sensitivity['ET_gain_vs_template']=diff
        sensitivity.iloc[np.unique(np.r_[top,topdiff,np.flatnonzero(tail)])].to_parquet(OUT/(fold+'_sensitivity_rows.parquet'),index=False)
        folds[fold]=dict(replay=replay,manifests=manifests,metrics=metrics,fit_floor=floor,pairs=pairs(pred),
            common_postblend_floor_metrics={name:p.prioraudit.metric(y,v) for name,v in post.items()},common_postblend_floor_pairs=pairs(post),
            guard_reference_beforeblend_metrics={name:p.prioraudit.metric(y,v) for name,v in before.items()},guard_reference_beforeblend_pairs=pairs(before),
            top2=dict(ids=ref.iloc[top][ID].tolist(),gain=float(gain[top].sum()),total_gain=float(gain.sum()),share=float(gain[top].sum()/gain.sum())),
            template_incremental_top2=dict(ids=ref.iloc[topdiff][ID].tolist(),gain=float(diff[topdiff].sum()),total_gain=float(diff.sum())),
            tune={name+'_vs_ET':p.prioraudit.comparison(meta.iloc[tr][TARGET].to_numpy(float),v,tunes['ET'],pd.to_datetime(meta.iloc[tr][TIME],utc=True).dt.strftime('%Y-%m-%d').to_numpy()) for name,v in tunes.items() if name!='ET'},
            original_purged_ids_labels_exact=True,finite_routes_unchanged=True)
        print('EXTRATREES_VERIFIED',fold,'top2share',folds[fold]['top2']['share'],'replay',replay,flush=True)
    seasonal={s:{name:p.season_score(folds['F1']['metrics'][s][name],folds['F3']['metrics'][s][name]) for name in folds['F1']['metrics'][s]} for s in folds['F1']['metrics']}
    common_floor={style:{name:p.season_score(folds['F1'][style][name],folds['F3'][style][name]) for name in folds['F1'][style]} for style in ['common_postblend_floor_metrics','guard_reference_beforeblend_metrics']}
    p.write_json(OUT/'audit.json',dict(status='passed',source_sha256=p.sha256(__file__),helper_sha256=p.sha256(Path(p.__file__)),
        first_attempt_source_sha256=p.sha256(Path(first.__file__)),folds=folds,seasonal=seasonal,common_floor_seasonal=common_floor,
        caveat='Originalfullcohorts/rawlabels retained. Removals/commonfitfloor diagnostic only. Firstattempt failedguard-blend assertion beforecomplete; preserved. Noensemblefit/GPU.'))
    print('SEASONAL',seasonal,flush=True)
    print('COMMON_FLOOR',common_floor,flush=True)


if __name__=='__main__':
    main()
