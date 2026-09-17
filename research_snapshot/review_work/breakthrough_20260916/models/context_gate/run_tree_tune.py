"""Single bounded tree gate on saved chronological tune partitions only."""
import lightgbm
import gc
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
from threadpoolctl import threadpool_limits
import tree_gate
import run_tune

ROOT=run_tune.ROOT
OUT=ROOT/'private_runs/breakthrough_20260916/models/context_gate/tree_early60_late40_v1'
read_json,write_json,sha256,object_hash,utc_now=run_tune.read_json,run_tune.write_json,run_tune.sha256,run_tune.object_hash,run_tune.utc_now
ID,TARGET,MOVEMENT=run_tune.ID,run_tune.TARGET,run_tune.MOVEMENT


def budget():
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',info.rss)
    if peak>1024**3:
        raise MemoryError('Bounded treegate exceeded1GiB process memory budget')
    return int(peak)


def main():
    pa.set_cpu_count(2)
    pa.set_io_thread_count(1)
    OUT.mkdir(parents=True,exist_ok=False)
    original=read_json(run_tune.OUT/'summary.json')
    conventions=ROOT/'private_runs/breakthrough_20260916/missing/source_conventions'
    marker=read_json(conventions/'manifest.json')
    assert sha256(conventions/'features.parquet')==marker['feature_sha256']
    protocol={'created_utc':utc_now(),'source_hashes':{p.name:sha256(p) for p in [Path(__file__),Path(tree_gate.__file__),Path(run_tune.__file__)]},
              'original_tune_protocol_sha256':sha256(run_tune.OUT/'protocol.json'),'chronological_summary_sha256':sha256(run_tune.OUT/'summary.json'),
              'parameters':tree_gate.PARAMS,'trees':tree_gate.TREES,
              'features':'85verifiedsourceconventions+airport+signedLGBminusTabMdisagreement',
              'objective':'DirectrawmixtureSSE:grad2d*(d*alpha-(Y-pTabM))/mean(d²);hess2d²/mean(d²);no divisionsbyd',
              'zero_d':'Omittedonlyfromgatefitbecauseconstantloss;all originaltunerowsretainedinmetrics',
              'initialization':'Earlyglobalpairalpha passedasinit_score; explicitlyaddedback toboosterprediction',
              'prediction':'clipalpha0..1 then pTabM+alpha*d; do not cliplabel or disagreement',
              'selection':'Singlefixed100treedepth2 fit;early60/late40exactpreviouspartitions;nohypergrid/no scorefiles/no fulltuneretrain',
              'memory':'FilteredconventionIDs andtunemetadataonly;2CPUthreads;1GiBprocessbudget',
              'caveat':'Unboundedtraining surrogate differsfromclipped inference objective;actualclippedlateMSEreported. Tunealreadyusedforexpertepochselection.'}
    write_json(OUT/'protocol.json',protocol)
    results={}
    for fold in ['F1','F3']:
        began=time.monotonic()
        destination=OUT/fold
        destination.mkdir()
        earlier=original['folds'][fold]
        path=run_tune.OUT/fold/'tune_diagnostic.parquet'
        assert sha256(path)==earlier['outputs'][path.name]
        tune=pd.read_parquet(path)
        assert tune[ID].is_unique and len(tune)==earlier['original_ordinary_tune_n']
        early=tune.early.to_numpy(bool)
        assert object_hash(tune.loc[early,ID].tolist())==earlier['early_id_hash']
        assert object_hash(tune.loc[~early,ID].tolist())==earlier['late_id_hash']
        start,end=next(iter(earlier['receipts'].values()))['fit_ids']['tune']['n'],None
        timestamps=pd.to_datetime(tune[MOVEMENT],utc=True)
        monthstart=timestamps.min().normalize().replace(day=1)
        monthend=monthstart+pd.offsets.MonthBegin(1)
        meta=pd.read_parquet(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',
                             columns=[ID,'ADEP_mvt','proxy_sec'],filters=[(MOVEMENT,'>=',monthstart),(MOVEMENT,'<',monthend)]).set_index(ID)
        meta=meta.loc[tune[ID]]
        assert np.isfinite(meta.proxy_sec).all() and meta.proxy_sec.between(0,7200).all()
        features=pd.read_parquet(conventions/'features.parquet',columns=[ID,*marker['columns']],
                                 filters=[(ID,'in',tune[ID].tolist())]).set_index(ID).loc[tune[ID]].reset_index(drop=True)
        assert len(features.columns)==85
        features['airport']=pd.Categorical(meta.ADEP_mvt.astype(str).to_numpy())
        first,second=tune.tabm_source.to_numpy(),tune.lgb63.to_numpy()
        features['expert_disagreement_sec']=second-first
        y=tune[TARGET].to_numpy()
        budget()
        with threadpool_limits(2):
            baseline=run_tune.simplex.solve(y[early],first[early],second[early],features.airport.to_numpy()[early])
            assert baseline==earlier['global2']
            model,evidence=tree_gate.fit(features.iloc[np.flatnonzero(early)],y[early],first[early],second[early],baseline['global_alpha'])
            prediction,alpha,unbounded=tree_gate.predict(model,features,first,second)
        budget()
        joblib.dump(model,destination/'early_model.joblib')
        replay,_,_=tree_gate.predict(joblib.load(destination/'early_model.joblib'),features,first,second)
        np.testing.assert_array_equal(prediction,replay)
        metrics={name:{'early_rmse':run_tune.rmse(y[early],value[early]),'late_rmse':run_tune.rmse(y[~early],value[~early])}
                 for name,value in {'tree_gate':prediction,'v3_global':tune.global2.to_numpy(),'v3_airport':tune.airport2.to_numpy(),
                                     'global3':tune.global3.to_numpy(),'airport3':tune.airport3.to_numpy()}.items()}
        tune[[ID,TARGET,MOVEMENT,'early']].assign(prediction_sec=prediction,alpha=alpha,unbounded_alpha=unbounded).to_parquet(destination/'tune_predictions.parquet',index=False)
        record={'status':'complete','metrics':metrics,'fit':evidence,'early_rows':int(early.sum()),'late_rows':int((~early).sum()),
                'zero_d_late_rows':int(((second==first)&~early).sum()),'late_clipped_rows':int(((unbounded<0)|(unbounded>1))[~early].sum()),
                'late_alpha_mean':float(alpha[~early].mean()),'peak_rss_bytes':budget(),'runtime_sec':time.monotonic()-began,
                'outputs':{p.name:sha256(p) for p in destination.iterdir() if p.is_file()}}
        write_json(destination/'manifest.json',record)
        results[fold]=record
        print('TREE_GATE_TUNE',fold,metrics,'peakMiB',record['peak_rss_bytes']/1024**2,flush=True)
        del features,meta,tune,model,replay,prediction,alpha,unbounded
        gc.collect()
    write_json(OUT/'summary.json',{'created_utc':utc_now(),'folds':results,'protocol_sha256':sha256(OUT/'protocol.json'),
                                 'score_reads':False,'full_tune_fit':False,'peak_rss_bytes':budget()})


if __name__=='__main__':
    main()
