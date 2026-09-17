"""Independent gated normalized missing-route native replay and full evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import lightgbm
from pathlib import Path
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import validate_candidate as validation

ROOT=validation.ROOT
sys.path.insert(0,str(ROOT/'review_work/tail240_20260916/models'))
import normalized_missing_tune as producer
BASE=ROOT/'private_runs/tail240_20260916/models/normalized_missing_refit_v2'
OUT=ROOT/'private_runs/tail240_20260916/validation/normalized_refit_v2'
read,sha,oh,write=validation.read_json,validation.sha256,validation.object_hash,validation.write_json
ID,TARGET=validation.ID,validation.TARGET


def main():
    protocol=read(BASE/'protocol.json')
    for path,digest in protocol['source_hashes'].items():
        assert sha(ROOT/path)==digest
    assert protocol['prior_source_sha256']==sha(producer.__file__)
    assert protocol['prior_protocol_sha256']==sha(producer.OUT/'protocol.json')
    assert protocol['prior_summary_sha256']==sha(producer.OUT/'summary.json')
    binding=read(validation.BINDING)
    assert protocol['baseline_binding_sha256']==sha(validation.BINDING)
    x,meta=producer.shared.load_missing()
    oldcolumns=read(ROOT/'private_runs/tail240_20260916/state/models_v2/control/F1/manifest.json')['feature_columns']
    assert list(x)==oldcolumns and len(x.columns)==76
    missing=~np.isfinite(meta.proxy_sec.to_numpy(float))
    results={}
    for fold in ('F1','F3'):
        idx,split,_=producer.common.fold_data(meta,fold,full=True)
        rows={stage:pos[missing[pos]] for stage,pos in idx.items()}
        frames={stage:x.loc[meta.iloc[rows[stage]][ID]] for stage in ('refit','score')}
        baseline_path=Path(binding['folds'][fold]['prediction_path'])
        assert sha(baseline_path)==binding['folds'][fold]['prediction_sha256']
        baseline=pd.read_parquet(baseline_path)
        for arm in protocol['arms']:
            folder=BASE/'models'/fold/arm
            record=read(folder/'manifest.json')
            assert record['status']=='complete' and record['protocol_sha256']==sha(BASE/'protocol.json')
            assert record['fit_ids']==binding['folds'][fold]['cohorts']['missing_nm']
            assert oh(record['split'])==binding['folds'][fold]['split_hash']
            previous=read(producer.OUT/fold/arm/'manifest.json')
            assert sha(producer.OUT/fold/arm/'manifest.json')==record['tune_manifest_sha256']
            assert record['steps']==previous['steps']
            for name,digest in record['outputs'].items():
                assert sha(folder/name)==digest
            gate_path=Path(record['gate_receipt_path'])
            assert sha(gate_path)==record['gate_receipt_sha256']
            gate=read(gate_path)
            assert gate['status']=='passed' and gate['advance_normalized_refit'] and gate['source_sha256']==sha(producer.__file__)
            fitted=joblib.load(folder/'model.joblib')
            assert fitted['encoder'].columns==oldcolumns
            newencoder=producer.FrameEncoder().fit(frames['refit'])
            assert fitted['encoder'].categories==newencoder.categories
            scales={}
            for stage,frame in frames.items():
                schedule=frame.schedule_proxy_sec.to_numpy(float)
                good=np.isfinite(schedule)&(schedule!=-999999)
                scales[stage]=np.ones(len(frame)) if arm=='unscaled' else np.sqrt(3600**2+np.where(good,schedule-900,0)**2)
            normalizer=float(np.mean(scales['refit']**2))
            assert normalizer==record['normalizer']
            assert np.isfinite(scales['refit']).all() and (scales['refit']>0).all()
            pred=900+scales['score']*fitted['model'].predict(fitted['encoder'].transform(frames['score']),num_threads=2)
            saved=pd.read_parquet(folder/'missing_predictions.parquet')
            np.testing.assert_array_equal(saved[ID],frames['score'].index)
            np.testing.assert_array_equal(saved.scale,scales['score'])
            np.testing.assert_array_equal(saved.prediction_sec,pred)
            mask=missing[idx['score']]
            for variant,weight in protocol['variants'].items():
                exchange=BASE/f'{arm}_{variant}'
                manifest=read(exchange/'manifest.json')
                assert manifest['status']=='complete'
                actual=pd.read_parquet(exchange/f'{fold}.parquet')
                expected=baseline.prediction_sec.to_numpy().copy()
                expected[mask]+=weight*(pred-expected[mask])
                np.testing.assert_array_equal(actual[ID],baseline[ID])
                np.testing.assert_array_equal(actual.prediction_sec,expected)
            results[f'{fold}_{arm}']={'manifest_sha256':sha(folder/'manifest.json'),'native_score_replay_max_abs_delta':0.,
                'missing_rows':len(pred),'refit_rows':len(frames['refit']),'steps':record['steps'],'refit_encoder_rebuilt_exact':True}
            validation.guard()
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'replay.json',{'status':'passed','source_sha256':sha(__file__),'producer_protocol_sha256':sha(BASE/'protocol.json'),
        'models':results,'peak_rss_bytes':validation.guard(),'scope':'All original missing refit rows and score predictions, rawobservable scale, refit-only encoder, frozen tune steps, gate receipt and composed scorevectors independently checked. Unmodified scorelabels evaluated separately.'})
    for arm in protocol['arms']:
        for variant in protocol['variants']:
            subprocess.run([sys.executable,'-B','-u',str(Path(validation.__file__)),'--candidate',str(BASE/f'{arm}_{variant}'),
                '--output',str(OUT/f'{arm}_{variant}_evaluation')],check=True)
    print('NORMALIZED_REFIT_VERIFIED',flush=True)


if __name__=='__main__':
    main()
