"""Receipt/full-cohort audit of fixed449 or387 extensions versus verified337."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
sys.path.insert(0,str(ROOT/'review_work/breakthrough_20260916/models/verification_id_template'))
import common
from audit import comparison,metric
from taxiout.metrics import season_score
ID,TARGET,TIME=common.ID,common.TARGET,common.MOVEMENT
BASE=ROOT/'private_runs/breakthrough_20260916'
OUT=BASE/'missing/sequence_deep_audit/extended'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def checked(folder):
    rec=common.read_json(folder/'manifest.json')
    assert rec['status']=='complete',str(folder)
    for name,digest in rec['outputs'].items():
        assert common.sha256(folder/name)==digest
    key=rec['family']+'_aobt_allfinite_s20260916'
    for name,digest in rec['source_hashes'].items():
        assert common.sha256(folder.parent/'source_snapshots'/key/Path(name).name)==digest
    assert common.sha256(folder.parent/'protocols'/(key+'.json'))==rec['protocol_sha256']
    assert rec['fit']['steps']==rec['refit']['steps'] and rec['reload_max_abs_delta']==0
    assert rec['fit']['rows']==rec['fit_ids']['fit']['n'] and rec['refit']['rows']==rec['fit_ids']['refit']['n']
    return rec


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extension',choices=['eight','union'],required=True)
    parser.add_argument('--folds',nargs='+',choices=['F1','F3'],default=['F1','F3'])
    args=parser.parse_args()
    out=OUT/args.extension
    out.mkdir(parents=True,exist_ok=True)
    meta_path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert common.sha256(meta_path)==common.read_json(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta_path.name]
    meta=pd.read_parquet(meta_path,columns=[ID,TARGET,TIME,'FLIGHT_ID_mvt','proxy_sec'])
    for fold in args.folds:
        destination=out/(fold+'.json')
        assert not destination.exists()
        olddir=BASE/'deeper_sequence_v2'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916'
        newdir=(BASE/'deeper_sequence8'/f'lightgbm_leaf63_sequence8_aobt_allfinite_{fold}_s20260916' if args.extension=='eight'
            else BASE/'deeper_context_union'/f'lightgbm_leaf63_sequence_aobt_allfinite_{fold}_s20260916')
        old,new=checked(olddir),checked(newdir)
        assert new['fit_ids']==old['fit_ids'] and new['fit']['params']==old['fit']['params']
        assert new['anchor']['feature_receipts']==old['anchor']['feature_receipts']
        if args.extension=='eight':
            cache=BASE/'missing/sequence_flatten8'
            cm=common.read_json(cache/'manifest.json')
            cv=common.read_json(cache/'verification.json')
            assert cm['status']=='complete' and cv['status']=='passed' and cv['all_last4_subsets_exact']
            assert cv['manifest_sha256']==common.sha256(cache/'manifest.json')
            assert common.sha256(cache/'training_features.parquet')==cm['outputs']['training_features.parquet']
            assert len(new['feature_columns'])==449
            assert new['feature_columns']==old['feature_columns'][:225]+cm['features']
            assert set(old['feature_columns'][225:]).issubset(cm['features'])
            payload=new['anchor']['ordered_history']
            assert payload==common.read_json(newdir.parent/'matched_protocol.json')
            assert payload['cache_manifest_sha256']==common.sha256(cache/'manifest.json')
            assert payload['cache_verification_sha256']==common.sha256(cache/'verification.json')
        else:
            cache=BASE/'retrospective_research'
            cm=common.read_json(cache/'manifest.json')
            cv=common.read_json(cache/'verification.json')
            assert cm['status']=='complete' and cv['status']=='passed'
            assert cv['manifest_sha256']==common.sha256(cache/'manifest.json')
            assert common.sha256(cache/'training_features.parquet')==cm['outputs']['training_features.parquet']
            assert len(new['feature_columns'])==387
            assert new['feature_columns']==old['feature_columns']+cm['features']
            payload=new['anchor']['context_union']
            assert payload==common.read_json(newdir.parent/'union_protocol.json')
            assert payload['cache_manifest_sha256']==common.sha256(cache/'manifest.json')
            assert payload['cache_verification_sha256']==common.sha256(cache/'verification.json')
            oracle=BASE/'models/sequence_result_audit/retrospective_oracle.json'
            assert common.sha256(oracle)==payload['independent_oracle_sha256']
            assert common.read_json(oracle)['status']=='passed'
        indices,split,_=common.fold_data(meta,fold,full=True)
        assert common.object_hash(split)==common.object_hash(new['split'])==common.object_hash(old['split'])
        finite=np.isfinite(meta.proxy_sec.to_numpy(float))
        for stage,pos in indices.items():
            rows=pos[finite[pos]]
            assert new['fit_ids'][stage]==dict(n=len(rows),hash=common.object_hash(meta.iloc[rows][ID].tolist()))
        ref,_=common.reference(fold)
        np.testing.assert_array_equal(ref[ID],meta.iloc[indices['score']][ID])
        np.testing.assert_array_equal(ref[TARGET],meta.iloc[indices['score']][TARGET])
        predictions={}
        tunes={}
        for name,folder in [('old337',olddir),('extended',newdir)]:
            frame=pd.read_parquet(folder/'candidate.parquet')
            np.testing.assert_array_equal(frame[ID],ref[ID])
            np.testing.assert_array_equal(frame[TARGET],ref[TARGET])
            p=frame.prediction_sec.to_numpy(float)
            missing=~np.isfinite(ref.proxy_sec.to_numpy(float))
            np.testing.assert_array_equal(p[missing],ref.prediction_sec.to_numpy(float)[missing])
            predictions[name]=p
            tune=pd.read_parquet(folder/'tune_predictions.parquet')
            positions=indices['tune'][finite[indices['tune']]]
            np.testing.assert_array_equal(tune[ID],meta.iloc[positions][ID])
            tunes[name]=metric(meta.iloc[positions][TARGET].to_numpy(float),tune.prediction_sec.to_numpy(float))
        y=ref[TARGET].to_numpy(float)
        a,b=predictions.values()
        comp=comparison(y,a,b,ref.day.to_numpy())
        gain=(a-y)**2-(b-y)**2
        top=np.argsort(gain)[-10:]
        keep=np.ones(len(y),bool)
        keep[top]=False
        result=dict(status='passed',source_sha256=common.sha256(__file__),fold=fold,extension=args.extension,
            old_manifest_sha256=common.sha256(olddir/'manifest.json'),new_manifest_sha256=common.sha256(newdir/'manifest.json'),
            all_source_output_cache_ID_label_param_receipts_match=True,missing_V2_exact=True,
            saved_fullscore_replay_receipts_zero=True,no_independent_model_inference=True,
            metrics={name:metric(y,p) for name,p in predictions.items()},tune=tunes,comparison=comp,
            remove_top10_gain_seconds=metric(y,a,keep)['rmse_sec']-metric(y,b,keep)['rmse_sec'],
            airport_sse_gain={str(ap):float(gain[ref.ADEP_mvt.eq(ap)].sum()) for ap in sorted(ref.ADEP_mvt.unique())})
        common.write_json(destination,result)
        print('EXTENDED_AUDIT',args.extension,fold,'gain',-comp['delta_rmse'],'top10',result['remove_top10_gain_seconds'],flush=True)
    if all((out/(f+'.json')).exists() for f in ['F1','F3']):
        folds={f:common.read_json(out/(f+'.json')) for f in ['F1','F3']}
        score={n:season_score(folds['F1']['metrics'][n],folds['F3']['metrics'][n]) for n in ['old337','extended']}
        common.write_json(out/'summary.json',dict(status='passed',seasonal_rmse=score,source_sha256=common.sha256(__file__),
            gain_seconds=score['old337']-score['extended'],all_day_removals_improve=all(r['comparison']['all_day_removals_improve'] for r in folds.values()),
            caveat='Adaptive exposeddevelopment; receipt audit and allrowscoremetric checks, not new independent modelinference.'))
        print('EXTENDED_SEASONAL',score,flush=True)


if __name__=='__main__':
    main()
