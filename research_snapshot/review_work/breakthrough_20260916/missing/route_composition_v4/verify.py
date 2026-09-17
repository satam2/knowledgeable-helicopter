"""Independent all-row coefficient, routing and sensitivity replay for final9."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET=common.ID,common.TARGET
BASE=ROOT/'private_runs/breakthrough_20260916'
OUT=BASE/'missing/route_composition_v4'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def load(folder,filename,reference):
    marker=common.read_json(folder/'manifest.json')
    assert marker['status']=='complete'
    assert common.sha256(folder/filename)==marker['outputs'][filename]
    frame=pd.read_parquet(folder/filename)
    np.testing.assert_array_equal(frame[ID],reference[ID])
    np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
    return frame


def main():
    assert not (OUT/'verification.json').exists()
    declared=common.read_json(OUT/'protocol.json')['declaration']
    assert declared['source_sha256']==common.sha256(Path(__file__).with_name('compose.py'))
    execution=common.read_json(OUT/'execution_inputs.json')
    assert execution['predeclaration_sha256']==common.sha256(OUT/'protocol.json')
    stack=Path(declared['final9_directory'])
    for name,digest in execution['stacker_receipts'].items():
        assert common.sha256(stack/name)==digest
    assert common.read_json(stack/'independent_verification.json')['status']=='passed'
    preparation=common.read_json(stack/'preparation.json')
    reports={}
    for fold in declared['folds']:
        reference,_=common.reference(fold)
        proxy=reference.proxy_sec.to_numpy(float)
        missing=~np.isfinite(proxy)
        ordinary=~missing&(proxy>=0)&(proxy<=7200)
        nonordinary=~missing&~ordinary
        base=reference.prediction_sec.to_numpy(float)
        y=reference[TARGET].to_numpy(float)
        weights=common.read_json(stack/(fold+'_weights.json'))
        assert common.sha256(stack/(fold+'_weights.json'))==preparation['folds'][fold]['weights_sha256']
        assert len(weights['experts'])==9
        experts=[]
        for name in weights['experts']:
            receipt=preparation['sources'][fold][name]
            folder=Path(receipt['directory'])
            assert common.sha256(folder/'manifest.json')==receipt['manifest_sha256']
            assert common.sha256(folder/'candidate.parquet')==receipt['outputs']['candidate.parquet']
            experts.append(load(folder,'candidate.parquet',reference).prediction_sec.to_numpy(float))
        p=np.column_stack(experts)
        components={}
        for name,receipt in declared['fixed_sources'][fold].items():
            folder=Path(receipt['directory'])
            assert common.sha256(folder/'manifest.json')==receipt['manifest_sha256']
            assert common.sha256(folder/receipt['filename'])==receipt['prediction_sha256']
            components[name]=load(folder,receipt['filename'],reference).prediction_sec.to_numpy(float)
        et_folder=Path(declared['fixed_sources'][fold]['missing']['directory'])
        et=load(et_folder,'candidate.parquet',reference).prediction_sec.to_numpy(float)
        np.testing.assert_array_equal(et[~missing],base[~missing])
        et_blend=base+.25*(et-base)
        np.testing.assert_array_equal(et_blend,components['missing'])
        np.testing.assert_array_equal(p[:,weights['experts'].index('lgb63')],components['nonordinary'])
        reports[fold]={}
        for variant in declared['variants']:
            coefficients=(np.tile(weights['global'],(ordinary.sum(),1)) if variant=='global9'
                else np.array([weights['airports'].get(str(a),weights['global']) for a in reference.ADEP_mvt.to_numpy()[ordinary]]))
            assert np.all(coefficients>=0) and np.allclose(coefficients.sum(1),1.)
            mixed=base.copy()
            mixed[ordinary]=np.sum(p[ordinary]*coefficients,axis=1)
            saved=load(stack/f'{fold}_score',variant+'.parquet',reference)
            np.testing.assert_array_equal(mixed,saved.prediction_sec)
            replay=np.select([missing,nonordinary],[et_blend,components['nonordinary']],default=mixed)
            current=load(OUT/variant/fold,'candidate.parquet',reference)
            np.testing.assert_array_equal(replay,current.prediction_sec)
            np.testing.assert_array_equal(replay[~ordinary],components[variant][~ordinary])
            np.testing.assert_array_equal(current.composition_route,np.select([missing,nonordinary],['missing','nonordinary'],default='ordinary'))
            pe=(replay-y)**2
            np.testing.assert_array_equal(pe,current.squared_error)
            marker=common.read_json(OUT/variant/fold/'manifest.json')
            assert abs(np.sqrt(pe.mean())-marker['metrics']['composed']['rmse_sec'])<1e-10
            controls={'V2':base,'previous_composition':components[variant],'own_ordinary':mixed}
            for name,value in controls.items():
                ce=(value-y)**2
                for row in marker['comparisons'][name]['day_removals']:
                    keep=reference.day.to_numpy()!=row['removed_day']
                    delta=float(np.sqrt(pe[keep].mean())-np.sqrt(ce[keep].mean()))
                    assert abs(delta-row['delta_rmse'])<1e-10
                if name!='own_ordinary':
                    top=np.argsort(ce-pe)[-10:]
                    np.testing.assert_array_equal(reference.iloc[top][ID],marker['remove_top10_gain'][name]['ids'])
                    keep=~np.isin(np.arange(len(y)),top)
                    gain=float(np.sqrt(ce[keep].mean())-np.sqrt(pe[keep].mean()))
                    assert abs(gain-marker['remove_top10_gain'][name]['gain_seconds'])<1e-10
            gains=np.where(missing,(base-y)**2-pe,-np.inf)
            top=np.argsort(gains)[-2:]
            np.testing.assert_array_equal(reference.iloc[top][ID],marker['missing_top2']['ids'])
            keep=~np.isin(np.arange(len(y)),top)
            assert abs(np.sqrt(pe[keep].mean())-marker['remove_missing_top2']['composed']['rmse_sec'])<1e-10
            reports[fold][variant]=dict(rows=len(replay),full_coefficient_and_route_replay_exact=True,
                all_day_top10_top2_sensitivities_verified=True,exceptional_and_missing_exact_previous=True,
                candidate_sha256=common.sha256(OUT/variant/fold/'candidate.parquet'),manifest_sha256=common.sha256(OUT/variant/fold/'manifest.json'))
            print('FINAL9_COMPOSITION_VERIFIED',fold,variant,flush=True)
    common.write_json(OUT/'verification.json',dict(status='passed',source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),execution_inputs_sha256=common.sha256(OUT/'execution_inputs.json'),folds=reports,
        scope='Independent fullsavedcoefficient/ET/routing/metrics/sensitivity replay. No modeltraining,inference,weightsfit,projection,or variantselection.'))


if __name__=='__main__':
    main()
