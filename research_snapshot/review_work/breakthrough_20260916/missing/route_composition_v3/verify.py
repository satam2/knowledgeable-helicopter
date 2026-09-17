"""Independent full-row coefficient and route replay for both fixed compositions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET = common.ID,common.TARGET
BASE = ROOT/'private_runs/breakthrough_20260916'
OUT = BASE/'missing/route_composition_v3'
STACK = BASE/'models/context_gate/simplex4_context_v1'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def load(directory,filename,reference):
    marker = common.read_json(directory/'manifest.json')
    assert marker['status']=='complete'
    assert common.sha256(directory/filename)==marker['outputs'][filename]
    frame = pd.read_parquet(directory/filename)
    np.testing.assert_array_equal(frame[ID],reference[ID])
    np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
    return frame


def main():
    assert not (OUT/'verification.json').exists()
    declared = common.read_json(OUT/'protocol.json')['declaration']
    assert declared['source_sha256']==common.sha256(Path(__file__).with_name('compose.py'))
    for filename,digest in declared['stacker_receipts'].items():
        assert common.sha256(STACK/filename)==digest
    preparation = common.read_json(STACK/'preparation.json')
    reports = {}
    for fold in ['F1','F3']:
        reference,_ = common.reference(fold)
        proxy = reference.proxy_sec.to_numpy(float)
        ordinary = np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        missing = ~np.isfinite(proxy)
        nonordinary = np.isfinite(proxy)&((proxy<0)|(proxy>7200))
        base = reference.prediction_sec.to_numpy(float)
        weights = common.read_json(STACK/(fold+'_weights.json'))
        assert common.sha256(STACK/(fold+'_weights.json'))==preparation['folds'][fold]['weights_sha256']
        p = []
        for name in weights['experts']:
            receipt = preparation['sources'][fold][name]
            folder = Path(receipt['directory'])
            assert common.sha256(folder/'manifest.json')==receipt['manifest_sha256']
            assert common.sha256(folder/'candidate.parquet')==receipt['outputs']['candidate.parquet']
            p.append(load(folder,'candidate.parquet',reference).prediction_sec.to_numpy(float))
        p = np.column_stack(p)
        components = {}
        for name,receipt in declared['sources'][fold].items():
            folder = Path(receipt['directory'])
            assert common.sha256(folder/'manifest.json')==receipt['manifest_sha256']
            assert common.sha256(folder/receipt['filename'])==receipt['prediction_sha256']
            components[name] = load(folder,receipt['filename'],reference).prediction_sec.to_numpy(float)
        et_folder = Path(declared['sources'][fold]['missing']['directory'])
        et = load(et_folder,'candidate.parquet',reference).prediction_sec.to_numpy(float)
        np.testing.assert_array_equal(et[~missing],base[~missing])
        missing_replay = base+.25*(et-base)
        np.testing.assert_array_equal(missing_replay,components['missing'])
        np.testing.assert_array_equal(p[:,weights['experts'].index('lgb63')],components['nonordinary'])
        reports[fold] = {}
        for variant in declared['variants']:
            coeff = (np.tile(weights['global'],(int(ordinary.sum()),1)) if variant=='global4'
                else np.array([weights['airports'].get(str(a),weights['global']) for a in reference.ADEP_mvt.to_numpy()[ordinary]]))
            assert np.all(coeff>=0) and np.allclose(coeff.sum(1),1)
            mixed = base.copy()
            mixed[ordinary] = np.sum(p[ordinary]*coeff,axis=1)
            np.testing.assert_array_equal(mixed,components[variant])
            replay = np.select([missing,nonordinary],[missing_replay,components['nonordinary']],default=mixed)
            current = load(OUT/variant/fold,'candidate.parquet',reference)
            np.testing.assert_array_equal(current.prediction_sec,replay)
            np.testing.assert_array_equal(replay[~ordinary],components['previous_composition'][~ordinary])
            np.testing.assert_array_equal(current.composition_route,np.select([missing,nonordinary],['missing','nonordinary'],default='ordinary'))
            squared = (replay-reference[TARGET].to_numpy(float))**2
            np.testing.assert_array_equal(squared,current.squared_error)
            marker = common.read_json(OUT/variant/fold/'manifest.json')
            assert abs(np.sqrt(squared.mean())-marker['metrics']['composed']['rmse_sec'])<1e-10
            reports[fold][variant] = dict(rows=len(replay), full_coefficient_and_route_replay_exact=True,
                nonordinary_and_missing_exact_previous=True, squared_errors_exact=True,
                candidate_sha256=common.sha256(OUT/variant/fold/'candidate.parquet'),
                manifest_sha256=common.sha256(OUT/variant/fold/'manifest.json'))
            print('VERIFIED',fold,variant,len(replay),flush=True)
    common.write_json(OUT/'verification.json',dict(status='passed',source_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),folds=reports,
        scope='Full saved expert coefficient,ET25percent,and disjoint routing replay. No model inference,training,weight fit or projection.'))


if __name__=='__main__':
    main()
