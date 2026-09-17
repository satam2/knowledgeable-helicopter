"""Independent frozen-component replay and disjoint route validation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'review_work/campaign_20260916'))
import common
ID,TARGET=common.ID,common.TARGET
OUT=ROOT/'private_runs/breakthrough_20260916/missing/route_composition_v2'


def load(directory,filename,reference):
    m=common.read_json(directory/'manifest.json')
    assert m['status']=='complete'
    assert common.sha256(directory/filename)==m['outputs'][filename]
    frame=pd.read_parquet(directory/filename)
    np.testing.assert_array_equal(frame[ID],reference[ID])
    np.testing.assert_array_equal(frame[TARGET],reference[TARGET])
    return frame


def main():
    assert not (OUT/'verification.json').exists()
    protocol=common.read_json(OUT/'protocol.json')['declaration']
    assert protocol['source_sha256']==common.sha256(Path(__file__).with_name('compose.py'))
    meta=pd.read_parquet(ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet',columns=[ID,'proxy_sec','ADEP_mvt']).set_index(ID)
    reports={}
    for fold in ['F1','F3']:
        reference,_=common.reference(fold)
        candidate=load(OUT/fold,'candidate.parquet',reference)
        info=meta.loc[reference[ID]]
        proxy=info.proxy_sec.to_numpy(float)
        ordinary=np.isfinite(proxy)&(proxy>=0)&(proxy<=7200)
        missing=~np.isfinite(proxy)
        nonordinary=np.isfinite(proxy)&((proxy<0)|(proxy>7200))
        components={}
        for name,r in protocol['sources'][fold].items():
            directory=Path(r['directory'])
            assert common.sha256(directory/'manifest.json')==r['manifest_sha256']
            assert common.sha256(directory/r['filename'])==r['prediction_sha256']
            components[name]=load(directory,r['filename'],reference).prediction_sec.to_numpy(float)
        v3dir=Path(protocol['sources'][fold]['ordinary']['directory'])
        v3record=common.read_json(v3dir/'manifest.json')
        assert common.sha256(v3dir/'weights.json')==v3record['outputs']['weights.json']
        weights=common.read_json(v3dir/'weights.json')
        tabm=load(v3dir,'tabm_source.parquet',reference).prediction_sec.to_numpy(float)
        lgb=load(v3dir,'lgb_leaf63.parquet',reference).prediction_sec.to_numpy(float)
        alpha=np.array([weights['airports'].get(str(a),{'alpha':weights['global_alpha']})['alpha'] for a in info.ADEP_mvt.to_numpy()])
        assert np.all((alpha>=0)&(alpha<=1))
        ordinary_replay=tabm+alpha*(lgb-tabm)
        np.testing.assert_array_equal(ordinary_replay[ordinary],components['ordinary'][ordinary])
        np.testing.assert_array_equal(lgb[ordinary],components['nonordinary'][ordinary])
        etdir=Path(protocol['sources'][fold]['missing']['directory'])
        et=load(etdir,'candidate.parquet',reference).prediction_sec.to_numpy(float)
        base=reference.prediction_sec.to_numpy(float)
        missing_replay=base+.25*(et-base)
        np.testing.assert_array_equal(missing_replay,components['missing'])
        np.testing.assert_array_equal(et[~missing],base[~missing])
        replay=np.select([missing,nonordinary],[missing_replay,components['nonordinary']],default=ordinary_replay)
        np.testing.assert_array_equal(candidate.prediction_sec,replay)
        expected_route=np.select([missing,nonordinary],['missing','nonordinary'],default='ordinary')
        np.testing.assert_array_equal(candidate.composition_route,expected_route)
        squared=(replay-candidate[TARGET].to_numpy(float))**2
        np.testing.assert_array_equal(squared,candidate.squared_error)
        record=common.read_json(OUT/fold/'manifest.json')
        assert record['no_floor'] and record['no_new_weight_fit']
        assert abs(np.sqrt(squared.mean())-record['metrics']['composed']['rmse_sec'])<1e-10
        reports[fold]=dict(all_rows=len(candidate),all_route_predictions_exact=True,
            V3_saved_airport_weight_arithmetic_exact=True,ET_fixed_quarter_arithmetic_exact=True,
            original_leaf63_saved_prediction_source_exact=True,route_selection_only_observed_proxy=True,
            composition_artifact_sha256=common.sha256(OUT/fold/'candidate.parquet'))
        print('VERIFIED_COMPOSITION',fold,len(candidate),flush=True)
    common.write_json(OUT/'verification.json',dict(status='passed',verifier_sha256=common.sha256(__file__),
        protocol_sha256=common.sha256(OUT/'protocol.json'),folds=reports,
        replay_scope='Frozen rowpredictions/weights/routing independentlyreplayed; componentmodelinference auditedseparately, notrefitted here.'))


if __name__=='__main__':
    main()
