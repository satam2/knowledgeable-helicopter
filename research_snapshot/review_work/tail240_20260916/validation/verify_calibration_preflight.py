"""Independent chronological calibration partitions and input-only contracts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import psutil
import validate_candidate as v

ROOT,ID,TARGET,TIME=v.ROOT,v.ID,v.TARGET,v.common.MOVEMENT
BASE=ROOT/'private_runs/tail240_20260916/forensics/current_calibration/v1'
OUT=ROOT/'private_runs/tail240_20260916/validation/current_calibration_preflight_v1'
ENSEMBLE=ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=v.read_json(BASE/'protocol.json')
    assert protocol['source_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/current_calibration/run_v1.py')
    assert protocol['tests_sha256']==v.sha256(ROOT/'review_work/tail240_20260916/forensics/current_calibration/test_v1.py')
    for path,digest in protocol['dependencies'].items():assert v.sha256(ROOT/path)==digest
    meta=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert v.sha256(meta)==protocol['meta_sha256']
    dataset=ds.dataset(meta,format='parquet');checks={}
    assert v.sha256(ENSEMBLE/'preparation.json')==protocol['preparation_sha256']
    for fold in ['F1','F3']:
        folder=BASE/fold;receipt=v.read_json(folder/'preflight.json');binding=protocol['folds'][fold]
        assert receipt['status']=='passed' and not receipt['model_fit']
        assert receipt['source_sha256']==protocol['source_sha256'] and receipt['protocol_sha256']==v.sha256(BASE/'protocol.json')
        assert v.sha256(folder/'inputs.parquet')==receipt['inputs_sha256']
        inputs=pd.read_parquet(folder/'inputs.parquet')
        assert inputs[ID].is_unique
        raw=dataset.to_table(columns=[ID,'FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec','ADEP_mvt'],filter=ds.field(ID).isin(inputs[ID].to_numpy()),use_threads=False).to_pandas().set_index(ID).loc[inputs[ID]]
        for name in ['FLIGHT_ID_mvt',TIME,TARGET,'proxy_sec']:
            np.testing.assert_array_equal(inputs[name],raw[name])
        np.testing.assert_array_equal(inputs.ADEP_mvt.astype(str),raw.ADEP_mvt.astype(str))
        assert inputs.proxy_sec.between(0,7200).all() and np.isfinite(inputs[TARGET]).all()
        alignedpath=ENSEMBLE/f'{fold}_aligned_tune.parquet'
        assert v.sha256(alignedpath)==binding['aligned_tune_sha256']
        aligned=pd.read_parquet(alignedpath);np.testing.assert_array_equal(aligned[ID],inputs[ID])
        np.testing.assert_array_equal(aligned[TARGET],inputs[TARGET])
        neuralroot=ROOT/'private_runs/tail240_20260916/state/neural_context'/('v1' if fold=='F1' else 'v3')/fold
        assert v.sha256(neuralroot/'manifest.json')==binding['neural_manifest_sha256']
        assert v.sha256(neuralroot/'tune_predictions.parquet')==binding['neural_tune_sha256']
        neural=pd.read_parquet(neuralroot/'tune_predictions.parquet').set_index(ID).loc[inputs[ID]]
        np.testing.assert_array_equal(neural[TARGET],inputs[TARGET])
        for name in protocol['experts']:
            expected=neural.prediction_sec if name=='tabm_ple8' else aligned[name]
            np.testing.assert_array_equal(inputs[name],expected)
        weightpath=ENSEMBLE/f'{fold}_weights.json';assert v.sha256(weightpath)==binding['weights_sha256']
        weights=v.read_json(weightpath)
        np.testing.assert_allclose(inputs[protocol['experts']].to_numpy()@np.asarray(weights['global']),inputs.frozen_current_sec,rtol=0,atol=1e-8)
        ordered=inputs[[ID,TIME]].sort_values([TIME,ID],kind='stable')
        cutoff=ordered[TIME].iloc[len(inputs)*3//5]
        early=inputs[TIME].to_numpy()<cutoff;late=~early
        flights=set(inputs.loc[late,'FLIGHT_ID_mvt'].dropna())
        purged=early&np.array([pd.notna(fid) and fid in flights for fid in inputs.FLIGHT_ID_mvt])
        fit=early&~purged
        for key,value in [('early',early),('late',late),('purged',purged),('fit',fit)]:
            np.testing.assert_array_equal(inputs[key],value)
        assert not set(inputs.loc[fit,'FLIGHT_ID_mvt'].dropna())&set(inputs.loc[late,'FLIGHT_ID_mvt'].dropna())
        assert receipt['fit_rows']==int(fit.sum()) and receipt['late_rows']==int(late.sum()) and receipt['purged_early_rows']==int(purged.sum())
        assert receipt['cutoff']==str(cutoff)
        for key,ids in [('original_ids_hash',inputs[ID]),('fit_ids_hash',inputs.loc[fit,ID]),('late_ids_hash',inputs.loc[late,ID])]:assert receipt[key]==v.object_hash(ids.tolist())
        assert receipt['raw_labels_hash']==v.object_hash(inputs[TARGET].tolist())
        for source in receipt['source_receipts']:assert v.sha256(source['path'])==source['sha256']
        checks[fold]=dict(rows=len(inputs),fit_rows=int(fit.sum()),late_rows=int(late.sum()),purged_rows=int(purged.sum()),cutoff=str(cutoff),all_inputs_and_partitions_exact=True)
    info=psutil.Process().memory_info();peak=max(info.rss,info.peak_wset)
    assert peak<2*1024**3 and psutil.virtual_memory().available>=8*1024**3
    v.write_json(OUT/'receipt.json',dict(status='passed',source_sha256=v.sha256(__file__),protocol_sha256=v.sha256(BASE/'protocol.json'),folds=checks,raw_labels_checked_only_for_query_IDs=True,source_bindings_exact=True,no_fit=True,no_gpu=True,peak_bytes=peak))
    print('VERIFIED_CALIBRATION_PREFLIGHT',checks,'peak',peak,flush=True)


if __name__=='__main__':
    pa.set_cpu_count(1);pa.set_io_thread_count(1);main()
