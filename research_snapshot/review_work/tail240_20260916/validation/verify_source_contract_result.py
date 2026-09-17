"""Bind the new checked loader to independent coverage and native-model evidence."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import lightgbm as lgb
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from audit_union387_sources import ROOT,read,sha,write,guard,object_hash

BASE=ROOT/'private_runs/tail240_20260916/forensics/source_contract/v2'
ID,TIME='MVT_ID_mvt','MVT_TIME_UTC_mvt'


def main():
    protocol,receipt,coverage=read(BASE/'protocol.json'),read(BASE/'receipt.json'),read(BASE/'contract_receipt.json')
    assert receipt['status']==coverage['status']=='passed'
    assert receipt['protocol_sha256']==sha(BASE/'protocol.json') and receipt['contract_receipt_sha256']==sha(BASE/'contract_receipt.json')
    for relative,digest in protocol['source_hashes'].items():assert sha(ROOT/relative)==digest
    audit=ROOT/'private_runs/tail240_20260916/validation/union387_source_audit_v1'
    prior=read(audit/'receipt.json');assert prior['status']=='passed'
    numeric=read(audit/'numeric_columns.json')['stats'];source_rows=read(audit/'source_coverage.json')
    prior_sources={str((ROOT/entry['path']).resolve()):entry for entry in source_rows}
    assert len(protocol['sources'])==len(coverage['sources'])==19
    for source,bound in zip(coverage['sources'],protocol['sources']):
        for name in ('path','sha256','fill','columns'):assert source[name]==bound[name]
        original=prior_sources[str(Path(source['path']).resolve())]
        assert source['sha256']==original['sha256']==sha(Path(source['path']))
        assert source['columns']==original['columns'] and source['fill']==original['loader_fill']
        assert source['requested_rows']==sum(original['rows'].values())
        assert source['outside_requested_rows']==pq.ParquetFile(source['path']).metadata.num_rows-source['requested_rows']
    for name,counts in coverage['coverage'].items():
        assert counts['rows']==996700
        if name in numeric:
            assert counts['preexisting_sentinel']==sum(stage['source_finite_sentinel'] for stage in numeric[name].values())
            assert counts['nonfinite_input']==sum(stage['source_nan']+stage['source_posinf']+stage['source_neginf'] for stage in numeric[name].values())
        else:assert counts['preexisting_sentinel']==counts['nonfinite_input']==0
    reference=Path(protocol['reference_matrix_path'])
    assert sha(reference)==protocol['reference_matrix_sha256']==sha(BASE/'matrix.float32')==receipt['matrix_byte_sha256']
    half_verified=ROOT/'private_runs/tail240_20260916/validation/half_curve_F1_v3/receipt.json'
    assert read(half_verified)['status']=='passed' and read(half_verified)['all_full_and_half_fit_tune_cells_exact']
    meta=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(meta)==read(ROOT/'private_runs/screening_230/reports/data_audit.json')['artifacts'][meta.name]
    selected=[]
    for batch in pq.ParquetFile(meta).iter_batches(batch_size=8192,columns=[ID,TIME,'proxy_sec'],use_threads=False):
        frame=batch.to_pandas();selected.append(frame.loc[frame[TIME].ge(pd.Timestamp('2025-01-01',tz='UTC'))&frame[TIME].lt(pd.Timestamp('2025-07-01',tz='UTC'))&np.isfinite(frame.proxy_sec)])
    frame=pd.concat(selected,ignore_index=True);assert object_hash(frame[ID].tolist())==protocol['ids_hash']
    tune=frame.loc[frame[TIME].ge(pd.Timestamp('2025-06-01',tz='UTC'))]
    assert len(frame)==protocol['rows']==996700 and len(frame)-len(tune)==protocol['nfit']==816060
    native=ROOT/'private_runs/tail240_20260916/models/following_groups_tune_v1/F1/control387'
    assert sha(native/'manifest.json')==protocol['native_manifest_sha256']
    marker=read(native/'manifest.json');assert sha(native/'model.txt')==marker['outputs']['model.txt']
    control=ROOT/'private_runs/breakthrough_20260916/deeper_context_union/lightgbm_leaf63_sequence_aobt_allfinite_F1_s20260916'
    assert sha(control/'manifest.json')==protocol['control_manifest_sha256']
    assert sha(control/'tune_predictions.parquet')==read(control/'manifest.json')['outputs']['tune_predictions.parquet']
    old=pd.read_parquet(control/'tune_predictions.parquet');np.testing.assert_array_equal(old[ID],tune[ID])
    matrix=np.memmap(BASE/'matrix.float32',mode='r',dtype='float32',shape=(996700,387),order='F')
    model=lgb.Booster(model_file=str(native/'model.txt'))
    assert model.feature_name()==protocol['columns']
    prediction=np.empty(len(tune));proxy=tune.proxy_sec.to_numpy(float)
    for start in range(0,len(tune),8192):
        stop=min(start+8192,len(tune));prediction[start:stop]=model.predict(matrix[816060+start:816060+stop],num_threads=1)+proxy[start:stop]
        assert guard()<2*1024**3
    np.testing.assert_array_equal(prediction,old.prediction_sec)
    out=ROOT/'private_runs/tail240_20260916/validation/source_contract_v2';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='passed',source_sha256=sha(Path(__file__)),protocol_sha256=sha(BASE/'protocol.json'),producer_receipt_sha256=sha(BASE/'receipt.json'),
        prior_exactcoverage_audit_sha256=sha(audit/'receipt.json'),prior_fullmatrix_rawreconstruction_sha256=sha(half_verified),
        independent_tests_sha256=sha(Path(__file__).with_name('test_source_contract_independent.py')),
        producer_tests_passed=9,independent_tests_passed=4,all19sourcebindings_and387coveragecounts_exact=True,
        unchanged_matrix_byte_hash=receipt['matrix_byte_sha256'],values=385722900,native_replay_rows=len(tune),native_max_abs_delta_sec=0.,
        no_target_labels_or_newfit_or_GPU=True,peak_bytes=guard(),
        scope='Opt-in checked-loader wrapper preserves frozen outputs and rejects tested invalid sources. Frozen historical loader unmodified; future experiments must adopt wrapper explicitly. Does not validate semantic truth of finite values or upstream rounding.')
    write(out/'receipt.json',result);print(result,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
