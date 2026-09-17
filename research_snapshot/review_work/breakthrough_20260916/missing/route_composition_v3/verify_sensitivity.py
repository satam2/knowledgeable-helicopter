"""Independent arithmetic check of saved day and influential-row diagnostics."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'private_runs/breakthrough_20260916/missing/route_composition_v3'
ID,TARGET='MVT_ID_mvt','TAXITIME_SEC_mvt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination=OUT/'sensitivity_verification.json'
    assert not destination.exists()
    declared=json.loads((OUT/'protocol.json').read_text())['declaration']
    receipts={}
    for fold in ['F1','F3']:
        reference_path=ROOT/'private_runs/next_230/models'/f'clock_and_rome_ensemble_{fold}_s20260910'/'score_predictions.parquet'
        reference=pd.read_parquet(reference_path)
        previous_receipt=declared['sources'][fold]['previous_composition']
        previous_path=Path(previous_receipt['directory'])/previous_receipt['filename']
        assert sha(previous_path)==previous_receipt['prediction_sha256']
        previous=pd.read_parquet(previous_path)
        np.testing.assert_array_equal(reference[ID],previous[ID])
        y=reference[TARGET].to_numpy(float)
        missing=~np.isfinite(reference.proxy_sec.to_numpy(float))
        for variant in declared['variants']:
            folder=OUT/variant/fold
            rec=json.loads((folder/'manifest.json').read_text())
            frame=pd.read_parquet(folder/'candidate.parquet')
            assert sha(folder/'candidate.parquet')==rec['outputs']['candidate.parquet']
            np.testing.assert_array_equal(frame[ID],reference[ID])
            pred=frame.prediction_sec.to_numpy(float)
            pe=(pred-y)**2
            for label,control in [('V2',reference),('previous_composition',previous)]:
                ce=(control.prediction_sec.to_numpy(float)-y)**2
                by_day={str(day):np.flatnonzero(reference.day.to_numpy()==day) for day in reference.day.unique()}
                for row in rec['comparisons'][label]['day_removals']:
                    keep=np.ones(len(y),bool)
                    keep[by_day[row['removed_day']]]=False
                    delta=float(np.sqrt(pe[keep].mean())-np.sqrt(ce[keep].mean()))
                    assert abs(delta-row['delta_rmse'])<1e-10
                gains=ce-pe
                indices=np.argsort(gains)[-10:]
                np.testing.assert_array_equal(reference.iloc[indices][ID],rec['remove_top10_gain'][label]['ids'])
                keep=np.ones(len(y),bool)
                keep[indices]=False
                gain=float(np.sqrt(ce[keep].mean())-np.sqrt(pe[keep].mean()))
                assert abs(gain-rec['remove_top10_gain'][label]['gain_seconds'])<1e-10
            base_error=(reference.prediction_sec.to_numpy(float)-y)**2
            gains=np.where(missing,base_error-pe,-np.inf)
            indices=np.argsort(gains)[-2:]
            np.testing.assert_array_equal(reference.iloc[indices][ID],rec['missing_top2']['ids'])
            keep=np.ones(len(y),bool)
            keep[indices]=False
            assert abs(float(np.sqrt(pe[keep].mean()))-rec['remove_missing_top2']['composed']['rmse_sec'])<1e-10
            receipts[variant+'/'+fold]=dict(manifest_sha256=sha(folder/'manifest.json'),
                day_removals_vsV2_and_previous_exact=True,top10_ids_and_metrics_exact=True,top2missing_ids_and_metric_exact=True)
    destination.write_text(json.dumps(dict(status='passed',source_sha256=sha(__file__),folds=receipts),indent=2),encoding='utf-8')
    print('SENSITIVITY_VERIFIED_BOTH_VARIANTS_BOTH_FOLDS',flush=True)


if __name__=='__main__':
    main()
