"""Readback check using independent vectorized interval and purge reconstruction."""
from materialize_cohorts import ROOT,OUT,COLS,ID,FLIGHT,TIME,sha,read,guard
import json
import pandas as pd
import numpy as np


def main():
    marker=read(OUT/'manifest.json')
    assert marker['status']=='prepared_no_fits' and marker['target_columns_read']==[]
    source=ROOT/'review_work/tail240_20260916/robustness/validation'
    for name,field in [('materialize_cohorts.py','source_sha256'),('contracts.py','contracts_sha256'),('test_contracts.py','tests_sha256')]:
        assert sha(source/name)==marker[field]
    path=ROOT/marker['metadata_path'];assert sha(path)==marker['metadata_sha256']
    raw=pd.read_parquet(path,columns=COLS)
    dates=raw[TIME];months=dates.dt.strftime('%Y-%m')
    findings={}
    for fold,starts,cal,evaluation in [('F1',['2025-01-01','2025-04-01','2025-05-01'],'2025-06-01',['2025-06','2025-07','2025-12']),
                                     ('F3',['2025-01-01','2025-08-01','2025-09-01'],'2025-10-01',['2025-10','2025-11','2025-12'])]:
        bounds=[pd.Timestamp(v,tz='UTC') for v in starts+[cal]]
        stages=['train','stop','calibration']
        masks={name:dates.ge(bounds[i])&dates.lt(bounds[i+1]) for i,name in enumerate(stages)}
        evaluation_mask=months.isin(evaluation)
        original={k:v.copy() for k,v in masks.items()}
        for i,name in enumerate(stages):
            later=evaluation_mask.copy()
            for other in stages[i+1:]:later|=original[other]
            forbidden=raw.loc[later,FLIGHT].dropna().unique()
            removed=masks[name]&raw[FLIGHT].notna()&raw[FLIGHT].isin(forbidden)
            assert int(removed.sum())==marker['folds'][fold]['purges'][name]
            masks[name]&=~removed
        masks['refit']=masks['train']|masks['stop']
        for month in evaluation:masks['evaluation_'+month]=months.eq(month)
        verified={}
        for stage,mask in masks.items():
            rec=marker['folds'][fold]['stages'][stage];saved=ROOT/rec['path']
            assert sha(saved)==rec['sha256']
            actual=pd.read_parquet(saved)
            expected=raw.loc[mask].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,expected,check_exact=True)
            assert len(actual)==rec['rows']
            verified[stage]=len(actual)
            guard()
        # Canonical score file equality is independent of JSON ID hashes.
        benchmark='evaluation_'+('2025-07' if fold=='F1' else '2025-11')
        canonical=ROOT/'private_runs/tail240_20260916/forensics/chronological_cohorts/v2'/f'{fold}_protected_score_ids.parquet'
        canonical_manifest=read(canonical.parent/'manifest.json')
        assert sha(canonical)==canonical_manifest['outputs'][canonical.name]
        current=pd.read_parquet(ROOT/marker['folds'][fold]['stages'][benchmark]['path'],columns=[ID])
        np.testing.assert_array_equal(current[ID],pd.read_parquet(canonical)[ID])
        findings[fold]=dict(stages=verified,canonical_score_ids_exact=True,every_metadata_cell_exact=True)
    result=dict(status='passed',source_sha256=sha(__file__),cohort_manifest_sha256=sha(OUT/'manifest.json'),folds=findings,
        projection=COLS,raw_target_not_read=True,all_14_stage_artifacts_exact=True,peak_bytes=guard(),fit_used=False)
    dest=OUT.parent/'cohorts_v1_readback.json';assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
