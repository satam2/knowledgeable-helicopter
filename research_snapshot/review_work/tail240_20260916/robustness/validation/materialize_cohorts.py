"""Materialize independently reconstructed chronology using no target columns."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import lightgbm
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import psutil
from contracts import ID,FLIGHT,TIME,BOUNDARIES,reconstruct_declared_cohorts,validate_refit

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import object_hash
from taxiout.paths import external_path
OUT=external_path(ROOT/'private_runs/tail240_20260916/robustness/validation/cohorts_v1')
COLS=[ID,FLIGHT,TIME,'ADEP_mvt','proxy_sec']
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def guard():
    mem=psutil.Process().memory_info()
    assert mem.peak_wset<4*1024**3 and psutil.virtual_memory().available>=8*1024**3
    return mem.peak_wset


def main():
    guard()
    assert not OUT.exists(),'Preserve existing version'
    path=ROOT/'private_runs/screening_230/data/interim/audit/departures.parquet'
    auditpath=ROOT/'private_runs/screening_230/reports/data_audit.json'
    source_hash=sha(path)
    assert source_hash==read(auditpath)['artifacts'][path.name]
    bindpath=ROOT/'private_runs/tail240_20260916/validation/baseline_binding.json'
    binding=read(bindpath)
    data=pd.read_parquet(path,columns=COLS)
    assert list(data.columns)==COLS and data[ID].is_unique
    metadata=data[[ID,FLIGHT,TIME]]
    assembled={}
    for fold in BOUNDARIES:
        ids,purges=reconstruct_declared_cohorts(metadata,fold)
        verification=validate_refit(metadata,ids)
        stages={stage:values for stage,values in ids.items() if stage!='evaluation'}
        evaluation=data.loc[data[ID].isin(ids['evaluation'])]
        for month in BOUNDARIES[fold]['evaluation']:
            panel=evaluation.loc[evaluation[TIME].dt.strftime('%Y-%m').eq(month),ID].tolist()
            assert panel
            stages['evaluation_'+month]=panel
        benchmark='evaluation_'+('2025-07' if fold=='F1' else '2025-11')
        bound=binding['folds'][fold]['cohorts']['all']['score']
        assert dict(n=len(stages[benchmark]),hash=object_hash(stages[benchmark]))==bound
        assembled[fold]=(stages,purges,verification)
        guard()
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='prepared_no_fits',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),contracts_sha256=sha(Path(__file__).with_name('contracts.py')),
        tests_sha256=sha(Path(__file__).with_name('test_contracts.py')),metadata_sha256=source_hash,
        metadata_path=str(path.relative_to(ROOT)),audit_receipt_sha256=sha(auditpath),
        baseline_binding_sha256=sha(bindpath),projection=COLS,target_columns_read=[],
        chronological_boundaries=BOUNDARIES,
        purge_policy='Each train/stop/calibration stage purged against all later raw named stages; evaluation panels never purged.',
        refit_policy='Source-ordered union of purged train and stop; intentional overlap only.',
        historically_exposed=True,fresh_holdout=False,reuse_previously_stopped_models=False,ranking_targets_accessed=False,
        folds={},peak_bytes=0)
    indexed=data.set_index(ID,drop=False)
    for fold,(stages,purges,verification) in assembled.items():
        folder=OUT/fold
        folder.mkdir()
        record=dict(purges=purges,independent_integrity_checks=verification,canonical_score_ids_exact=True,stages={})
        for stage,ids in stages.items():
            selected=indexed.loc[ids].reset_index(drop=True)[COLS]
            dest=folder/(stage+'.parquet')
            selected.to_parquet(dest,index=False)
            saved=pd.read_parquet(dest,columns=[ID])
            assert saved[ID].tolist()==ids
            record['stages'][stage]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),rows=len(ids),
                ordered_id_hash=object_hash(ids),min_time=str(selected[TIME].min()),max_time=str(selected[TIME].max()))
            guard()
        manifest['folds'][fold]=record
    manifest['peak_bytes']=guard()
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status=manifest['status'],manifest=str(OUT/'manifest.json'),manifest_sha256=sha(OUT/'manifest.json'),
        folds={f:{'purges':r['purges'],'stages':{s:v['rows'] for s,v in r['stages'].items()}} for f,r in manifest['folds'].items()},
        peak_bytes=manifest['peak_bytes']),indent=2),flush=True)


if __name__=='__main__':main()
