"""Prove full-year steps and adapter hashes inherit the original fold models."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import lightgbm
from preflight import ROOT, OUT, read, sha
import json
import numpy as np


def main():
    destination=OUT/'ordinary_protocols_receipt.json';assert not destination.exists()
    base=ROOT/'private_runs/breakthrough_20260916'
    release=ROOT/'private_runs/tail240_20260916/final_ordinary/v2'
    roots=dict(lgb225=('deeper_lgb/combined','lightgbm_leaf63'),lgb449=('deeper_sequence8','lightgbm_leaf63_sequence8'),
               lgb387=('deeper_context_union','lightgbm_leaf63_sequence'),tabm225=('full_neural','tabm_combined_standard'),cat225=('combined_catboost','catboost_combined'))
    expected=dict(lgb225=2117,lgb449=2256,lgb387=2250,tabm225=22,cat225=4594)
    records={}
    for name,(directory,family) in roots.items():
        proto=read(release/name/'protocol.json')
        for relative,digest in proto['sources'].items():assert sha(ROOT/relative)==digest
        adapter=('review_work/breakthrough_20260916/deeper_lgb/adapter.py' if name.startswith('lgb') else
                 'review_work/breakthrough_20260916/models/tabm_gpu.py' if name=='tabm225' else
                 'review_work/breakthrough_20260916/models_retrieval/combined_catboost/adapter.py')
        encoder='review_work/campaign_20260916/lgbm_adapter.py' if name.startswith('lgb') else 'review_work/breakthrough_20260916/models/encoders.py'
        checks=[];steps=[]
        for fold in ('F1','F3'):
            path=base/directory/f'{family}_aobt_allfinite_{fold}_s20260916'/'manifest.json'
            marker=read(path);assert sha(path)==proto['controls'][fold] and marker['status']=='complete'
            for source in (adapter,encoder):
                digest=sha(ROOT/source)
                keys=[key for key,value in marker['source_hashes'].items() if value==digest]
                assert keys,(name,fold,source,'current hash absent from original manifest')
                checks.append(dict(fold=fold,path=source,sha256=digest,original_manifest_keys=keys))
            steps.append(marker['fit']['steps'])
            assert proto['columns']==marker['feature_columns']
        assert steps==proto['fold_steps'] and int(np.median(steps))==proto['final_steps']==expected[name]
        records[name]=dict(protocol_sha256=sha(release/name/'protocol.json'),steps=steps,final_steps=expected[name],original_hash_containment=checks)
    result=dict(status='passed',source_sha256=sha(__file__),components=records,gpu_used=False,fitting_used=False)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
