"""Bind complete tree-family contract/replay/vocabulary receipts across both folds."""
import argparse
import hashlib
import json
from pathlib import Path
import math
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'private_runs/breakthrough_20260916/missing/combined_family_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['catboost','xgb'],required=True)
    args=parser.parse_args()
    folder=OUT/args.family
    destination=folder/'summary.json'
    assert not destination.exists()
    records={}
    receipts={}
    for fold in ['F1','F3']:
        records[fold]=json.loads((folder/(fold+'.json')).read_text())
        for name in [fold+'.json',fold+'_vocabulary_v3.json']:
            path=folder/name
            rec=json.loads(path.read_text())
            assert rec['status']=='passed'
            receipts[name]=sha(path)
        assert records[fold]['model_replay']['refit']['max_abs_delta']<=1e-4
        assert records[fold]['same225_schema_IDs_receipts_as_leaf63_and_TabM']
    weights={'F1':192122/344841,'F3':152719/344841}
    score=math.sqrt(sum(weights[f]*records[f]['metrics']['rmse_sec']**2 for f in weights))
    payload=dict(status='passed',family=args.family,source_sha256=sha(__file__),receipts=receipts,
        seasonal_rmse=score,full_score_rows=sum(r['metrics']['n'] for r in records.values()),
        CPU_independent_replayed_rows=sum(r['model_replay']['refit']['rows'] for r in records.values()),
        max_CPU_replay_abs_delta=max(r['model_replay']['refit']['max_abs_delta'] for r in records.values()),
        exhaustive_fit_and_refit_category_vocabularies_exact=True,GPU_used=False,
        caveat='Matched225information/cohorts, not parameter-matched model capacity. Adaptive exposeddevelopment;128CPUscore rows per fold, not full independentmodelinference. Fullscoreproducerreplay separatelyhash-verified.')
    destination.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)


if __name__=='__main__':
    main()
