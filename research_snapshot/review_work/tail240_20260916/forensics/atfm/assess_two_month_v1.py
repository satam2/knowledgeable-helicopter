"""Fixed seasonal weighting of completed ATFM tune assessments only."""
import json
import hashlib
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT/'private_runs/tail240_20260916/forensics/atfm'
OUT = BASE/'two_month_assessment_v1'
PATHS = {'F1':BASE/'tune_f1_v3/F1/assessment.json', 'F3':BASE/'tune_f3_v2/F3/assessment.json'}
WEIGHTS = {'F1':192122/344841, 'F3':152719/344841}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    results = {fold:json.loads(path.read_text()) for fold,path in PATHS.items()}
    assert all(r['status']=='complete' and r['no_score_prediction'] for r in results.values())
    combined = {}
    for key in ['matched_all_finite','matched_ordinary','ordinary_fixed25','ordinary_union_replacement']:
        reference = math.sqrt(sum(WEIGHTS[f]*r[key]['reference_rmse']**2 for f,r in results.items()))
        candidate = math.sqrt(sum(WEIGHTS[f]*r[key]['rmse']**2 for f,r in results.items()))
        combined[key] = dict(reference_rmse=reference, candidate_rmse=candidate, improvement=reference-candidate,
            both_months_improve=all(r[key]['rmse_improvement']>0 for r in results.values()),
            both_month_day_removals_improve=all(r[key]['all_day_removals_improve'] for r in results.values()))
    receipt = dict(status='complete',source_sha256=sha(__file__),input_sha256={f:sha(p) for f,p in PATHS.items()},
        weights=WEIGHTS,results=combined,gate_passes=all(r['fold_gate'] for r in results.values()),
        scope='Fixed ranking-season weights applied to ordinary/allfinite tune-cohort MSEs. Not full-departure score RMSE; missingNM not included; no score/refit/ranking predictions.',
        limitations='Exposed tune months and global9weights fit on same tune. Union replacement diagnostic is not an alternative promotion gate.')
    OUT.mkdir()
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
