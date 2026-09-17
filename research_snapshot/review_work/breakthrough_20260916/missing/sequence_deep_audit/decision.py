"""Correct requested gate: >=2seasonal seconds and both months positive/stable."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'private_runs/breakthrough_20260916/missing/sequence_deep_audit'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    destination=OUT/'advancement_decision.json'
    assert not destination.exists()
    summary=json.loads((OUT/'summary.json').read_text())
    folds={f:json.loads((OUT/(f+'.json')).read_text()) for f in ['F1','F3']}
    gain=summary['seasonal_rmse']['old']-summary['seasonal_rmse']['new']
    positive=all(r['gain_seconds']>0 for r in folds.values())
    stable=all(r['daily_and_top10_gain_robust'] for r in folds.values())
    result=dict(source_sha256=sha(__file__),audit_summary_sha256=sha(OUT/'summary.json'),
        fold_receipts={f:sha(OUT/(f+'.json')) for f in folds},requested_gate='Atleast2seasonalRMSEseconds,bothmonthsimprove,everydayremovalandtop10gainremovalretainimprovement.',
        seasonal_gain_seconds=gain,both_months_improve=positive,day_and_top10_robust=stable,
        meets_requested_gate=gain>=2 and positive and stable,
        supersedes='Originalaudit summary meets_fixed_proposal_gate/perfoldtwo_secondflag usedoverstrict>=2eachmonth draftinginstruction; retainhistoricalaudit unchanged anduse this correctedrequestedgate.',
        no_new_composition_or_inference=True)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
