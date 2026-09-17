"""Bind both independent fold receipts and recompute the frozen seasonal gate."""
from audit_union387_sources import ROOT, read, sha, write
from pathlib import Path
import numpy as np

base=ROOT/'private_runs/tail240_20260916/models/preprocessed_locations_v1'
protocol=read(base/'protocol.json');assessment=read(base/'assessment.json')
checks={}
for fold,version in [('F1','v2'),('F3','v3')]:
    path=ROOT/f'private_runs/tail240_20260916/validation/preprocessed_locations_{fold}_{version}/receipt.json'
    receipt=read(path)
    assert receipt['status']=='passed'
    assert receipt['producer_manifest_sha256']==assessment['manifests'][fold]==sha(base/fold/'manifest.json')
    checks[fold]=dict(receipt_sha256=sha(path),data=receipt)
results={}
for endpoint in ('matched','primary'):
    before=float(np.sqrt(sum(w*checks[f]['data']['metrics'][endpoint]['control']['mse'] for f,w in zip(['F1','F3'],protocol['weights']))))
    after=float(np.sqrt(sum(w*checks[f]['data']['metrics'][endpoint]['candidate']['mse'] for f,w in zip(['F1','F3'],protocol['weights']))))
    result=dict(reference_rmse=before,rmse=after,gain=before-after,
        both_months_positive=all(checks[f]['data']['metrics'][endpoint]['delta_rmse']<0 for f in checks),
        all_day_removals_positive=all(checks[f]['data']['metrics'][endpoint]['all_day_removals_improve'] for f in checks))
    for name,value in result.items():
        if isinstance(value,bool):assert value==assessment['seasonal'][endpoint][name]
        else:np.testing.assert_allclose(value,assessment['seasonal'][endpoint][name],rtol=0,atol=1e-10)
    results[endpoint]=result
passed=results['primary']['gain']>=2 and results['primary']['both_months_positive'] and results['primary']['all_day_removals_positive']
assert passed==assessment['gate_passed']==False
out=ROOT/'private_runs/tail240_20260916/validation/preprocessed_locations_seasonal_v1'
out.mkdir(parents=True,exist_ok=False)
write(out/'receipt.json',dict(status='passed',source_sha256=sha(Path(__file__)),assessment_sha256=sha(base/'assessment.json'),
    fold_receipt_hashes={f:r['receipt_sha256'] for f,r in checks.items()},seasonal=results,gate_passed=passed,no_score=True))
print(results,flush=True)
