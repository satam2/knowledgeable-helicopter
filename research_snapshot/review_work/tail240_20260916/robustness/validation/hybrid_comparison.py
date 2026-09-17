"""Bound post-score hybrid decomposition; no causal attribution to stopping."""
from materialize_cohorts import ROOT,sha,read,guard
import json
import numpy as np
import pandas as pd


def main():
    root=ROOT/'private_runs/tail240_20260916/robustness'
    evaluation=root/'chronological_v1/ensemble/F1/evaluation'
    verified=read(root/'validation/ensemble/F1_evaluation.json')
    assert verified['status']=='passed' and verified['evaluation_manifest_sha256']==sha(evaluation/'manifest.json')
    marker=read(evaluation/'manifest.json');path=evaluation/'historical_hybrid.parquet'
    assert sha(path)==marker['outputs'][path.name]
    frame=pd.read_parquet(path);mask=frame.proxy_sec.between(0,7200).to_numpy()
    y=frame.TAXITIME_SEC_mvt.to_numpy();candidate=frame.candidate.to_numpy();v3=frame.v3.to_numpy()
    np.testing.assert_array_equal(candidate[~mask],v3[~mask])
    slices={}
    for name,keep in [('ordinary',mask),('nonordinary',~mask),('all',np.ones(len(frame),bool))]:
        old=np.square(v3[keep]-y[keep]);new=np.square(candidate[keep]-y[keep])
        slices[name]=dict(rows=int(keep.sum()),v3_rmse=float(np.sqrt(old.mean())),candidate_rmse=float(np.sqrt(new.mean())),
            candidate_minus_v3_rmse=float(np.sqrt(new.mean())-np.sqrt(old.mean())),
            candidate_minus_v3_sse=float(new.sum()-old.sum()))
    np.testing.assert_allclose(slices['ordinary']['candidate_minus_v3_sse'],slices['all']['candidate_minus_v3_sse'],rtol=1e-12)
    out=root/'validation/F1_hybrid_comparison.json';assert not out.exists()
    result=dict(status='passed',source_sha256=sha(__file__),independent_evaluation_sha256=sha(root/'validation/ensemble/F1_evaluation.json'),
        hybrid_sha256=sha(path),fold='F1',month='2025-07',slices=slices,
        mechanical_location='Every changed prediction is ordinary; frozen nonordinary predictions are bit-identical. This localizes changed predictions, not the reason for model degradation.',
        confounders=[
            'New bases refit January-April; historical local V3 score experts generally refit January-June, two additional months.',
            'New three-family set is LGB387,PLE387,CB387; historical V3 comes from a nine-expert mixture with different feature sets and fitted weights.',
            'New stopping month is April and calibration is May; historical local recipe uses June for selection/calibration.',
            'New objective for comparison is predeclared half-simplex/half-equal versus equal; historical V3 uses its previously chosen blend.',
            'The hybrid retains historical missing and finite-nonordinary routes, so it is not a strict new complete-pipeline estimate.'],
        conclusion='This is not a controlled stopping-leakage ablation and cannot estimate how much historical optimism came from stopping reuse. The cleaner chronology changes history, expert set and calibration together; a worse result does not establish that leakage correction caused the entire difference.',
        no_new_fits=True,no_new_label_reads=True,peak_bytes=guard())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
