"""Close the bounded campaign using declared gates, not a new score-fit rule."""
import numpy as np
import pandas as pd
from common import OUT, HERE, WORKSPACE, read_json, write_json, sha256, utc_now


def main():
    summary=read_json(OUT/'summary.json')
    assert not summary['incomplete_or_failed'], 'Active or failed primary model records need review'
    candidates=[r for r in summary['rows'] if r.get('candidate')!='submitted_reference']
    primary=next(r for r in candidates if r['candidate']=='lightgbm_correction_arrival_full_SCREEN_s20260916' and r['variant']=='blend25')
    secondary=next(r for r in candidates if r['candidate']=='lightgbm_correction_arrival_full_SCREEN_s20260917' and r['variant']=='blend25')
    checks=sorted((OUT/'validation').glob('completed_runs_*.json'))
    latest=read_json(checks[-1])
    complete=list((OUT/'models').glob('*/manifest.json'))+list((OUT/'missing_mixture/models').glob('*/manifest.json'))
    verified={r['name'] for r in latest['verified']}
    assert all(read_json(p)['name'] in verified for p in complete),'Final independent metric audit incomplete'
    diff={}
    for fold in ['F1','F3']:
        a=pd.read_parquet(OUT/'models'/f'lightgbm_correction_arrival_full_{fold}_s20260916'/'blend25.parquet')
        b=pd.read_parquet(OUT/'models'/f'lightgbm_correction_arrival_full_{fold}_s20260917'/'blend25.parquet')
        diff[fold]=float(np.max(np.abs(a.prediction_sec-b.prediction_sec)))
    decision=dict(created_utc=utc_now(),status='checkpoint complete; retain submitted reference',
        recommendation='Do not promote or submit this round. Retain V2; keep full-data LightGBM plus arrivals as a complementary research lead.',
        reference_official_rmse_sec=294.626,reference_local_seasonal_rmse_sec=summary['reference_seasonal_rmse_sec'],
        primary_candidate=primary,component_seed_repeat=secondary,seed_prediction_max_abs_delta_sec=diff,
        seed_scope='LightGBM expert fit/tune/refit varied; submitted CatBoost/gate/Rome experts fixed. Not full-pipeline replication.',
        decision_reason='Approximate one-second gain is stable on exposed development cohorts but below the predeclared two-second material gain threshold.',
        promotion_gates={'minimum_gain_two_seconds':primary['gain_sec']>=2,'both_screen_months_improve':primary['both_improve'],
            'day_removal_stable':primary['day_removal_stable'],'missing_rmse_regression_below_five_percent':primary['max_missing_rmse_ratio']<=1.05},
        broader_evaluation='F2/G1 not run because material-gain gate failed; all available score months including December already exposed.',
        final_year_fit='Not run: no promoted candidate. No ranking prediction candidate or competition upload created.',
        completed_fit_records=summary['completed_fit_records'],paired_candidate_variant_records=len(candidates),
        model_families=['CatBoost','LightGBM','XGBoost','Ridge','TabM'],
        primary_uncertainty=latest['seasonal_paired_uncertainty'][primary['candidate']]['blend25'],
        limitations=['Bounded family hyperparameters, not convergence/equal compute guarantees.',
            'Only LightGBM ordinary corrections and missing mixture advanced to all permitted fit/refit rows.',
            'Adaptive multiple testing over exposed score folds; paired intervals measure stability, not fresh confirmation.',
            'Physical pilot used empirical q10 priors; no real taxiway route network.',
            'Arrival inventory bounded at 120 minutes, not full aircraft surface occupancy; final NM times do not prove real-time publication availability.',
            'Missing-expert failure does not reject better source-reliability formulations or calibration.'],
        next_investments=[
            'Prioritize missing-clock and source-disagreement mechanisms: they account for 47.14 percent of reference error in 2.19 percent of rows. Diagnose source conventions and conditional reliability with chronological calibration before another broad capacity search.',
            'Retain full-data LightGBM arrival expert as a fixed complementary lead. A later bounded study can isolate arrival durations from inventory and encoding effects; do not repeatedly tune blend weights on these score months.',
            'Build full-period weather coverage/issue-time audit before a local chronological join. Geometry requires historical stand mapping and connected taxiway routes.',
            'TabM merits convergence/encoding work only as a separately budgeted study; FT-Transformer and gated TabPFN-3.5 remain untested locally.'],
        provenance={'summary_sha256':sha256(OUT/'summary.json'),'independent_receipt':str(checks[-1]),
                    'independent_receipt_sha256':sha256(checks[-1]),'decision_script_sha256':sha256(__file__)})
    assert not any(r.get('meets_screen_gates') for r in candidates), 'New qualifying candidate requires broader evaluation before closure'
    write_json(OUT/'decision.json',decision)
    rows=[]
    for p in complete:
        r=read_json(p)
        rows.append(dict(name=r['name'],status=r['status'],path=str(p),sha256=sha256(p),fold=r.get('fold') or r['name'].split('_')[-2],
                         family=r['family'],target=r['target'],features=r['features'],full=r['full'],seed=r['seed']))
    write_json(OUT/'experiment_registry.json',dict(created_utc=utc_now(),experiments=rows,
        correlated_variants='Candidate and fixed25 percent blend; direct also missing-only and missing-only25 percent. Not independent discoveries.',
        failures_retained=['XGBoost F1 metadata serialization recovered with exact model replay; original timing unknown.',
            'Aviation pack-month boundary assumption caught in feature preparation; failed partial cache retained and rebuilt.',
            'Windows library import-order smoke failures; working wrappers documented.',
            'Missing mixture post-run source amendment preserved with original hash-matched executed source and protocol.']))
    output=WORKSPACE/'output/campaign_20260916'
    output.mkdir(parents=True,exist_ok=True)
    text=f'''# PRC 2026 diverse research checkpoint

Retain V2. The new full-data LightGBM correction expert with completed-arrival features,
combined with 75% of the submitted reference, scores **{primary['seasonal_rmse_sec']:.6f} seconds**
on exposed July/November development cohorts. Reference: **{summary['reference_seasonal_rmse_sec']:.6f}**.
The gain of **{primary['gain_sec']:.3f} seconds** repeats at **{secondary['gain_sec']:.3f} seconds**
with a second LightGBM seed, but misses the predeclared two-second promotion threshold.
Both months improve and day-removal checks retain the gain; missing-clock predictions remain unchanged.
These are local measurements, not new official scores. The official V2 score remains 294.626 seconds.

The campaign completed {summary['completed_fit_records']} model/fold records and {len(candidates)} correlated
seasonal candidate/variant comparisons, using three actual subagents with exclusive ownership.
CatBoost, LightGBM, XGBoost, Ridge and actual TabM were evaluated. Direct models, a clock-free
stand/runway q10 plus delay expert, and an all-airport soft missing-clock mixture did not qualify.
Only LightGBM correction and missing-clock models used full eligible fit/refit cohorts; other
family screens used common 200k training samples and unchanged complete score cohorts.

## Evidence

- Visual report: `PRC_2026_Diverse_Campaign.pdf` in this directory.
- Consolidated table: `../../private_runs/campaign_20260916/experiment_table.csv`.
- Decision and updated error budget: `../../private_runs/campaign_20260916/decision.json` and `summary.json`.
- Source-backed model shortlist: `../research_20260916/research.md` with dated public-source receipts.
- Reproduction and dependency versions: `../../private_runs/campaign_20260916/REPRODUCE.md`.
- Independent review: `../../private_runs/campaign_20260916/validation/final_validation.json`.
- Final integrity inventory: `../../private_runs/campaign_20260916/final_verification.json`.

No F2/G1 expansion, final all-year fit, ranking prediction candidate or submission was performed.
All score folds were already exposed. Weather/geometry pilots establish small public-source coverage,
not predictive gain. FT-Transformer and TabPFN-3.5 remain researched but untested locally.

Next investment should target missing clocks and source disagreement, which retain 47.14% of
reference squared error in 2.19% of rows, while preserving the small complementary arrival expert.
All confidential inputs and artifacts remain outside both Git checkouts. Existing dirty work is preserved.
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    print(decision['recommendation'],flush=True)


if __name__=='__main__':
    main()
