"""Comparison-specific seasonal assessment from independently verified tune receipts."""
from pathlib import Path
import math
import validate_candidate as v

ROOT=v.ROOT
OUT=ROOT/'private_runs/tail240_20260916/validation/neural_context_seasonal_v1'
PRODUCER=ROOT/'private_runs/tail240_20260916/state/neural_context'
RECEIPTS={fold:ROOT/f'private_runs/tail240_20260916/validation/neural_context_{fold}_{version}_verifier_v4/receipt.json'
    for fold,version in [('F1','v1'),('F3','v3')]}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    binding=v.read_json(v.BINDING);weights=binding['weights']
    source=v.read_json(PRODUCER/'v3/protocol.json')
    source_paths={'wrapper_sha256':'review_work/tail240_20260916/state/neural_context/run_f3_v3.py',
        'grouped_bins_source_sha256':'review_work/tail240_20260916/state/neural_context/run_f3_v2.py',
        'schema_helper_sha256':'review_work/tail240_20260916/models/schema_discovery.py',
        'v2_protocol_sha256':'private_runs/tail240_20260916/state/neural_context/v2/protocol.json',
        'source_sha256':'review_work/tail240_20260916/state/neural_context/run.py',
        'adapter_sha256':'review_work/breakthrough_20260916/models/tabm_ple_gpu.py',
        'trainer_sha256':'review_work/breakthrough_20260916/models/tabm_gpu.py',
        'encoder_sha256':'review_work/breakthrough_20260916/models/encoders.py',
        'loader_sha256':'review_work/tail240_20260916/state/risk/run_risk.py'}
    for key,path in source_paths.items():assert v.sha256(ROOT/path)==source[key]
    records={fold:v.read_json(path) for fold,path in RECEIPTS.items()}
    for fold,record in records.items():
        assert record['status']=='passed'
        assert record['source_sha256']==v.sha256(Path(__file__).with_name('verify_neural_context_v4.py'))
        version='v1' if fold=='F1' else 'v3'
        assert record['producer_manifest_sha256']==v.sha256(PRODUCER/version/fold/'manifest.json')
        assert record['candidate_native_max_abs_delta_sec']==record['control_native_max_abs_delta_sec']==0
    protocol={'source_sha256':v.sha256(__file__),'weights':weights,'baseline_binding_sha256':v.sha256(v.BINDING),
        'receipt_hashes':{fold:v.sha256(path) for fold,path in RECEIPTS.items()},
        'frozen_advancement_text':source['advancement'],'frozen_evaluation_text':source['evaluation'],
        'operation':'Comparison-specific sqrt(weighted MSE) seasonal point assessment only; no new weights, fits, predictions or score access.'}
    v.write_json(OUT/'protocol.json',protocol)
    reports={}
    for comparison in ['all_finite','ordinary_proxy','global9_replacement','global9_blend25']:
        candidate=math.sqrt(sum(weights[fold]*record['metrics'][comparison]['candidate']['mse'] for fold,record in records.items()))
        control=math.sqrt(sum(weights[fold]*record['metrics'][comparison]['control']['mse'] for fold,record in records.items()))
        month_gains={fold:record['metrics'][comparison]['gain'] for fold,record in records.items()}
        stable={fold:record['metrics'][comparison]['paired']['all_day_removals_improve'] for fold,record in records.items()}
        reports[comparison]={'candidate_seasonal_rmse':candidate,'control_seasonal_rmse':control,'gain':control-candidate,
            'month_gains':month_gains,'all_day_removals_improve_by_month':stable,'materiality_ge2':control-candidate>=2.,
            'comparison_specific_gate':all(x>0 for x in month_gains.values()) and all(stable.values()) and control-candidate>=2.}
    result={'status':'complete','source_sha256':v.sha256(__file__),'protocol_sha256':v.sha256(OUT/'protocol.json'),
        'source_chain_verified':source_paths,'comparisons':reports,
        'interpretation':'The frozen text says materiality against comparison and lists matched PLE225 plus global9 replacement/blend. It does not explicitly designate ensemble as sole comparator. Report separate gates, never an unqualified global pass. Matched-information advancement cannot be described as passing ensemble materiality. No automatic score advancement; any refit/score needs a separate declaration naming its comparison and preserving these below-threshold ensemble results.'}
    v.write_json(OUT/'receipt.json',result)
    print(reports,flush=True)


if __name__=='__main__':main()
