"""Seasonal following-group summary from independently replayed predictions."""
import math
import validate_candidate as v

ROOT=v.ROOT
OUT=ROOT/'private_runs/tail240_20260916/validation/following_seasonal_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    paths={f:ROOT/f'private_runs/tail240_20260916/validation/following_models_following_groups_tune_{version}_{f}/receipt.json' for f,version in [('F1','v1'),('F3','v2')]}
    receipts={f:v.read_json(p) for f,p in paths.items()};weights=v.read_json(v.BINDING)['weights']
    for receipt in receipts.values():
        assert receipt['status']=='passed'
        assert all(x['native_max_abs_delta_sec']==0 for x in receipt['models'].values())
    result={}
    for name in ['all_finite','ordinary_proxy','global9_fixed25','global9_component_replacement_diagnostic']:
        current=math.sqrt(sum(weights[f]*r['metrics'][name]['candidate']['mse'] for f,r in receipts.items()))
        control=math.sqrt(sum(weights[f]*r['metrics'][name]['control']['mse'] for f,r in receipts.items()))
        result[name]={'candidate_rmse':current,'control_rmse':control,'gain':control-current,
            'both_months_positive':all(r['metrics'][name]['gain']>0 for r in receipts.values()),
            'both_months_all_day_removals_improve':all(r['metrics'][name]['paired']['all_day_removals_improve'] for r in receipts.values()),
            'materiality_ge2':control-current>=2.}
    v.write_json(OUT/'receipt.json',{'status':'complete','source_sha256':v.sha256(__file__),
        'input_receipts':{f:v.sha256(p) for f,p in paths.items()},'weights':weights,'comparisons':result,
        'interpretation':'Both months native replay exact and positive point/day gains, but fixed25 materiality fails. Component replacement diagnostic is not alternate promotion. No score advancement justified.'})
    print(result,flush=True)


if __name__=='__main__':main()
