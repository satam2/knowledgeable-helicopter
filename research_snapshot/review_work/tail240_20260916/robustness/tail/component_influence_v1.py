"""Remove component-supporting rows, retaining the fixed four ablation arms."""
import ablate_v1 as a
import numpy as np
import pandas as pd


def main():
    marker = a.common.read_json(a.OUT/'summary.json')
    assert marker['status'] == 'complete'
    out = a.common.external_path(a.OUT/'component_influence.json')
    assert not out.exists()
    result = {}
    for fold,w in a.WEIGHTS.items():
        path = a.OUT/f'{fold}_replay.parquet'
        assert a.common.sha256(path) == marker['replay_bindings'][fold]['sha256']
        f = pd.read_parquet(path)
        y = f[a.TARGET].to_numpy(float)
        cur = (f.current.to_numpy()-y)**2
        result[fold] = {}
        for name in a.ARMS:
            if name == 'current':
                continue
            other = (f[name].to_numpy()-y)**2
            benefit = other-cur
            idx = np.flatnonzero(f.missing.to_numpy())
            order = idx[np.argsort(-benefit[idx], kind='stable')]
            removal = {}
            for k in [1,2,5,10]:
                keep = np.ones(len(f), bool)
                keep[order[:k]] = False
                removal[str(k)] = dict(current_mse=float(cur[keep].mean()), ablation_mse=float(other[keep].mean()),
                    component_rmse_benefit=float(np.sqrt(other[keep].mean())-np.sqrt(cur[keep].mean())),
                    removed_component_sse_benefit=float(benefit[order[:k]].sum()),
                    net_component_sse_benefit=float(benefit.sum()),
                    removed_share_of_net_benefit=float(benefit[order[:k]].sum()/benefit.sum()),
                    removed_ids=f.iloc[order[:k]][a.ID].tolist())
            result[fold][name] = dict(component_benefit_rmse=float(np.sqrt(other.mean())-np.sqrt(cur.mean())),
                remove_component_supporting_rows=removal,
                day_removal_component_benefit_range=[-max(r['paired_rmse_gain'] for r in marker['folds'][fold][name]['day_removals']),
                    -min(r['paired_rmse_gain'] for r in marker['folds'][fold][name]['day_removals'])])
    seasonal = {}
    for name in a.ARMS:
        if name == 'current':
            continue
        seasonal[name] = {}
        for k in ['1','2','5','10']:
            cur = sum(a.WEIGHTS[f]*result[f][name]['remove_component_supporting_rows'][k]['current_mse'] for f in a.WEIGHTS)
            alt = sum(a.WEIGHTS[f]*result[f][name]['remove_component_supporting_rows'][k]['ablation_mse'] for f in a.WEIGHTS)
            seasonal[name][k] = dict(current_rmse=float(np.sqrt(cur)), ablation_rmse=float(np.sqrt(alt)),
                                     component_rmse_benefit=float(np.sqrt(alt)-np.sqrt(cur)))
    a.common.write_json(out, dict(status='complete', source_sha256=a.common.sha256(__file__),
        main_source_sha256=a.common.sha256(a.__file__), summary_sha256=a.common.sha256(a.OUT/'summary.json'),
        direction='Positive means current beats ablation. Remove the rows with largest squared-error benefit of retaining components.',
        folds=result, seasonal_remove_k_component_supporting_rows_per_fold=seasonal,
        limitation='Row deletion is diagnostic only; all official comparisons retain every original score row.'))
    print(seasonal, flush=True)


if __name__ == '__main__':
    main()
