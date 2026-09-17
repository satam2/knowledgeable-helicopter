"""Posthoc tune-only error attribution and exact saved global9 composition check."""
import lightgbm
import argparse
import numpy as np
import pandas as pd
import run_sign_mixture as mixture

common, ROOT = mixture.common, mixture.ROOT
OUT = common.external_path(ROOT/'private_runs/tail240_20260916/state/risk/sign_mixture_assessment_v1')


def main(fold):
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT/(fold+'.json')
    assert not output.exists()
    folder = mixture.OUT/fold
    manifest = common.read_json(folder/'manifest.json')
    path = folder/'tune_predictions.parquet'
    assert common.sha256(path) == manifest['outputs'][path.name]
    data = pd.read_parquet(path)
    classes = mixture.classes(data[mixture.TARGET]-data.proxy_sec)
    class_report = {}
    names = ['mixture', 'matched_single', 'union', 'blend25_mixture_union']
    for k, name in enumerate(mixture.NAMES):
        mask = classes == k
        class_report[name] = dict(rows=int(mask.sum()), prevalence=float(mask.mean()),
            mean_probability=float(data['probability_'+name].mean()),
            rmse={model:float(np.sqrt(np.mean((data.loc[mask, mixture.TARGET]-data.loc[mask, model])**2))) for model in names})
    base = ROOT/'private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1'
    preparation = common.read_json(base/'preparation.json')['folds'][fold]
    aligned_path = base/(fold+'_aligned_tune.parquet')
    assert common.sha256(aligned_path) == preparation['aligned_tune_sha256']
    weights_path = base/(fold+'_weights.json')
    weights = common.read_json(weights_path)
    assert weights == preparation['weights']
    aligned = pd.read_parquet(aligned_path)
    ordinary = data.loc[data.proxy_sec.between(0, 7200)].set_index(mixture.ID)
    assert ordinary.index.is_unique and aligned[mixture.ID].is_unique
    assert set(ordinary.index) == set(aligned[mixture.ID])
    ordinary = ordinary.loc[aligned[mixture.ID]]
    assert np.array_equal(ordinary[mixture.TARGET], aligned[mixture.TARGET])
    assert np.array_equal(ordinary.proxy_sec, aligned.proxy_sec)
    global9 = aligned[weights['experts']].to_numpy(float) @ np.asarray(weights['global'])
    truth = aligned[mixture.TARGET].to_numpy(float)
    reference_rmse = float(np.sqrt(np.mean((truth-global9)**2)))
    assert abs(reference_rmse-preparation['global_tune_rmse']) < 1e-9
    codes, days = pd.factorize(aligned[common.MOVEMENT].dt.floor('D'), sort=True)
    bootstrap = np.random.default_rng(20260916).multinomial(len(days), np.full(len(days), 1/len(days)), size=300)
    comparisons = {}
    for name in ['mixture', 'matched_single']:
        pred = .25*ordinary[name].to_numpy()+.75*global9
        gain = (truth-global9)**2-(truth-pred)**2
        rmse = float(np.sqrt(np.mean((truth-pred)**2)))
        comparisons[name] = dict(rmse=rmse, rmse_improvement=reference_rmse-rmse,
            mse_improvement=float(gain.mean()),
            mse_improvement_ci95=mixture.risk.bootstrap_interval(gain, np.ones(len(gain)), codes, bootstrap))
    common.write_json(output, dict(status='complete', fold=fold, source_sha256=common.sha256(__file__),
        mixture_manifest_sha256=common.sha256(folder/'manifest.json'),
        aligned_tune_sha256=common.sha256(aligned_path), weights_sha256=common.sha256(weights_path),
        class_diagnostics=class_report, global9_ordinary_tune=dict(rows=len(aligned),
            reference_rmse=reference_rmse, fixed25_blends=comparisons),
        limitation='Posthoc exposed-tune diagnostic. Global9 weights were fitted on these same June/October labels; this is in-sample ensemble calibration evaluation, not a fresh complementarity holdout. No class used to choose inference predictions. No score/ranking accessed.'))
    print(fold, class_report, comparisons, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', choices=['F1', 'F3'], required=True)
    main(parser.parse_args().fold)
