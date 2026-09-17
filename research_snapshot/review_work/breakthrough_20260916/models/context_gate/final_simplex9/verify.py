"""Independent active-set quadratic oracle and saved nine-expert predictions."""
import argparse
import itertools
import numpy as np
import pandas as pd
import run


def active_set_oracle(y, p, center=None, penalty=0.):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    relative = p - p[:, :1]
    target = y - p[:, 0]
    center = np.zeros(p.shape[1]) if center is None else np.asarray(center)
    scale = max(float(np.mean(target**2)), 1.) * len(y)
    gram = (relative.T @ relative + penalty * np.eye(p.shape[1])) / scale
    rhs = (relative.T @ target + penalty * center) / scale
    best = None
    for n in range(1, p.shape[1] + 1):
        for active in itertools.combinations(range(p.shape[1]), n):
            active = np.array(active)
            a = gram[np.ix_(active, active)]
            kkt = np.block([[a, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
            try:
                answer = np.linalg.solve(kkt, np.r_[rhs[active], 1.])[:n]
            except np.linalg.LinAlgError:
                answer = np.linalg.lstsq(kkt, np.r_[rhs[active], 1.], rcond=None)[0][:n]
            if np.any(answer < -1e-9) or not np.isclose(answer.sum(), 1., atol=1e-8):
                continue
            weights = np.zeros(p.shape[1])
            weights[active] = np.maximum(answer, 0)
            weights /= weights.sum()
            value = float(weights @ gram @ weights - 2 * rhs @ weights + (target @ target + penalty * (center @ center)) / scale)
            if best is None or value < best[0]:
                best = value, weights
    assert best is not None
    return best


def objective(y, p, weights, center=None, penalty=0.):
    center = np.zeros(p.shape[1]) if center is None else np.asarray(center)
    scale = max(float(np.mean(np.square(y - p[:, 0]))), 1.) * len(y)
    return (float(np.square(p @ weights - y).sum()) + penalty * np.square(weights - center).sum()) / scale


def synthetic():
    rng = np.random.default_rng(20260916)
    p = rng.normal(1000, 400, (4096, 9))
    y = p @ (np.arange(1., 10.) / 45.) + rng.normal(0, 10, len(p))
    for penalty in (0., 100000.):
        center = np.full(9, 1. / 9.)
        fitted = run.gate.constant_weights(y, p, center if penalty else None, penalty)
        best, weights = active_set_oracle(y, p, center, penalty)
        gap = objective(y, p, fitted, center, penalty) - best
        assert -1e-10 <= gap < 1e-9
        np.testing.assert_allclose(fitted, weights, atol=1e-5, rtol=0)
    print('SYNTHETIC_ACTIVE_SET_PASSED', flush=True)


def main():
    synthetic()
    preparation = run.read_json(run.OUT / 'preparation.json')
    score = run.read_json(run.OUT / 'score_summary.json')
    checks = {}
    for fold in ('F1', 'F3'):
        rec = preparation['folds'][fold]
        assert run.sha256(run.OUT / f'{fold}_aligned_tune.parquet') == rec['aligned_tune_sha256']
        assert run.sha256(run.OUT / f'{fold}_weights.json') == rec['weights_sha256']
        saved = run.read_json(run.OUT / f'{fold}_weights.json')
        data = pd.read_parquet(run.OUT / f'{fold}_aligned_tune.parquet')
        y, p = data[run.TARGET].to_numpy(), data[run.EXPERTS].to_numpy()
        best, oracle = active_set_oracle(y, p)
        global_w = np.array(saved['global'])
        gaps = {'global': objective(y, p, global_w) - best}
        coefficient_delta = {'global': float(np.max(np.abs(oracle - global_w)))}
        for airport, weights in saved['airports'].items():
            selected = data.ADEP_mvt.astype(str).eq(airport).to_numpy()
            best, oracle = active_set_oracle(y[selected], p[selected], global_w, saved['penalty'])
            gaps[airport] = objective(y[selected], p[selected], np.asarray(weights), global_w, saved['penalty']) - best
            coefficient_delta[airport] = float(np.max(np.abs(oracle - weights)))
        assert min(gaps.values()) > -1e-9 and max(gaps.values()) < 1e-8
        reference, _ = run.run_tune.common.reference(fold)
        protected = ~np.isfinite(reference.proxy_sec) | ~reference.proxy_sec.between(0, 7200)
        pscore = []
        for name in run.EXPERTS:
            receipt = preparation['sources'][fold][name]
            path = run.Path(receipt['directory']) / 'candidate.parquet'
            assert run.sha256(path) == receipt['outputs'][path.name]
            expert = pd.read_parquet(path)
            np.testing.assert_array_equal(expert[run.ID], reference[run.ID])
            pscore.append(expert.prediction_sec.to_numpy())
        pscore = np.column_stack(pscore)
        for variant in ('global9', 'airport9'):
            path = run.OUT / f'{fold}_score' / f'{variant}.parquet'
            assert run.sha256(path) == score['folds'][fold]['outputs'][path.name]
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame[run.ID], reference[run.ID])
            np.testing.assert_array_equal(frame[run.TARGET], reference[run.TARGET])
            np.testing.assert_array_equal(frame.prediction_sec[protected], reference.prediction_sec[protected])
            coefficients = np.tile(saved['global'], (len(frame), 1)) if variant == 'global9' else np.array([saved['airports'].get(a, saved['global']) for a in reference.ADEP_mvt.astype(str)])
            replay = reference.prediction_sec.to_numpy().copy()
            replay[~protected] = np.sum(pscore[~protected] * coefficients[~protected], axis=1)
            np.testing.assert_array_equal(replay, frame.prediction_sec.to_numpy())
            sse = float(np.square(frame.prediction_sec - frame[run.TARGET]).sum())
            report = score['folds'][fold]['variants'][variant]['metrics']['overall']
            np.testing.assert_allclose(sse, report['sse'], rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(sse / len(frame)), report['rmse_sec'], rtol=1e-12)
        checks[fold] = {'normalized_objective_gap': gaps, 'max_coefficient_delta': coefficient_delta,
                        'fullscore_coefficient_replay_delta': 0., 'protected_v2_exact_rows': int(protected.sum())}
        print('INDEPENDENT_SIMPLEX9_VERIFIED', fold, max(gaps.values()), max(coefficient_delta.values()), flush=True)
    for variant in ('global9', 'airport9'):
        recomputed = run.season_score(score['folds']['F1']['variants'][variant]['metrics']['overall'], score['folds']['F3']['variants'][variant]['metrics']['overall'])
        assert recomputed == score['seasonal_rmse'][variant]
    run.write_json(run.OUT / 'independent_verification.json', {'status': 'passed', 'source_sha256': run.sha256(__file__),
        'score_summary_sha256': run.sha256(run.OUT / 'score_summary.json'), 'folds': checks,
        'oracle': 'All511nonemptyactiveweightsets, independentKKTlinearsolve; coefficients unchanged.', 'gpu_used': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    with run.threadpool_limits(2):
        synthetic() if args.synthetic_only else main()
