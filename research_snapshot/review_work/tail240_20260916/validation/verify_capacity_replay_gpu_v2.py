"""Require exact fixed-bin state before the declared independent GPU replay."""
from pathlib import Path
import joblib
import torch
import verify_capacity_replay as frozen


def main():
    prep, _, _, _ = frozen.preparation()
    marker = frozen.read(frozen.MODEL / 'manifest.json')
    assert frozen.sha(frozen.MODEL / 'fit_model.joblib') == marker['outputs']['fit_model.joblib']
    saved = joblib.load(frozen.MODEL / 'fit_model.joblib')
    expected = frozen.ple.rtdl_num_embeddings.PiecewiseLinearEmbeddings(prep['bins'], d_embedding=16, activation=False, version='B')
    left = expected.impl.state_dict()
    right = saved['estimator'].numeric_embeddings.impl.state_dict()
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key
    assert torch.equal(saved['estimator'].embedded, prep['embedded'])
    assert torch.equal(saved['estimator'].passthrough, prep['passthrough'])
    frozen.write(frozen.OUT / 'candidate_bins_receipt.json', dict(
        status='passed', source_sha256=frozen.sha(Path(__file__)),
        model_sha256=marker['outputs']['fit_model.joblib'], fixed_bin_impl_exact=True,
        embedded_and_passthrough_exact=True, peak_bytes=frozen.guard()))
    del saved, expected, left, right
    frozen.gc.collect()
    frozen.replay_gpu()


if __name__ == '__main__':
    frozen.torch.set_num_threads(1)
    with frozen.threadpool_limits(1):
        main()
