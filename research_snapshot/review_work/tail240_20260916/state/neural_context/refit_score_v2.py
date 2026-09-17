"""Immutable v1 refit runner with canonical fold metadata serialization."""
import torch
import argparse
import json
import refit_score as original

common = original.common
OUT = common.external_path(original.ROOT / 'private_runs/tail240_20260916/state/neural_context/refit_score_v2')


def canonical_fold_data(loader, *args, **kwargs):
    indices, split, cohorts = loader(*args, **kwargs)
    normalized = json.loads(json.dumps(split, default=str))
    assert common.object_hash(normalized) == common.object_hash(split)
    return indices, normalized, cohorts


def declare():
    prior = original.declare()
    payload = dict(prior, wrapper_sha256=common.sha256(__file__),
        original_protocol_sha256=common.sha256(original.OUT / 'protocol.json'),
        metadata_only_change='Serialize datetime.date values in live split metadata to their canonical ISO strings, matching saved original manifest. Same indices/cohorts and canonical split hash; no feature/label/training/prediction changes.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == payload
    else:
        common.write_json(path, payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    else:
        original.OUT = OUT
        original.declare = lambda: protocol
        loader = common.fold_data
        common.fold_data = lambda *a, **kw: canonical_fold_data(loader, *a, **kw)
        if args.summary:
            original.summary()
        else:
            assert args.fold
            original.frozen.pa.set_cpu_count(2)
            original.frozen.pa.set_io_thread_count(1)
            with original.frozen.threadpool_limits(2):
                original.run(args.fold)
