"""Fixed, explicitly recorded schedule-head seeds; no best-seed selection."""

import argparse

import next230_schedule_mixture as mixture
from next230_common import OUT, load_data
from taxiout.artifacts import read_json, sha256, write_json


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, choices=[20260911, 20260912], required=True)
    p.add_argument('--folds', nargs='+', default=['F1', 'F3'])
    args = p.parse_args()
    file = OUT / f'mixture_seed_protocol_{args.seed}.json'
    config = {**mixture.CONFIG, 'seed': args.seed}
    protocol = {'seed': args.seed, 'config': config, 'runner_sha256': sha256(__file__),
                'training_source_sha256': sha256(mixture.__file__),
                'selection': 'Fixed seeds 20260910/11/12; evaluate every seed and a fixed equal average. Main residual remains frozen.'}
    if file.exists() and read_json(file) != protocol:
        raise ValueError('Seed protocol changed')
    write_json(file, protocol)
    mixture.CONFIG = config
    data = load_data()
    for fold in args.folds:
        mixture.train(fold, data)
