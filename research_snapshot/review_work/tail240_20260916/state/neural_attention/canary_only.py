"""Separately scheduled full-preparation canary; cannot start optimizer training."""
import torch
import lightgbm
import argparse
import gc
import joblib
import run_attention as subject

common, adapter = subject.common, subject.adapter
OUT = common.external_path(subject.OUT.parent / 'canary_v1')


class CanaryComplete(Exception):
    pass


def declare():
    original = subject.declare()
    value = dict(source_sha256=common.sha256(__file__),
        producer_protocol_sha256=common.sha256(subject.OUT / 'protocol.json'),
        scope='Saved387 entire tune native parity, allfit-only encoder/token statistics/PLE bins, 4096 backward8192 inference noopt canary only. Deliberate stop before original trainer.',
        resources=original['resources'], sources=original['source_hashes'])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return original


def run(fold):
    original = declare()
    folder = OUT / fold
    old_fit = adapter.fit
    old_out, old_declare = subject.OUT, subject.declare
    old_trainer = adapter.frozen.fit

    def forbidden_trainer(*args, **kwargs):
        raise AssertionError('Canary-only authorization forbids optimizer training')

    def prepare_only(x, y, tuning=None, *, steps=None, seed=20260916, threads=2):
        assert steps is None and tuning is not None
        torch.set_num_threads(threads)
        encoder = adapter.FrameEncoder().fit(x, neural=True)
        stats = adapter.token_statistics(x, encoder)
        numbers, _ = adapter.frozen.tensors(encoder, x)
        bins, embedded, passthrough = adapter.grouped_bins(numbers, len(encoder.numeric))
        del numbers
        gc.collect()
        subject.guard()
        payload = dict(encoder=encoder, stats=stats, bins=bins, embedded=embedded, passthrough=passthrough)
        joblib.dump(payload, folder / 'fit_preparation.joblib')
        receipt = adapter.canary(x, encoder, bins, embedded, passthrough, stats, seed)
        receipt.update(status='passed', protocol_sha256=common.sha256(OUT / 'protocol.json'),
            producer_protocol_sha256=common.sha256(old_out / 'protocol.json'),
            fit_ids_hash=common.object_hash(x.index.tolist()),
            fit_rows=len(x), columns=list(x),
            stats_hash=common.object_hash(stats),
            bin_hash=common.object_hash([b.tolist() for b in bins]),
            encoder_hash=common.object_hash(dict(means=encoder.means.tolist(), scales=encoder.scales.tolist(),
                medians=encoder.medians.tolist(), categories=encoder.categories, numeric=encoder.numeric)),
            canary_ids_hash=common.object_hash(x.index[:8192].tolist()),
            canary_input_hash=common.object_hash([value.tolist() for value in adapter.frozen.tensors(encoder, x.iloc[:8192])]),
            preparation_sha256=common.sha256(folder / 'fit_preparation.joblib'),
            control_receipt_sha256=common.sha256(folder / 'control_replay.json'), resources=subject.guard(),
            full_training_started=False)
        common.write_json(folder / 'receipt.json', receipt)
        raise CanaryComplete()

    adapter.fit = prepare_only
    adapter.frozen.fit = forbidden_trainer
    subject.OUT = OUT
    subject.declare = lambda: original
    try:
        try:
            subject.run(fold)
        except CanaryComplete:
            print('CANARY_ONLY_COMPLETE', fold, common.sha256(folder / 'receipt.json'), flush=True)
        else:
            raise AssertionError('Expected canary-only termination')
    finally:
        adapter.fit = old_fit
        adapter.frozen.fit = old_trainer
        subject.OUT, subject.declare = old_out, old_declare


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    subject.pa.set_cpu_count(2)
    subject.pa.set_io_thread_count(1)
    with subject.threadpool_limits(2):
        if args.declare_only:
            declare()
            print(common.sha256(OUT / 'protocol.json'), flush=True)
        else:
            assert args.fold
            run(args.fold)
