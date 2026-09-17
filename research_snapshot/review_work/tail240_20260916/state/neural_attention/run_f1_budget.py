"""F1-only tighter resource envelope after successful full-preparation canary."""
import torch
import lightgbm
import argparse
import run_attention as frozen

common = frozen.common
OUT = common.external_path(frozen.OUT.parent / 'f1_budget_v1')
CANARY = frozen.OUT.parent / 'canary_v1/F1/receipt.json'


def guard12():
    info = frozen.psutil.Process().memory_info()
    assert max(info.rss, info.peak_wset) < 12 * 1024**3, f'F1 12GiB resource budget: {info}'
    available = frozen.psutil.virtual_memory().available
    assert available >= 8 * 1024**3, 'Host reserve below8GiB'
    return dict(rss_bytes=info.rss, peak_wset_bytes=info.peak_wset, available_bytes=available)


def declare():
    producer = frozen.declare()
    receipt = common.read_json(CANARY)
    assert receipt['status'] == 'passed' and not receipt['full_training_started']
    assert receipt['optimizer_updates'] == 0
    assert receipt['producer_protocol_sha256'] == common.sha256(frozen.OUT / 'protocol.json')
    assert receipt['resources']['peak_wset_bytes'] < 12 * 1024**3
    record = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=common.sha256(frozen.OUT / 'protocol.json'),
        canary_receipt_sha256=common.sha256(CANARY), fold='F1',
        change='Resource-only guard currentRSS/historicalpeak <12GiB; eightGiBhostreserve;20GiBminimumwrapperstartup plus originalstricter28GiBstartup remains. Sameoriginalproducerdestination/source/algorithm/seed/batch/fullcohorts.',
        limitation='Periodic guard uses historical process peak; not an OS enforced hard allocation limit. No F3 allocation under this wrapper.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == record
    else:
        common.write_json(path, record)
    return producer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    declare()
    if args.declare_only:
        print(common.sha256(OUT / 'protocol.json'), flush=True)
    else:
        assert frozen.psutil.virtual_memory().available >= 20 * 1024**3
        assert not (OUT / 'launch.json').exists()
        common.write_json(OUT / 'launch.json', dict(created_utc=common.utc_now(), resources=guard12(),
            protocol_sha256=common.sha256(OUT / 'protocol.json')))
        frozen.guard = guard12
        frozen.pa.set_cpu_count(2)
        frozen.pa.set_io_thread_count(1)
        with frozen.threadpool_limits(2):
            frozen.run('F1')
