"""Fresh F1 retry with14GiB process budget; scientific protocol unchanged."""
import torch
import lightgbm
import argparse
from pathlib import Path
import run_capacity as frozen

common = frozen.common
ORIGINAL_OUT = frozen.OUT
OUT = common.external_path(ORIGINAL_OUT.parent / 'v2')
FAILED = ORIGINAL_OUT / 'models/F1/resource_failure.json'
CANARY = ORIGINAL_OUT / 'canary/F1/receipt.json'


def guard14(fold):
    assert fold == 'F1', 'No F3 allocation'
    info = frozen.psutil.Process().memory_info()
    assert max(info.rss, info.peak_wset) < 14*1024**3, f'F1 14GiB process budget {info}'
    available = frozen.psutil.virtual_memory().available
    assert available >= 8*1024**3, '8GiB host reserve'
    return dict(rss_bytes=info.rss, peak_wset_bytes=info.peak_wset, available_bytes=available)


def declare():
    frozen.declare()
    failed = common.read_json(FAILED)
    assert failed['status'] == 'failed_resource_guard' and failed['exit_code'] == 1
    assert not (FAILED.parent / 'fit_model.joblib').exists()
    assert not (FAILED.parent / 'manifest.json').exists()
    canary = common.read_json(CANARY)
    assert canary['status'] == 'passed' and canary['members'] == 32 and canary['optimizer_updates'] == 0
    protocol_hash = common.sha256(ORIGINAL_OUT / 'protocol.json')
    assert failed['producer_protocol_sha256'] == canary['protocol_sha256'] == protocol_hash
    value = dict(source_sha256=common.sha256(__file__), producer_protocol_sha256=protocol_hash,
        canary_receipt_sha256=common.sha256(CANARY), failed_receipt_sha256=common.sha256(FAILED),
        failed_attempt_files={p.name:common.sha256(p) for p in FAILED.parent.iterdir() if p.is_file() and p.name!='matrix.float32'},
        fold='F1', resources='14GiB currentRSS/historicalpeak,>=22GiBstartup,8GiBhostreserve,2CPU/exclusiveGPU. Holdheavyvalidation; noF3allocation.',
        only_changes='Resourceguard14GiB and reroute exactly originalOUT/models/F1 to newOUT/models/F1 using scoped external_path wrapper. Original scientific protocol/canary/source/data/seed/optimizer/batch untouched; fresh fit from seed, no partial weights.',
        output_binding='Model manifest retains original scientific protocol; v2launch/protocol bind resource override and new artifact root. Original failed outputs remain intact.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert common.read_json(path) == value
    else:
        common.write_json(path, value)
    return value


def run():
    declare()
    assert frozen.psutil.virtual_memory().available >= 22*1024**3
    assert not (OUT / 'launch.json').exists()
    common.write_json(OUT / 'launch.json', dict(created_utc=common.utc_now(), resources=guard14('F1'),
        protocol_sha256=common.sha256(OUT / 'protocol.json')))
    original_path = common.external_path
    original_guard = frozen.guard
    def redirect(path):
        checked = original_path(path)
        if Path(checked).resolve() == (ORIGINAL_OUT / 'models/F1').resolve():
            return original_path(OUT / 'models/F1')
        return checked
    common.external_path = redirect
    frozen.guard = guard14
    try:
        frozen.fit_fold('F1')
    finally:
        common.external_path = original_path
        frozen.guard = original_guard


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    frozen.torch.set_num_threads(2)
    frozen.pa.set_cpu_count(2)
    frozen.pa.set_io_thread_count(1)
    with frozen.threadpool_limits(2):
        if args.declare_only:
            declare()
            print(common.sha256(OUT / 'protocol.json'), flush=True)
        else:
            run()
