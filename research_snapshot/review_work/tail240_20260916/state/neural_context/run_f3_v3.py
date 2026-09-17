"""F3 unchanged neural experiment with bounded metadata and quantile allocation."""
import torch
import argparse
import sys
import run_f3_v2 as prior

frozen = prior.frozen
sys.path.insert(0, str(frozen.ROOT / 'review_work/tail240_20260916/models'))
import schema_discovery
OUT = frozen.common.external_path(frozen.ROOT / 'private_runs/tail240_20260916/state/neural_context/v3')


def declare():
    old = prior.declare()
    record = dict(old, wrapper_sha256=frozen.common.sha256(__file__),
        v2_protocol_sha256=frozen.common.sha256(prior.OUT / 'protocol.json'),
        grouped_bins_source_sha256=frozen.common.sha256(prior.__file__),
        schema_helper_sha256=frozen.common.sha256(schema_discovery.__file__),
        metadata_change='Only feature-source discovery caches pq.read_schema once per path. Same original discovery function, returned tuples/order/hashes/columns. Arrow restored before matrix loading. No encoder/model/numeric/cohort/training changes.',
        resources='ExclusiveCUDA,2CPU,20GiB sampledRSS and historicalpeak_wset checkpoints,8GiB hostreserve,>=28GiB startupavailable; no OS hardlimit. Await parent coordination and validator GPUrelease.')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        assert frozen.common.read_json(path) == record
    else:
        frozen.common.write_json(path, record)
    return record


def peak_guard():
    info = frozen.psutil.Process().memory_info()
    peak = getattr(info, 'peak_wset', info.rss)
    assert max(info.rss, peak) < 20 * 1024**3, f'20GiB resource budget: RSS={info.rss},peak={peak}'
    assert frozen.psutil.virtual_memory().available >= 8 * 1024**3, 'Host reserve below8GiB'
    return int(info.rss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    protocol = declare()
    if args.declare_only:
        print(frozen.common.sha256(OUT / 'protocol.json'), flush=True)
        return
    available = frozen.psutil.virtual_memory().available
    assert available >= 28 * 1024**3, 'Coordinated launch requires>=28GiB available'
    path = OUT / 'launch_resource.json'
    assert not path.exists(), 'Preserve prior launch'
    frozen.common.write_json(path, dict(created_utc=frozen.common.utc_now(), available_bytes=available,
        protocol_sha256=frozen.common.sha256(OUT / 'protocol.json'), source_chain=dict(
        wrapper=frozen.common.sha256(__file__), grouped_bins=frozen.common.sha256(prior.__file__),
        helper=frozen.common.sha256(schema_discovery.__file__), trainer=frozen.common.sha256(frozen.__file__))))
    original = frozen.risk.feature_sources

    def cached(desired):
        values, receipt = schema_discovery.cached_discovery(original, desired)
        receipt['tuples'] = [[str(p), digest, fill, columns] for p, digest, fill, columns in values]
        frozen.common.write_json(OUT / 'discovery_receipt.json', receipt)
        peak_guard()
        return values

    frozen.risk.feature_sources = cached
    frozen.OUT = OUT
    frozen.freeze = lambda: protocol
    frozen.guard = peak_guard
    frozen.adapter.fit_bins = prior.grouped_fit_bins
    frozen.pa.set_cpu_count(2)
    frozen.pa.set_io_thread_count(1)
    with frozen.threadpool_limits(2):
        frozen.run('F3')
    info = frozen.psutil.Process().memory_info()
    frozen.common.write_json(OUT / 'completed_resource.json', dict(rss_bytes=info.rss,
        peak_wset_bytes=getattr(info, 'peak_wset', info.rss), available_bytes=frozen.psutil.virtual_memory().available))


if __name__ == '__main__':
    main()
