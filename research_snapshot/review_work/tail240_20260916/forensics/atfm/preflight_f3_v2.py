"""Bounded source-discovery preflight; no matrices, labels, or model fitting."""
import tune_atfm_f3_v2 as fit


def main():
    protocol = fit.declare()
    common = fit.common
    path = fit.OUT / 'discovery_preflight.json'
    assert not path.exists()
    original, counts = fit.schema_discovery.cached_discovery(fit.ORIGINAL_SOURCES, protocol['columns'][:387])
    candidate = fit.sources(protocol['columns'])
    assert candidate[:-1] == original
    marker_path = fit.CONTROL / 'F3/control387/manifest.json'
    marker = common.read_json(marker_path)
    assert marker['status'] == 'complete' and marker['params'] == protocol['params']
    frozen = [{key: item[key] for key in ['path', 'sha256', 'columns']} for item in marker['feature_receipts']]
    actual = [dict(path=str(p), sha256=h, columns=names) for p, h, fill, names in original]
    assert actual == frozen
    base = fit.ROOT / 'private_runs/screening_230/data/interim/features/a2a101f52a0aa418'
    for p, _, fill, _ in original:
        assert fill == (p.parent != base and 'sequence_flatten' not in str(p.parent))
    assert candidate[-1][2] is False and candidate[-1][3] == fit.addition.FEATURES
    encoder = common.read_json(fit.CONTROL / 'F3/control387/encoder.json')
    assert encoder['columns'] == protocol['columns'][:387]
    assert counts['physical_reads'] == 19 and fit.DISCOVERY_RECEIPTS[-1]['physical_reads'] == 19
    memory = fit.psutil.Process().memory_info()
    peak = getattr(memory, 'peak_wset', memory.rss)
    assert peak < 2 * 1024**3
    common.write_json(path, dict(status='passed', source_sha256=common.sha256(__file__),
        wrapper_sha256=common.sha256(fit.__file__), helper_sha256=common.sha256(fit.schema_discovery.__file__),
        protocol_sha256=common.sha256(fit.OUT / 'protocol.json'), control_manifest_sha256=common.sha256(marker_path),
        original_discovery=counts, candidate_discovery=fit.DISCOVERY_RECEIPTS[-1],
        original_base_tuple_equality=True, ordered_control_receipt_equality=True, frozen_fill_policy_verified=True,
        peak_bytes=peak, rss_bytes=memory.rss, private_targets_read=False, model_fit=False,
        tuples=[dict(path=str(p), sha256=h, fill=fill, columns=names) for p, h, fill, names in candidate]))
    print('PREFLIGHT_PASS', 'peak_bytes', peak, 'original', counts, 'candidate', fit.DISCOVERY_RECEIPTS[-1], flush=True)


if __name__ == '__main__':
    main()


