"""Bind the unchanged 436-field test to the Windows-safe peer cache producer."""
import nm_clock_peers_tune as frozen
import build_v2 as peers
import argparse

common = frozen.common
OLD_OUT = frozen.OUT
ORIGINAL_DECLARE = frozen.declaration
frozen.peers = peers
frozen.OUT = common.external_path(OLD_OUT.parent / 'nm_clock_peers_tune_v2')


def declaration():
    record = ORIGINAL_DECLARE()
    path = frozen.OUT / 'runtime_binding.json'
    binding = dict(wrapper_sha256=common.sha256(__file__), producer_sha256=common.sha256(peers.__file__),
        producer_protocol_sha256=common.sha256(peers.OUT / 'protocol.json'),
        tuner_protocol_sha256=common.sha256(frozen.OUT / 'protocol.json'),
        scope='Only peer producer/cache binding changes. v1 builder wrotecompleteparquet but failedWindowsmemmapcleanup beforemanifest. No model hasrununder v1tuner.')
    if path.exists():
        assert common.read_json(path) == binding
    else:
        common.write_json(path, binding)
    return record


frozen.declaration = declaration

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--fold', choices=['F1', 'F3'])
    args = parser.parse_args()
    if args.declare_only:
        declaration()
    else:
        assert args.fold
        frozen.run(args.fold)
