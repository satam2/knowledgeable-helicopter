"""Preserve the first attempt; normalize fold receipt dates before comparison."""
import hashlib
from pathlib import Path
import run as frozen
from taxiout.artifacts import json_safe

EXPECTED = 'a3aed582c3111be19be3c7e1dc18a96e578a0acdb8af3ce939c11bce7d76e7f2'
SOURCE = Path(frozen.__file__)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    original = frozen.base.common.fold_data

    def normalized(*args, **kwargs):
        index, split, remaining = original(*args, **kwargs)
        normalized_split = json_safe(split)
        assert frozen.base.object_hash(split) == frozen.base.object_hash(normalized_split)
        return index, normalized_split, remaining

    frozen.OUT = frozen.base.external_path(frozen.ROOT / 'private_runs/breakthrough_20260916/models/opdi_missing_v2')
    frozen.__file__ = str(Path(__file__))
    frozen.base.common.fold_data = normalized
    try:
        frozen.main()
    finally:
        frozen.base.common.fold_data = original
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED


if __name__ == '__main__':
    main()
