"""Use canonical JSON fold dates while preserving the prepared ARR experiment."""
import lightgbm
import hashlib
import json
from pathlib import Path
import monthly_arrival as frozen

EXPECTED = 'f5ecc7b676c236319c5aaef275e2fe273c5cceaca47d110c810b1a65039aa2d1'
SOURCE = Path(frozen.__file__)


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    original = frozen.core.common.fold_data

    def normalized(*args, **kwargs):
        index, split, remaining = original(*args, **kwargs)
        normalized_split = json.loads(json.dumps(split, default=str))
        assert frozen.core.object_hash(split) == frozen.core.object_hash(normalized_split)
        return index, normalized_split, remaining

    frozen.OUT = frozen.ROOT / 'private_runs/breakthrough_20260916/retrospective_models/monthly_arrival_v3'
    frozen.__file__ = str(Path(__file__))
    frozen.core.common.fold_data = normalized
    try:
        frozen.main()
    finally:
        frozen.core.common.fold_data = original
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED


if __name__ == '__main__':
    main()
