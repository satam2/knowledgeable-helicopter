"""Preserve v1 and normalize saved V3 route labels to the declared audit groups."""
from pathlib import Path
import audit_v1 as audit

ORIGINAL_READ = audit.pd.read_parquet
LABELS = {'ordinary_ensemble': 'ordinary', 'missing_blend': 'missing',
          'finite_nonordinary_lgb225': 'finite_nonordinary'}


def read_with_route_labels(path, *args, **kwargs):
    frame = ORIGINAL_READ(path, *args, **kwargs)
    if Path(path).resolve() == (audit.NEW / 'ranking_predictions.parquet').resolve():
        assert set(frame.route.unique()) == set(LABELS)
        frame['route'] = frame.route.map(LABELS)
    return frame


if __name__ == '__main__':
    audit.OUT = audit.common.external_path(audit.ROOT / 'private_runs/tail240_20260916/score_gap/drift/v2')
    audit.pd.read_parquet = read_with_route_labels
    try:
        audit.main()
    finally:
        audit.pd.read_parquet = ORIGINAL_READ
    audit.common.write_json(audit.OUT / 'wrapper_receipt.json', dict(
        source_sha256=audit.common.sha256(__file__), imported_source_sha256=audit.common.sha256(audit.__file__),
        route_label_mapping=LABELS, receipt_sha256=audit.common.sha256(audit.OUT / 'receipt.json')))
