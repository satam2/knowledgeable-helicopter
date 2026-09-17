"""Same independent reconstruction against final external-path-bound v2 artifacts."""
from pathlib import Path
import verify_chronological_cohorts as frozen

ROOT = frozen.ROOT
frozen.BASE = ROOT / 'private_runs/tail240_20260916/forensics/chronological_cohorts/v2'
frozen.OUT = ROOT / 'private_runs/tail240_20260916/validation/chronological_cohorts_v2'


if __name__ == '__main__':
    frozen.main()
    old = frozen.read(ROOT / 'private_runs/tail240_20260916/forensics/chronological_cohorts/v1/manifest.json')
    new = frozen.read(frozen.BASE / 'manifest.json')
    assert old['folds'] == new['folds'] and old['outputs'] == new['outputs']
    frozen.write(frozen.OUT / 'version_binding.json', dict(
        wrapper_sha256=frozen.sha(Path(__file__)), verifier_sha256=frozen.sha(Path(frozen.__file__)),
        unchanged_all_v1_v2_cohort_and_ID_artifact_hashes=True,
        final_receipt_sha256=frozen.sha(frozen.OUT / 'receipt.json')))
