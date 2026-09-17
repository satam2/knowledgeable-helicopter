"""Assess frozen F3 v2 artifacts with unchanged v1 formulas."""
import assess_following_tune as frozen

frozen.fit.OUT = frozen.common.external_path(frozen.ROOT / 'private_runs/tail240_20260916/models/following_groups_tune_v2')
frozen.OUT = frozen.common.external_path(frozen.ROOT / 'private_runs/tail240_20260916/models/following_groups_assessment_v2')

if __name__ == '__main__':
    frozen.run('F3')
