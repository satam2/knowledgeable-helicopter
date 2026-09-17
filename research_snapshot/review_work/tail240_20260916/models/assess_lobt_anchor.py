"""The predeclared seasonal fixed25 gate, with identical raw target cohorts."""
import assess_ordinary_state as assessment
import lobt_anchor_tune as fit

assessment.fit = fit
assessment.OUT = fit.common.external_path(fit.OUT.parent / 'lobt_anchor_assessment_v1')

if __name__ == '__main__':
    assessment.run()
