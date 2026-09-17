"""Apply the frozen two-month NM-peer gate to the current neural ensemble."""
import assess_ordinary_state as frozen
import nm_clock_peers_tune as fit

frozen.fit = fit
frozen.OUT = fit.common.external_path(fit.OUT.parent / 'nm_clock_peers_assessment_v1')

if __name__ == '__main__':
    frozen.run()
