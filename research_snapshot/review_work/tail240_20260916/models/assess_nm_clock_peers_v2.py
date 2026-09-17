"""Evaluate the identical NM436 gate with its final cache/runtime binding."""
import assess_ordinary_state as assessment
import nm_clock_peers_tune_v2 as wrapper

assessment.fit = wrapper.frozen
assessment.OUT = wrapper.common.external_path(wrapper.frozen.OUT.parent / 'nm_clock_peers_assessment_v2')

if __name__ == '__main__':
    assessment.run()
