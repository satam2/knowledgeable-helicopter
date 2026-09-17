"""Small synthetic checks for interaction signs and paired day removal."""
import numpy as np
from factorial_analysis import contrast_summary


def main():
    days = np.array(['a', 'a', 'b', 'b'])
    errors = {'past': np.array([10., 20., 30., 40.]),
              'past_future': np.array([8., 17., 26., 35.]),
              'past_day': np.array([9., 16., 27., 38.]),
              'all': np.array([5., 11., 20., 29.])}
    result = contrast_summary(errors, days)
    interaction = errors['all'] - errors['past_future'] - errors['past_day'] + errors['past']
    np.testing.assert_allclose(result['future_by_day_interaction']['delta_mse_sec2'], interaction.mean())
    for index, day in enumerate(['a', 'b']):
        np.testing.assert_allclose(result['future_by_day_interaction']['daily'][index]['leave_one_day_out_sec2'], interaction[days != day].mean())
    assert result['future_by_day_interaction']['delta_mse_sec2'] < 0
    additive = dict(errors, all=errors['past_future'] + errors['past_day'] - errors['past'])
    assert contrast_summary(additive, days)['future_by_day_interaction']['delta_mse_sec2'] == 0
    np.testing.assert_allclose(result['future_without_day']['delta_mse_sec2'], -3.5)
    print('PASS: explicit paired differences, negative synergy sign, additive zero, exact day removal')


if __name__ == '__main__':
    main()
