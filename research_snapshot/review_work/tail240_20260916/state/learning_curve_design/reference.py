"""Original union predictions contain IDs/predictions, never target labels."""
import numpy as np


def reference_predictions(frame, expected_ids):
    np.testing.assert_array_equal(frame['MVT_ID_mvt'], expected_ids)
    prediction=frame['prediction_sec'].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite reference prediction')
    return prediction
