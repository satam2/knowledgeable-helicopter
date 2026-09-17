"""Distinguish missing-source date conventions from observed-clock offsets."""

import numpy as np
import pandas as pd
from analyze_clocks import prepare, OUT, ID, TARGET, common, profile, json_finite
from taxiout.artifacts import sha256, utc_now, write_json
from taxiout.config import load_config
from taxiout.schema import FLIGHT_ID, MOVEMENT
from taxiout.splits import make_fold


def describe(frame):
    y = frame[TARGET].to_numpy(float)
    schedule = frame.proxy_schedule.to_numpy(float)
    missing = frame.missing_nm.to_numpy()
    masks = {'y_gt12h': y > 43200, 'y_gt2h': y > 7200,
        'missing_y_gt12h': missing & (y > 43200), 'missing_schedule_gt12h': missing & (schedule > 43200),
        'missing_schedule_negative': missing & (schedule < 0),
        'rome_missing_stand_missing': missing & frame.ADEP_mvt.eq('LIRF') & frame.STAND_mvt.isna(),
        'rome_missing_schedule_gt12h': missing & frame.ADEP_mvt.eq('LIRF') & (schedule > 43200)}
    if 'prediction' in frame:
        prediction = frame.prediction.to_numpy()
        masks.update(pred_gt12h=prediction > 43200,
            false_giant_y_lt1h_error_gt1h=(y < 3600) & (prediction - y > 3600),
            missed_giant_y_gt2h_error_gt1h=(y > 7200) & (y - prediction > 3600))
    result = {}
    for name, mask in masks.items():
        selected = frame.loc[mask]
        sy = selected[TARGET].to_numpy(float)
        ss = selected.proxy_schedule.to_numpy(float)
        item = {'n': len(selected), 'missing_nm_n': int(selected.missing_nm.sum()),
            'schedule_gap_within60_n': int((np.abs(sy - ss) <= 60).sum()),
            'schedule_gap_within300_n': int((np.abs(sy - ss) <= 300).sum()),
            'target_mod24h_in_0_2h_n': int(((sy % 86400 >= 0) & (sy % 86400 <= 7200)).sum()),
            'schedule_proxy_negative_n': int((ss < 0).sum()),
            'target_quantiles': np.quantile(sy, [0, .25, .5, .75, 1]).tolist() if len(sy) else [],
            'schedule_quantiles': np.quantile(ss[np.isfinite(ss)], [0, .25, .5, .75, 1]).tolist() if np.isfinite(ss).any() else [],
            'airport_counts': selected.ADEP_mvt.value_counts().loc[lambda x: x > 0].to_dict(),
            'day_counts': selected.day.value_counts().loc[lambda x: x > 0].head(10).to_dict()}
        if 'prediction' in frame:
            item['prediction_quantiles'] = np.quantile(selected.prediction, [0, .25, .5, .75, 1]).tolist() if len(selected) else []
            item['reference_sse_share'] = float(selected.squared_error.sum() / frame.squared_error.sum())
        result[name] = item
    return result


def main():
    data, _ = prepare()
    report = {'created_utc': utc_now(), 'source_sha256': sha256(__file__),
        'scope': 'Diagnostic target and prediction slices only; no inference rule selected or applied.',
        'all_training': describe(data), 'folds': {}}
    for fold in ['F1', 'F3']:
        idx, _ = make_fold(data[[ID, FLIGHT_ID, MOVEMENT]], load_config('configs/folds.yaml')[fold])
        score = data.iloc[idx['score']].copy()
        reference, _ = common.reference(fold)
        assert np.array_equal(score[ID], reference[ID])
        score['prediction'] = reference.prediction_sec.to_numpy()
        score['squared_error'] = reference.squared_error.to_numpy()
        report['folds'][fold] = {'fit': describe(data.iloc[idx['fit']]), 'score': describe(score)}
    write_json(OUT / 'extreme_days.json', json_finite(report))
    print('DONE extreme days', flush=True)


if __name__ == '__main__':
    main()
