"""Known-location masks; preserve the original category as a separate signal."""
import numpy as np
import pandas as pd


def known_location(values, *, encoded=False):
    values = pd.Series(values, copy=False).astype('string')
    unknown = ['m:', 's0:', 's7:UNKNOWN', 's2:NA'] if encoded else ['', 'UNKNOWN', 'NA', '<missing>', 'MISSING']
    return (values.notna() & ~values.isin(unknown)).to_numpy(dtype=bool)


def feature_groups(columns):
    return {
        'stand': [c for c in columns if c.startswith('batch_source_stand_')],
        'query_runway': [c for c in columns if c.startswith('batch_surface_') and '_runway_' in c and 'active_runways' not in c],
        'arrival_runway': [c for c in columns if c.startswith('retro_') and '_arr_runway_' in c],
        'stand_equality': [c for c in columns if c.startswith('flat_') and c.endswith('_same_stand')],
        'runway_equality': [c for c in columns if c.startswith('flat_') and c.endswith('_same_runway')],
    }


def mask_query_features(frame, stand, runway, *, encoded=False):
    """Return copy; unknown groups become unavailable, equality remains false.

    Exact equality means an unknown peer can never match a known query. Hence
    query masks also remove every positive unknown/unknown equality. Airport
    active-runway counts cannot be corrected with a query mask and stay intact.
    """
    if len(frame) != len(stand) or len(frame) != len(runway):
        raise ValueError('Location masks must match feature rows')
    result = frame.copy()
    stand_known = known_location(stand, encoded=encoded)
    runway_known = known_location(runway, encoded=encoded)
    groups = feature_groups(frame.columns)
    for key, mask in [('stand', stand_known), ('query_runway', runway_known), ('arrival_runway', runway_known)]:
        result.loc[~mask, groups[key]] = np.nan
    for key, mask in [('stand_equality', stand_known), ('runway_equality', runway_known)]:
        result.loc[~mask, groups[key]] = 0.
    return result
