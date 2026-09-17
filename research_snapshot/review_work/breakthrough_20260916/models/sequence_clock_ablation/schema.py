"""Exact declared 24-channel departure-clock ablation from flatten112."""
CLOCK_FIELDS = ['offset_AOBT_3_flt', 'offset_EOBT_1_flt', 'offset_IOBT_flt',
                'offset_LOBT_flt', 'offset_SCHED_TIME_UTC_mvt', 'aobt_second']
OTHER_FIELDS = ['movement_second', 'arrival_duration', 'age_sec', 'landing_age_sec',
                'same_runway', 'same_stand', 'same_operator', 'padding_missing']


def selection(features):
    expected = [f'flat_{phase}{rank}_{name}' for phase in ('dep', 'arr')
                for rank in range(1, 5) for name in CLOCK_FIELDS + OTHER_FIELDS]
    if list(features) != expected:
        raise ValueError('Expected frozen ordered flatten112 schema')
    dropped = [f'flat_dep{rank}_{name}' for rank in range(1, 5) for name in CLOCK_FIELDS]
    retained = [name for name in features if name not in set(dropped)]
    assert len(dropped) == 24 and len(retained) == 88
    assert len(set(dropped) | set(retained)) == 112 and not set(dropped).intersection(retained)
    return dropped, retained
