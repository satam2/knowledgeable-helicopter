"""Independent chronological integrity checks; no training dependencies."""
import itertools
import pandas as pd

ID = 'MVT_ID_mvt'
FLIGHT = 'FLIGHT_ID_mvt'
TIME = 'MVT_TIME_UTC_mvt'
STRICT_STAGES = ('train', 'stop', 'calibration', 'evaluation')
BOUNDARIES = {
    'F1': {'train': ('2025-01-01','2025-04-01'), 'stop': ('2025-04-01','2025-05-01'),
           'calibration': ('2025-05-01','2025-06-01'), 'evaluation': ('2025-06','2025-07','2025-12')},
    'F3': {'train': ('2025-01-01','2025-08-01'), 'stop': ('2025-08-01','2025-09-01'),
           'calibration': ('2025-09-01','2025-10-01'), 'evaluation': ('2025-10','2025-11','2025-12')},
}


def reconstruct_declared_cohorts(metadata, fold):
    """All later raw stages participate in purges, even if later purged themselves."""
    if list(metadata.columns) != [ID, FLIGHT, TIME]:
        raise ValueError('Only identifier, related-flight and timestamp metadata are allowed')
    times = pd.to_datetime(metadata[TIME], utc=True, errors='raise')
    spec = BOUNDARIES[fold]
    masks = {stage: times.ge(pd.Timestamp(spec[stage][0],tz='UTC')) &
             times.lt(pd.Timestamp(spec[stage][1],tz='UTC')) for stage in STRICT_STAGES[:-1]}
    masks['evaluation'] = times.dt.strftime('%Y-%m').isin(spec['evaluation'])
    original = {stage: selected.copy() for stage, selected in masks.items()}
    purges = {}
    for index, stage in enumerate(STRICT_STAGES[:-1]):
        later = pd.Series(False,index=metadata.index)
        for later_stage in STRICT_STAGES[index+1:]:
            later |= original[later_stage]
        related = set(metadata.loc[later,FLIGHT].dropna())
        removed = masks[stage] & metadata[FLIGHT].notna() & metadata[FLIGHT].isin(related)
        purges[stage] = int(removed.sum())
        masks[stage] &= ~removed
    ids = {stage:metadata.loc[mask,ID].tolist() for stage,mask in masks.items()}
    validate_stages(metadata,ids)
    ids['refit'] = metadata.loc[masks['train'] | masks['stop'],ID].tolist()
    validate_refit(metadata,ids)
    return ids,purges


def validate_refit(metadata, stage_ids):
    base = {stage:stage_ids[stage] for stage in STRICT_STAGES}
    summary = validate_stages(metadata,base)
    if set(stage_ids) != set(STRICT_STAGES) | {'refit'}:
        raise ValueError('Unexpected stages or missing refit')
    union = set(base['train']) | set(base['stop'])
    expected = metadata.loc[metadata[ID].isin(union),ID].tolist()
    if list(stage_ids['refit']) != expected:
        raise ValueError('Refit must be the source-ordered union of train and stopping')
    summary['refit'] = dict(rows=len(expected), intentional_overlap_with=['train','stop'])
    return summary


def validate_stages(metadata, stage_ids, order=STRICT_STAGES):
    if list(metadata.columns) != [ID, FLIGHT, TIME]:
        raise ValueError('Only identifier, related-flight and timestamp metadata are allowed')
    if not metadata[ID].is_unique or metadata[ID].isna().any():
        raise ValueError('Metadata IDs must be unique and nonnull')
    if set(stage_ids) != set(order):
        raise ValueError('All declared stages, including independent stopping, are required')
    data = metadata.copy()
    data[TIME] = pd.to_datetime(data[TIME], utc=True, errors='raise')
    if data[TIME].isna().any():
        raise ValueError('Missing movement timestamp')
    indexed = data.set_index(ID, drop=False)
    groups = {}
    for name in order:
        ids = list(stage_ids[name])
        if not ids or pd.Index(ids).has_duplicates or pd.isna(ids).any():
            raise ValueError('Stage IDs must be nonempty, unique and nonnull')
        if not set(ids).issubset(set(indexed.index)):
            raise ValueError('Stage contains unknown ID')
        expected_order = data.loc[data[ID].isin(ids), ID].tolist()
        if ids != expected_order:
            raise ValueError('Stage changed source order')
        groups[name] = indexed.loc[ids]
    for earlier, later in itertools.combinations(order, 2):
        a, b = groups[earlier], groups[later]
        if set(a[ID]) & set(b[ID]):
            raise ValueError('Movement ID crosses stages')
        if not a[TIME].max() < b[TIME].min():
            raise ValueError('Stages are not strictly chronological')
        if set(a[FLIGHT].dropna()) & set(b[FLIGHT].dropna()):
            raise ValueError('Related flight crosses stages')
    return {name: dict(rows=len(group), minimum=str(group[TIME].min()), maximum=str(group[TIME].max()))
            for name, group in groups.items()}


def validate_declared_access(events, stage_ids):
    """Check an access ledger; this cannot prove unlogged reads did not occur."""
    allowed = {'gradient': 'train', 'preprocessing': 'train', 'stopping': 'stop',
               'calibration': 'calibration', 'evaluation': 'evaluation'}
    frozen = False
    observed = set()
    for event in events:
        operation = event['operation']
        if operation == 'freeze':
            if frozen or not event.get('state_sha256') or not event.get('prediction_sha256'):
                raise ValueError('Require one bound state/prediction freeze')
            frozen = True
            continue
        if operation not in allowed:
            raise ValueError('Unrecognized operation')
        if operation == 'evaluation' and not frozen:
            raise ValueError('Evaluation labels opened before freeze')
        if operation != 'evaluation' and frozen:
            raise ValueError('Fitting or selection after freeze')
        ids = list(event['label_or_fit_ids'])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Access IDs must be nonempty and unique')
        if not set(ids).issubset(set(stage_ids[allowed[operation]])):
            raise ValueError('Operation reads labels or fits preprocessing outside its stage')
        observed.add(operation)
    if observed != set(allowed) or not frozen:
        raise ValueError('Incomplete training/stopping/calibration/evaluation ledger')
    return True


def validate_refit_access(events, stage_ids):
    allowed = {'gradient':'train','preprocessing':'train','stopping':'stop',
               'refit_gradient':'refit','refit_preprocessing':'refit',
               'calibration':'calibration','evaluation':'evaluation'}
    frozen = False
    observed = set()
    preprocess_bindings = {}
    for event in events:
        operation = event['operation']
        if operation == 'freeze':
            if frozen or not event.get('state_sha256') or not event.get('prediction_sha256'):
                raise ValueError('Require one bound state/prediction freeze')
            if observed != set(allowed)-{'evaluation'}:
                raise ValueError('Freeze before all fit stages completed')
            frozen = True
            continue
        if operation not in allowed:
            raise ValueError('Unrecognized operation')
        if operation == 'evaluation' and not frozen:
            raise ValueError('Evaluation labels opened before freeze')
        if operation != 'evaluation' and frozen:
            raise ValueError('Fitting or selection after freeze')
        if operation.startswith('refit_') and 'stopping' not in observed:
            raise ValueError('Refit started before stopping completed')
        if operation == 'calibration' and not {'refit_gradient','refit_preprocessing'}.issubset(observed):
            raise ValueError('Calibration started before refit completed')
        ids = list(event['label_or_fit_ids'])
        if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(set(stage_ids[allowed[operation]])):
            raise ValueError('Operation access outside its stage or invalid IDs')
        if operation in ('preprocessing','refit_preprocessing'):
            binding = event.get('fit_cohort_sha256')
            if not binding:
                raise ValueError('Preprocessor requires fit cohort binding')
            preprocess_bindings[operation] = binding
        observed.add(operation)
    if observed != set(allowed) or not frozen:
        raise ValueError('Incomplete refit access ledger')
    if preprocess_bindings['preprocessing'] == preprocess_bindings['refit_preprocessing']:
        raise ValueError('Refit preprocessing must be freshly fitted on refit cohort')
    return True


def validate_prior_support(support, whole_query_month, excluded_flight_ids=()):
    """Support is the actual prior fit cohort; queries contain the entire month."""
    if support.empty or whole_query_month.empty:
        raise ValueError('Empty prior support requires a separately declared cold-start fallback')
    a = support.copy()
    b = whole_query_month.copy()
    for frame in (a, b):
        if list(frame.columns) != [ID, FLIGHT, TIME] or not frame[ID].is_unique:
            raise ValueError('Invalid prior metadata')
        frame[TIME] = pd.to_datetime(frame[TIME], utc=True, errors='raise')
        if frame[TIME].isna().any():
            raise ValueError('Missing prior timestamp')
    months = b[TIME].dt.strftime('%Y-%m').unique()
    if len(months) != 1:
        raise ValueError('Prior query must cover one calendar month')
    boundary = pd.Timestamp(months[0]+'-01', tz='UTC')
    if not a[TIME].lt(boundary).all():
        raise ValueError('Prior support contains query-month or future labels')
    excluded = set(b[FLIGHT].dropna()) | set(excluded_flight_ids)
    if set(a[FLIGHT].dropna()) & excluded:
        raise ValueError('Prior support includes related query or outer-purged flights')
    if set(a[ID]) & set(b[ID]):
        raise ValueError('Prior support includes query IDs')
    return True


def validate_exposure(protocol):
    if protocol.get('historically_exposed') is not True:
        raise ValueError('Known development exposure must be explicit')
    if protocol.get('fresh_holdout') is not False:
        raise ValueError('Previously exposed months cannot be declared fresh holdout')
    if protocol.get('reuse_previously_stopped_models') is not False:
        raise ValueError('Previously selected checkpoints cannot be fresh chronological experts')
    if protocol.get('ranking_targets_accessed') is not False:
        raise ValueError('Ranking labels must remain inaccessible')
    return True
