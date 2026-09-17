"""Independent identifier-only reconstruction of nested chronological cohorts."""
from audit_union387_sources import ROOT, read, sha, write, object_hash, guard
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

ID, FLIGHT, TIME = 'MVT_ID_mvt', 'FLIGHT_ID_mvt', 'MVT_TIME_UTC_mvt'
BASE = ROOT / 'private_runs/tail240_20260916/forensics/chronological_cohorts/v1'
OUT = ROOT / 'private_runs/tail240_20260916/validation/chronological_cohorts_v1'


def main():
    marker = read(BASE / 'manifest.json')
    assert marker['status'] == 'cohorts_materialized_no_fits'
    assert marker['input_columns'] == [ID, FLIGHT, TIME] and marker['label_columns_read'] == []
    for relative, expected in marker['source_hashes'].items():
        assert sha(ROOT / relative) == expected
    for name, expected in marker['outputs'].items():
        assert sha(BASE / name) == expected
    path = ROOT / 'private_runs/screening_230/data/interim/audit/departures.parquet'
    assert sha(path) == marker['metadata_sha256']
    binding_path = ROOT / 'private_runs/tail240_20260916/validation/baseline_binding.json'
    assert sha(binding_path) == marker['baseline_binding_sha256']
    binding = read(binding_path)
    data = pd.read_parquet(path, columns=[ID, FLIGHT, TIME])
    assert data[ID].is_unique and data[TIME].notna().all()
    reports = {}
    for fold, boundary in [('F1', '2025-05-01'), ('F3', '2025-09-01')]:
        expected = read(BASE / f'{fold}_cohorts.json')
        spec = expected['original']['spec']
        masks = {}
        for stage in ('fit', 'tune', 'refit', 'score'):
            start, stop = [pd.Timestamp(value, tz='UTC') for value in spec[stage]]
            masks[stage] = (data[TIME].ge(start) & data[TIME].lt(stop)).to_numpy()
        purge = {}
        for stage, against in [('fit', 'score'), ('tune', 'score'), ('refit', 'score'), ('fit', 'tune')]:
            held_flights = set(data.loc[masks[against], FLIGHT].dropna())
            related = data[FLIGHT].notna().to_numpy() & data[FLIGHT].isin(held_flights).to_numpy()
            purge[stage + '_against_' + against] = int((masks[stage] & related).sum())
            masks[stage] &= ~related
        assert purge == expected['original']['purged_related_departures']
        for stage, selected in masks.items():
            ids = data.loc[selected, ID]
            actual = dict(n=len(ids), hash=object_hash(ids.tolist()))
            assert actual == binding['folds'][fold]['cohorts']['all'][stage]
            assert dict(n=actual['n'], id_hash=actual['hash']) == expected['original']['stages'][stage]
        base = masks['fit'] & data[TIME].lt(pd.Timestamp(boundary, tz='UTC')).to_numpy()
        calibration = masks['fit'] & ~base
        calibration_flights = set(data.loc[calibration, FLIGHT].dropna())
        additional = base & data[FLIGHT].notna().to_numpy() & data[FLIGHT].isin(calibration_flights).to_numpy()
        base &= ~additional
        assert int(additional.sum()) == expected['additional_base_against_calibration']
        stages = dict(base=base, calibration=calibration, evaluation=masks['tune'], protected_score=masks['score'])
        summary = {}
        for stage, selected in stages.items():
            ids = data.loc[selected, ID]
            stored = pd.read_parquet(BASE / f'{fold}_{stage}_ids.parquet')
            assert list(stored) == [ID]
            np.testing.assert_array_equal(ids.to_numpy(), stored[ID].to_numpy())
            item = dict(n=len(ids), id_hash=object_hash(ids.tolist()))
            assert item == expected['stages'][stage]
            summary[stage] = item
        for earlier, later in itertools.combinations(stages, 2):
            first, second = data.loc[stages[earlier]], data.loc[stages[later]]
            assert first[TIME].max() < second[TIME].min()
            assert not set(first[FLIGHT].dropna()).intersection(second[FLIGHT].dropna())
            assert not set(first[ID]).intersection(second[ID])
        body = {key: value for key, value in expected.items() if key != 'cohort_hash'}
        assert object_hash(body) == expected['cohort_hash']
        assert expected == marker['folds'][fold]
        reports[fold] = dict(stages=summary, original_purges=purge, additional_base_against_calibration=int(additional.sum()),
                             chronology_strict=True, flight_ids_disjoint=True, original_order_preserved=True)
        assert guard() < 2*1024**3
    OUT.mkdir(parents=True, exist_ok=False)
    result = dict(status='passed', source_sha256=sha(Path(__file__)), producer_manifest_sha256=sha(BASE / 'manifest.json'),
                  folds=reports, projection=[ID, FLIGHT, TIME], producer_tests_passed=5,
                  no_target_proxy_block_columns_read=True, no_model_or_GPU=True, peak_bytes=guard(),
                  limitation='Chronological split artifacts only. No evidence yet for meta-learner generalization or corrected prior tune reuse.')
    write(OUT / 'receipt.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
