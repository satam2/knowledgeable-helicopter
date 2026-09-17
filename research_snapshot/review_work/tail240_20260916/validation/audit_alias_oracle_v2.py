"""Repair schema spelling and normalize comparison dtypes; preserve v1 source."""
from pathlib import Path
import audit_alias_oracle as frozen

frozen.PHASE = 'PHASE_mvt'
frozen.FIELDS = [frozen.ID, frozen.FID, frozen.PHASE, frozen.TIME, 'FLIGHT_mvt',
                 'CALLSIGN_flt', 'ADEP_mvt', 'ADES_mvt', 'AOBT_3_flt']
frozen.OUT = frozen.ROOT / 'private_runs/tail240_20260916/validation/alias_oracle_v2'
original_norm = frozen.norm


def norm(series):
    clean, parts = original_norm(series)
    return clean, parts.astype(object)


frozen.norm = norm
if __name__ == '__main__':
    frozen.main()
