"""Correct completion eligibility before prior-leg selection and bind frozen inputs."""
import json
import numpy as np
import build as base

def build_month(query, public, month):
    public = public.copy()
    public["first_seen"] = base.timestamps(public["first_seen"])
    public["last_seen"] = base.timestamps(public["last_seen"])
    public = public.loc[public["last_seen"].gt(public["first_seen"]) & public["first_seen"].dt.strftime("%Y%m").eq(month)].copy()
    result = base.build_month(query, public, month)
    result[base.FEATURES[3:5]] = np.float32(0)
    result[base.FEATURES[5:]] = np.float32(np.nan)
    for name in ["icao24", "adep", "ades"]:
        public[name] = base.normalize(public[name])
    public["cs"] = base.normalize(public["flt_id"])
    query = query.copy()
    query[base.TIME] = base.timestamps(query[base.TIME])
    for name in ["ADEP_mvt", "ADES_mvt", "FLIGHT_mvt"]:
        query[name] = base.normalize(query[name])
    matched = query.loc[query[base.ID].isin(result.index[result["opdi_match"].eq(1)])]
    candidates = public.loc[public["cs"].isin(matched["FLIGHT_mvt"])]
    groups = {key: group for key, group in candidates.groupby(["cs", "adep", "ades"], sort=False)}
    hits = []
    for row in matched.itertuples(index=False):
        fields = row._asdict()
        group = groups[(fields["FLIGHT_mvt"], fields["ADEP_mvt"], fields["ADES_mvt"])]
        hit = group.loc[(fields[base.TIME] - group["first_seen"]).dt.total_seconds().abs().le(600)].iloc[0]
        hits.append((fields, hit))
    tails = {hit["icao24"] for _, hit in hits}
    by_tail = {key: group for key, group in public.loc[public["icao24"].isin(tails)].groupby("icao24", sort=False)}
    for fields, hit in hits:
        legs = by_tail[hit["icao24"]]
        before = legs.loc[legs["first_seen"].lt(hit["first_seen"])]
        if before["last_seen"].ge(hit["first_seen"]).any():
            continue
        eligible = before.loc[before["last_seen"].lt(hit["first_seen"]) & before["last_seen"].lt(fields[base.TIME])]
        if eligible.empty:
            continue
        latest = eligible.loc[eligible["last_seen"].eq(eligible["last_seen"].max())]
        if len(latest) != 1:
            continue
        previous = latest.iloc[0]
        result.loc[fields[base.ID], "opdi_previous_leg_available"] = 1
        if previous["ades"] != fields["ADEP_mvt"]:
            continue
        result.loc[fields[base.ID], "opdi_previous_same_airport"] = 1
        result.loc[fields[base.ID], "opdi_ground_interval_sec"] = (hit["first_seen"] - previous["last_seen"]).total_seconds()
        result.loc[fields[base.ID], "opdi_previous_leg_duration_sec"] = (previous["last_seen"] - previous["first_seen"]).total_seconds()
        result.loc[fields[base.ID], "opdi_previous_leg_age_at_takeoff_sec"] = (fields[base.TIME] - previous["last_seen"]).total_seconds()
    return result

def main():
    frozen = json.loads((base.ROOT / "private_runs/submission_v2/protocol.json").read_text())
    audit = json.loads((base.ROOT / "private_runs/screening_230/reports/data_audit.json").read_text())
    paths = sorted(base.RAW.glob("training_*.parquet")) + [base.RAW / "ranking.parquet"]
    metadata = base.ROOT / "private_runs/screening_230/data/interim/audit/departures.parquet"
    expected = {str(path): frozen["raw_hashes"][path.name] for path in paths}
    expected[str(metadata)] = audit["artifacts"][metadata.name]
    for path, digest in expected.items():
        assert base.sha(path) == digest, path
    for name in ["training_features.parquet", "ranking_features.parquet", "manifest.json"]:
        assert not (base.OUT / name).exists(), "Preserve existing output"
    original_month = base.build_month
    def dispatched(query, public, month):
        base.build_month = original_month
        try:
            return build_month(query, public, month)
        finally:
            base.build_month = dispatched
    base.build_month = dispatched
    base.main()
    for path, digest in expected.items():
        assert base.sha(path) == digest, path
    path = base.OUT / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["builder_base_sha256"] = manifest["source_sha256"]
    manifest["source_sha256"] = base.sha(__file__)
    manifest["builder_base_path"] = str(base.Path(base.__file__).resolve())
    manifest["builder_source_path"] = str(base.Path(__file__).resolve())
    manifest["frozen_input_hashes_verified_before_and_after"] = expected
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
