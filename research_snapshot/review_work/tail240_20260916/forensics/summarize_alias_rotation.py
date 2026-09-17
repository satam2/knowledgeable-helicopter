"""Post-build alias coverage and explicitly label-aware case diagnosis."""
from pathlib import Path
import numpy as np
import pandas as pd
from alias_rotation import ROOT,OUT,BASE,ID,SUFFIXES
from taxiout.artifacts import read_json,write_json,sha256,utc_now


def main():
    dest=ROOT / "private_runs/tail240_20260916/forensics/alias_rotation/assessment_v1"
    dest.mkdir(parents=True,exist_ok=False)
    write_json(dest / "protocol.json",{"created_utc":utc_now(),"source_sha256":sha256(__file__),
        "scope":"Coverage and posthoc exposed-score case diagnosis after feature cache completion. No target-driven feature changes or matchselection.",
        "case_selection":"Previously frozen100worstV4cases; report anyRomepilotintersection and everycoveragevariant.",
        "feature_manifest_sha256":sha256(OUT / "manifest.json")})
    receipt=read_json(OUT / "manifest.json")
    assert receipt["status"]=="complete"
    for name,digest in receipt["outputs"].items():assert sha256(OUT / name)==digest
    feature=pd.read_parquet(OUT / "features.parquet").set_index(ID)
    query=pd.read_parquet(OUT / "queries.parquet").set_index(ID)
    feature=feature.loc[query.index]
    oldmanifest=read_json(BASE / "manifest.json")
    assert sha256(BASE / "training_features.parquet")==oldmanifest["outputs"]["training_features.parquet"]
    old=pd.read_parquet(BASE / "training_features.parquet").set_index(ID).loc[feature.index]
    np.testing.assert_array_equal(feature[["alias_raw_"+s for s in SUFFIXES]],old[["opdi_"+s for s in SUFFIXES]])
    cases_path=ROOT / "private_runs/tail240_20260916/forensics/initial_v1/cases.parquet"
    assert sha256(cases_path)==read_json(cases_path.parent / "manifest.json")["outputs"][cases_path.name]
    cases=pd.read_parquet(cases_path).set_index(ID)
    matched_cases=cases.loc[cases.index.isin(feature.index)].join(feature,validate="one_to_one")
    matched_cases.reset_index().to_csv(dest / "top_cases_diagnostic.csv",index=False)
    allrows=query.join(feature)
    new=feature[["alias_learned_match","alias_lexical_match"]].max(axis=1).eq(1) & feature.alias_raw_match.eq(0)
    allrows.loc[new].reset_index().to_csv(dest / "additional_matches_observed.csv",index=False)
    totals={"queries":len(query),"raw_matches":int(feature.alias_raw_match.sum()),"raw_rotation":int(feature.alias_raw_previous_same_airport.sum()),
        "new_match_union":int(new.sum()),"top_rome_missing_cases":len(matched_cases),"top_cases_new_match_union":int(new.reindex(matched_cases.index).sum())}
    for mode in ["learned","lexical"]:
        totals[mode+"_new_matches"]=int((feature[f"alias_{mode}_match"].eq(1) & feature.alias_raw_match.eq(0)).sum())
        totals[mode+"_new_rotations"]=int((feature[f"alias_{mode}_previous_same_airport"].eq(1) & feature.alias_raw_previous_same_airport.eq(0)).sum())
        totals[mode+"_coverage_gate_ge10"]=totals[mode+"_new_matches"]>=10
    write_json(dest / "summary.json",{"status":"passed","totals":totals,"months":receipt["months"],"raw_control_parity":True,
        "source_sha256":sha256(__file__),"protocol_sha256":sha256(dest / "protocol.json"),
        "outputs":{p.name:sha256(p) for p in dest.iterdir() if p.is_file()}})
    print(totals)
    cols=["fold","FLIGHT_mvt","TAXITIME_SEC_mvt","prediction_sec","alias_raw_match","alias_learned_match","alias_lexical_match","alias_learned_previous_same_airport","alias_lexical_previous_same_airport"]
    print(matched_cases[cols].to_string())


if __name__=="__main__":main()
