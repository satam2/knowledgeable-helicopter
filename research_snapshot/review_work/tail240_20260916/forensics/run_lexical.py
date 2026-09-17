"""Fixed historical-template ablation for observable flight-code morphology."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "review_work/breakthrough_20260916/missing"))
import run_missing_models as prior
from taxiout.artifacts import read_json, write_json, sha256, utc_now, object_hash

OUT = ROOT / "private_runs/tail240_20260916/forensics/models/lexical_v1"
BASE = ROOT / "private_runs/breakthrough_20260916/missing/route_composition_v4"
ARMS = ["control", "lexical"]
NUMERIC = ["lex_length", "lex_prefix_length", "lex_number_length", "lex_suffix_length", "lex_parse_ok",
    "lex_repeated_prefix_suffix", "lex_repeated_extra_length", "lex_leading_zero", "lex_trailing_repeat"]
CATEGORICAL = ["lex_kind", "lex_prefix_core", "lex_prefix_full", "lex_suffix", "lex_canonical_flight", "lex_character_pattern"]


def lexical_features(flights):
    """Only an observed flight Series enters; no timestamps, labels, or other records."""
    rows = []
    for raw in flights:
        text = "" if pd.isna(raw) else re.sub(r"\s+", "", str(raw).upper())
        matched = re.fullmatch(r"([A-Z]*)([0-9]+)([A-Z]*)", text)
        prefix, number, suffix = matched.groups() if matched else ("", "", "")
        core, extra = prefix[:3], prefix[3:]
        repeated = bool(extra) and any(extra == core[-n:] for n in (1, 2, 3))
        kind = "empty" if not text else "unparsed" if not matched else "repeated_suffix" if repeated else "standard" if len(prefix) in (2, 3) else "other_prefix"
        canonical = core + number + suffix if repeated else text
        pattern = re.sub("[A-Z]", "A", re.sub("[0-9]", "9", text))
        rows.append([len(text), len(prefix), len(number), len(suffix), int(matched is not None), int(repeated),
            len(extra) if repeated else 0, int(number.startswith("0")), int(len(suffix)>1 and len(set(suffix))==1),
            kind, core, prefix, suffix, canonical, pattern])
    result = pd.DataFrame(rows, index=flights.index, columns=NUMERIC + CATEGORICAL)
    result[NUMERIC] = result[NUMERIC].astype("float32")
    for column in CATEGORICAL:
        result[column] = prior.token(result[column]).astype(object)
    return result


def decode_flight_tokens(series):
    values = []
    for item in series:
        if item == "m:":
            values.append(None)
            continue
        match = re.fullmatch(r"s([0-9]+):(.*)", item, flags=re.DOTALL)
        if match is None or len(match[2]) != int(match[1]):
            raise ValueError("Unexpected frozen flight token")
        values.append(match[2])
    return pd.Series(values, index=series.index, dtype="string")


def resource_check():
    peak = getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)
    assert peak < 6 * 1024**3 and psutil.virtual_memory().available > 8 * 1024**3
    return peak


def baseline(fold):
    receipt = read_json(BASE / "verification.json")
    assert receipt["status"] == "passed"
    directory = BASE / "global9" / fold
    record = read_json(directory / "manifest.json")
    assert sha256(directory / "manifest.json") == receipt["folds"][fold]["global9"]["manifest_sha256"]
    assert sha256(directory / "candidate.parquet") == record["outputs"]["candidate.parquet"]
    return pd.read_parquet(directory / "candidate.parquet"), record


def declare():
    sources = [Path(__file__), Path(prior.__file__), ROOT / "review_work/campaign_20260916/common.py"]
    definition = {"arms": ARMS, "params": prior.PARAMS, "threads": 2, "seed": 20260916,
        "features": {"control": "Exactly frozen airport_features plus earlier-month crossfit HistoricalTemplate.",
            "lexical": NUMERIC + CATEGORICAL},
        "parsing": "Whitespace removed and uppercase. Full letters/digits/letters match. Extra alphabetic prefix after first3 matching last1/2/3 letters is repeated_suffix. Canonical is hypothesis only, never aircraft identity.",
        "training": "Original full purged missing-NM fit/tune/refit for F1/F3, unchanged raw targets; same historical_template early stopping and chronology. No clipping, calibration, target-defined routing, or score selection.",
        "evaluation": "candidate100% and fixed25% replacement on missing rows only; all finite proxy rows byte-equal verified global9 baseline; both arms reported.",
        "new_information": "Lexical repetition and canonical spelling are not in frozen airport_features. Morphology was motivated by exposed-score diagnostic then quantified on earlier fit/tune; this remains development evidence.",
        "baseline_verification_sha256": sha256(BASE / "verification.json"),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "resources": "2CPU,6GiB process peak,8GiB available reserve."}
    path = OUT / "protocol.json"
    if path.exists():
        assert read_json(path)["declaration"] == definition
    else:
        OUT.mkdir(parents=True, exist_ok=False)
        write_json(path, {"created_utc": utc_now(), "declaration": definition})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--declare-only", action="store_true")
    args = parser.parse_args()
    declare()
    if args.declare_only:
        print("Declared lexical ablation; no fit launched", flush=True)
        return
    x, meta = prior.load_data()
    extra = lexical_features(decode_flight_tokens(x.flight))
    assert extra.index.equals(x.index) and not set(extra).intersection(x)
    resource_check()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([x, extra], axis=1).reset_index().to_parquet(OUT / "all_missing_features.parquet", index=False)
    prior.common.reference = baseline
    results = {}
    for arm in ARMS:
        prior.OUT = OUT / arm
        prior.OUT.mkdir(exist_ok=False)
        write_json(prior.OUT / "protocol.json", read_json(OUT / "protocol.json"))
        features = x if arm == "control" else pd.concat([x, extra], axis=1)
        for fold in ["F1", "F3"]:
            resource_check()
            prior.run_arm("historical_template", fold, features, meta, 20260916, 2)
            resource_check()
            manifest = prior.OUT / "models" / f"historical_template_{fold}_s20260916" / "manifest.json"
            results[arm + "_" + fold] = {"manifest_sha256": sha256(manifest), "manifest_path": str(manifest.relative_to(ROOT))}
    write_json(OUT / "completion.json", {"status": "complete", "completed_utc": utc_now(),
        "source_sha256": sha256(__file__), "protocol_sha256": sha256(OUT / "protocol.json"),
        "feature_sha256": sha256(OUT / "all_missing_features.parquet"), "models": results,
        "peak_bytes": resource_check()})


if __name__ == "__main__":
    main()
