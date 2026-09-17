"""Independent local-only release contract and inherited V2 integrity check."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "private_runs/tail240_20260916/validation/final_submission_best"
V2 = ROOT / "private_runs/submission_v2"
RAW = ROOT / "data/09-15-2026-18-55-03_files_list"
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "preflight_receipt.json"
    assert not dest.exists(), "Preserve prior receipt"
    protocol, ready = read(V2 / "protocol.json"), read(V2 / "submission_ready.json")
    assert sha(V2 / "protocol.json") == ready["protocol_sha256"]
    raw_hashes = {}
    for name, expected in protocol["raw_hashes"].items():
        actual = sha(RAW / name)
        assert actual == expected, name
        raw_hashes[name] = actual
    source_hashes = {}
    for name, expected in protocol["source_hashes"].items():
        actual = sha(ROOT / "knowledgeable-helicopter-screening" / name)
        assert actual == expected, name
        source_hashes[name] = actual
    assert sha(ROOT / "review_work/final_submission_v2.py") == protocol["runner_sha256"]
    models = {}
    for name, expected in ready["components"].items():
        folder = V2 / "components" / name
        rec = read(folder / "record.json")
        assert sha(folder / "model.cbm") == expected == rec["model_sha256"]
        assert rec["protocol_sha256"] == ready["protocol_sha256"]
        klass = CatBoostClassifier if name.endswith("classifier") else CatBoostRegressor
        model = klass().load_model(str(folder / "model.cbm"))
        trees = 300 if name == "gate" else protocol["final_trees"][name]
        assert model.tree_count_ == trees and model.feature_names_ == rec["feature_names"]
        assert rec["status"] == "complete" and rec["reload_exact"]
        models[name] = dict(model_sha256=expected, trees=trees, rows=rec["rows"])
        del model
    for name, expected in read(V2 / "ranking_inputs.json")["files"].items():
        assert sha(V2 / name) == expected
    predictions = pd.read_parquet(V2 / "ranking_predictions.parquet")
    assert sha(V2 / "ranking_predictions.parquet") == ready["ranking_predictions_sha256"]
    template = pq.read_table(RAW / "submitting.parquet")
    saved = pq.read_table(V2 / "knowledgeable-helicopter_v2.parquet")
    assert sha(V2 / "knowledgeable-helicopter_v2.parquet") == ready["sha256"]
    assert saved.schema.remove_metadata() == template.schema.remove_metadata()
    ref, final = template.to_pandas(), saved.to_pandas()
    assert len(ref) == 344841 and list(ref) == [ID, TARGET]
    assert ref[ID].notna().all() and ref[ID].is_unique and predictions[ID].is_unique
    np.testing.assert_array_equal(final[ID], ref[ID])
    ordered = predictions.set_index(ID).loc[ref[ID], "prediction_sec"].to_numpy(float)
    assert np.isfinite(ordered).all()
    np.testing.assert_array_equal(final[TARGET], np.rint(ordered).astype(np.int32))
    ranking = pq.read_table(RAW / "ranking.parquet", columns=[ID, "PHASE_mvt"]).to_pandas()
    rank_ids = ranking.loc[ranking.PHASE_mvt.eq("DEP"), ID]
    assert len(rank_ids) == 344841 and rank_ids.is_unique
    assert set(rank_ids) == set(ref[ID])
    weights_root = ROOT / "private_runs/breakthrough_20260916/models/context_gate/final_simplex9_v1"
    first, third = (read(weights_root / f"{f}_weights.json") for f in ("F1", "F3"))
    assert first["experts"] == third["experts"]
    values = (192122 * np.array(first["global"]) + 152719 * np.array(third["global"])) / 344841
    original = values.copy()
    values[values < 1e-12] = 0
    values /= values.sum()
    assert np.all(values >= 0) and abs(values.sum() - 1) < 1e-15
    receipt = dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
        verifier_sha256=sha(__file__), raw_hashes=raw_hashes, frozen_source_hashes=source_hashes,
        models=models, v2_ready_sha256=sha(V2 / "submission_ready.json"),
        v2_prediction_sha256=ready["ranking_predictions_sha256"],
        template=dict(rows=len(ref), schema=str(template.schema.remove_metadata()), sha256=sha(RAW / "submitting.parquet"),
                      id_order_sha256=hashlib.sha256(ref[ID].to_numpy(dtype="<f8").tobytes()).hexdigest()),
        ranking_template_ids_exact=True, v2_serialization_exact=True,
        proposed_weights=dict(rule="Competition-season weighted original F1/F3 tune coefficients; declared numerical zeros below 1e-12 then renormalize", 
            sources={f:sha(weights_root / f"{f}_weights.json") for f in ("F1", "F3")},
            original=dict(zip(first["experts"], original.tolist())), final=dict(zip(first["experts"], values.tolist())),
            removed_total_mass=float(original[original < 1e-12].sum())),
        routes=dict(ordinary="finite 0<=proxy<=7200: full-year global mixture with PLE387 replacing PLE225",
            finite_nonordinary="Full-year original225 leaf63 LightGBM prediction, not V2",
            missing="0.25 scaled-normalized + 0.1875 ExtraTrees + 0.5625 V2 raw prediction"),
        network_used=False, fitting_used=False, native_v2_prediction_replay_this_check=False,
        limitations=["V2 model bytes and metadata checked; historical independent native replay receipt inherited", 
            "Final ordinary/missing/Ple models and composed file still require verification",
            "Only historical local upload receipts checked; current quota and unused destination must be confirmed by uploader"],
        upload_identity=dict(endpoint="https://s3.opensky-network.org", bucket="prc-2026-knowledgeable-helicopter",
            last_local_key="knowledgeable-helicopter_v2.parquet", last_verified_modified="2026-09-16 08:08:46+00:00",
            last_local_score=294.626, accepted_pairs=344841, proposed_unused_key="knowledgeable-helicopter_v3.parquet"))
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k:receipt[k] for k in ("status", "template", "proposed_weights", "routes", "limitations")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
