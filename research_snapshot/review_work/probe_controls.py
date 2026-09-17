"""Synthetic, isolated demonstrations of review findings; never train on PRC data."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from taxiout import train, release
from taxiout.artifacts import inference_source_hashes, object_hash
from taxiout.cli import parser
from taxiout.config import load_config

OUT = Path(__file__).resolve().parent


class TrainingReached(Exception):
    pass


results = {}
config = load_config("configs/release.yaml")
inputs = {"files": []}
with tempfile.TemporaryDirectory(prefix="control-probe-", dir=OUT) as folder:
    root = Path(folder)
    model_path = root / "models" / "old-run"
    model_path.mkdir(parents=True)
    stale = {"status": "complete", "fold": "F1", "config_hash": object_hash(config),
             "input_manifest_hash": object_hash(inputs), "source_hashes": {"src/taxiout/train.py": "obsolete"},
             "split": {"split_hash": "obsolete"}}
    (model_path / "manifest.json").write_text(json.dumps(stale), encoding="utf-8")
    with patch.object(train, "ROOT", root):
        found = train.find_run(config, "F1", inputs)
    results["run_reuse_ignores_changed_source_and_split"] = found is not None

    cli_args = parser().parse_args(["confirm", "--fold", "F1", "--config", "configs/release.yaml"])
    results["cli_accepts_confirm_F1"] = cli_args.fold == "F1"
    with patch.object(train, "verified_manifest", return_value=inputs), \
         patch.object(train, "read_json", return_value={"config_hash": object_hash(config)}), \
         patch.object(train, "find_run", return_value=None), \
         patch.object(train, "Run", return_value=SimpleNamespace(manifest={}, path=root, id="synthetic")), \
         patch.object(train, "write_json"), \
         patch.object(train, "load_training", side_effect=TrainingReached):
        try:
            train.evaluate_candidate(config, "F1", confirm=True)
        except TrainingReached:
            results["confirm_F1_reaches_training_without_rejection"] = True

    (root / "data/interim/audit").mkdir(parents=True)
    (root / "data/interim/audit/departures.parquet").touch()
    with patch.object(release, "ROOT", root), \
         patch.object(release, "verified_manifest", return_value=inputs), \
         patch.object(train, "train_final", return_value={"run_id": "synthetic"}), \
         patch.object(release, "predict_candidate", return_value=None), \
         patch.object(release, "build_submission", return_value={"passed": True, "sha256": "not-the-published-hash"}), \
         patch.object(release, "write_json"):
        result = release.reproduce(config)
    results["reproduce_completes_without_published_hash_comparison_when_reference_absent"] = result["status"] == "complete"

protected = inference_source_hashes()
results["inference_fingerprint_omissions"] = [p for p in ["src/taxiout/predict.py", "src/taxiout/cache.py", "src/taxiout/io.py"] if p not in protected]
assert all(value is True for key,value in results.items() if key != "inference_fingerprint_omissions")
(OUT / "control_probes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
