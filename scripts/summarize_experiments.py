"""Index every attempted configuration, including incomplete/failed work."""
from taxiout.artifacts import read_json, write_json
from taxiout.config import ROOT


def main():
    registry = {}
    for path in sorted((ROOT / "models").glob("*/manifest.json")):
        run = read_json(path)
        key = run["config_hash"]
        entry = registry.setdefault(key, {"config": run["config"], "runs": []})
        item = {"run_id": run["run_id"], "status": run["status"], "fold": run.get("fold"),
                "training_rows": run.get("training_rows"), "runtime_sec": run.get("runtime_sec"),
                "input_manifest_hash": run["input_manifest_hash"]}
        if run["status"] != "complete":
            item["note"] = "Incomplete artifact, not eligible for scoring or release. See search_protocol.md for timeout/interruption context."
        entry["runs"].append(item)
    write_json(ROOT / "reports/configuration_registry.json", registry)


if __name__ == "__main__":
    main()
