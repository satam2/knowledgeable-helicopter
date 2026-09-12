"""Sequential, resumable bounded development; each model gets a fresh process."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from taxiout.artifacts import object_hash, read_json, utc_now, write_json
from taxiout.config import ROOT, load_config
from taxiout.io import verified_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["configs/direct.yaml", "configs/residual.yaml"])
    parser.add_argument("--folds", nargs="+", default=["F1", "F2", "F3"])
    args = parser.parse_args()
    history = []
    for path in args.configs:
        config = load_config(path)
        inputs = verified_manifest(config)
        for fold in args.folds:
            matching = []
            for marker in (ROOT / "models").glob("*/manifest.json"):
                record = read_json(marker)
                if record.get("status") == "complete" and record.get("fold") == fold and record.get("config_hash") == object_hash(config) and record.get("input_manifest_hash") == object_hash(inputs):
                    matching.append(record["run_id"])
            if matching:
                history.append({"candidate": config["candidate"], "fold": fold, "reused_run": sorted(matching)[-1]})
                print("REUSE", history[-1], flush=True)
                continue
            command = [sys.executable, "-m", "taxiout.cli", "evaluate", "--config", path, "--fold", fold]
            record = {"started_utc": utc_now(), "candidate": config["candidate"], "fold": fold, "command": command}
            write_json(ROOT / "reports/active_experiment.json", record)
            status = ROOT / "reports/project_status.md"
            best = "No complete seasonal comparison yet."
            comparison_path = ROOT / "reports/comparison.json"
            if comparison_path.exists():
                candidates = read_json(comparison_path)["candidates"]
                scored = [(name, value["season_score"]) for name, value in candidates.items() if "season_score" in value]
                if scored:
                    name, score = min(scored, key=lambda item: item[1])
                    best = f"Best completed development comparison: {name}, season score {score:.3f}s; remaining gates pending."
            status.write_text("# Project Status\n\nCurrent phase: 3/4, measured development.\n\n"
                              "Phases 0-2: audit and tests passed; diagnostic and anytime artifacts recorded.\n\n"
                              f"Active: {config['candidate']} {fold}.\n\nResume: `" + " ".join(command) + "`\n\n"
                              "Completed experiments: see reports/experiments.csv and models/*/manifest.json.\n\n"
                              + best + "\n\n"
                              "External submission/publication: pending, outside current local scope.\n", encoding="utf-8")
            print("START", config["candidate"], fold, flush=True)
            result = subprocess.run(command, cwd=ROOT)
            record.update(exit_code=result.returncode, finished_utc=utc_now())
            history.append(record)
            write_json(ROOT / "reports/development_progress.json", history)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
