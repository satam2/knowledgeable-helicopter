"""Rebuild the frozen recipe in an isolated source tree with initially empty caches."""
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from taxiout.artifacts import read_json, sha256, write_json
from taxiout.config import ROOT


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    destination = ROOT / "data/interim" / ("clean-reproduction-" + stamp)
    destination.mkdir(parents=True, exist_ok=False)
    files = [ROOT / name for name in ["pyproject.toml", "README.md", "LICENSE", "requirements.lock.txt"]]
    files += [p for folder in ["src", "configs", "tests"] for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".yaml"}]
    files += [ROOT / "reports" / name for name in ["input_manifest.json", "data_audit.json", "freeze.json", "confirmation.json"]]
    files += [ROOT / "submissions/local_candidate.parquet"]
    for file in files:
        target = destination / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    raw = destination / "data/raw/prc"
    raw.mkdir(parents=True)
    for file in (ROOT / "data/raw/prc").glob("*.parquet"):
        (raw / file.name).hardlink_to(file)
    python = ROOT / ".venv-verify/Scripts/python.exe"
    commands = [
        [str(python), "-m", "pip", "check"],
        [str(python), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "--force-reinstall", str(destination)],
        [str(python), "-m", "pytest"],
        [str(python), "-m", "taxiout.cli", "reproduce", "--config", "configs/release.yaml", "--offline"],
    ]
    record = {"isolated_tree": destination.relative_to(ROOT).as_posix(), "initial_audit_cache_present": False,
              "initial_feature_cache_present": False, "raw_files": "local hardlinks; raw pack hashes revalidated",
              "lock_sha256": sha256(ROOT / "requirements.lock.txt"), "commands": commands, "status": "incomplete"}
    write_json(ROOT / "reports/clean_verification.json", record)
    for index, command in enumerate(commands):
        print("CLEAN STEP", index + 1, " ".join(command), flush=True)
        with (destination / f"step-{index + 1}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=destination, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            record.update(failed_step=index + 1, exit_code=result.returncode)
            write_json(ROOT / "reports/clean_verification.json", record)
            print((destination / f"step-{index + 1}.log").read_text(encoding="utf-8"), flush=True)
            return result.returncode
    reconstruction = read_json(destination / "reports/reproduction.json")
    record.update(status="complete", reproduction=reconstruction,
                  logs={f"step-{i}.log": sha256(destination / f"step-{i}.log") for i in range(1, 5)})
    write_json(ROOT / "reports/clean_verification.json", record)
    write_json(ROOT / "reports/reproduction.json", reconstruction)
    print("CLEAN REPRODUCTION COMPLETE", reconstruction, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
