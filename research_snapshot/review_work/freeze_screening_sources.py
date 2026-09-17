"""Archive only whitelisted code, configs and runner scripts for reproducibility."""

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "knowledgeable-helicopter-screening"
OUT = ROOT / "private_runs/screening_230"
protocol = json.loads((OUT / "protocol.json").read_text())
files = {"code/" + name: REPO / name for name in protocol["source_hashes"]}
files.update({"code/" + p.relative_to(REPO).as_posix(): p for p in (REPO / "tests").glob("*.py")})
files["code/docs/SCREENING.md"] = REPO / "docs/SCREENING.md"
runner_names = ["run_screening.py", "run_refinement.py", "run_transfer.py", "run_rome_seed.py",
                "combine_screened.py", "compose_best_routes.py", "ensemble_rome.py", "test_transfer.py",
                "summarize_screening.py", "summarize_followups.py", "analyze_remaining_error.py",
                "analyze_clock_regimes.py", "probe_schedule_precision.py", "export_experiment_table.py",
                "build_screening_report.py", "verify_screening_artifacts.py", "freeze_screening_sources.py",
                "decide_development_reference.py", "run_gpu_residual.py"]
files.update({"runners/" + name: ROOT / "review_work" / name for name in runner_names})
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
archive = OUT / "screening_source_snapshot.zip"
with ZipFile(archive, "w", ZIP_DEFLATED) as z:
    for name, path in files.items():
        assert path.suffix in {".py", ".yaml", ".toml", ".txt", ".md"}
        z.write(path, name)
    z.writestr("source_hashes.json", json.dumps(hashes, indent=2))
with ZipFile(archive) as z:
    assert all(hashlib.sha256(z.read(name)).hexdigest() == digest for name, digest in hashes.items())
    assert not any(name.endswith((".parquet", ".cbm")) for name in z.namelist())
(OUT / "source_snapshot_manifest.json").write_text(json.dumps({"archive": str(archive), "files": len(hashes),
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "contents": "Whitelisted source only; no raw data or models"}, indent=2), encoding="utf-8")
print(f"Saved {len(hashes)} source/config files: {archive}")
