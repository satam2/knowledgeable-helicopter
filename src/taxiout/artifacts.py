import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import psutil

from taxiout.config import ROOT
from taxiout.paths import artifact_path, external_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = external_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*.yaml"), ROOT / "pyproject.toml"]
    if (ROOT / "requirements.lock.txt").exists():
        paths.append(ROOT / "requirements.lock.txt")
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def inference_source_hashes():
    package = Path(__file__).resolve().parent
    paths = [*package.glob("features/*.py"), package / "availability.py", package / "schema.py", package / "models/residual.py", package / "predict.py"]
    return {"src/taxiout/" + p.relative_to(package).as_posix(): sha256(p) for p in sorted(paths)}


def environment():
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None
    memory = psutil.virtual_memory()
    dirty = git("status", "--porcelain")
    return {
        "utc": utc_now(), "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": None if dirty is None else bool(dirty), "python": platform.python_version(),
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "ram_bytes": memory.total, "available_ram_bytes": memory.available,
        "libraries": {n: version(n) for n in ["numpy", "pandas", "pyarrow", "catboost", "scikit-learn", "tzdata"]},
    }


def peak_rss():
    memory = psutil.Process().memory_info()
    return getattr(memory, "peak_wset", memory.rss)


class Run:
    def __init__(self, name, config, input_manifest):
        self.started = time.monotonic()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        self.id = f"{name}-{stamp}-{object_hash(config)[:8]}"
        self.path = artifact_path("models", self.id)
        self.path.mkdir(parents=True)
        self.manifest = {
            "run_id": self.id, "status": "incomplete", "config": config,
            "config_hash": object_hash(config), "input_manifest_hash": object_hash(input_manifest),
            "raw_inputs": input_manifest, "environment": environment(), "source_hashes": source_hashes(),
            "external_sources": [], "label_policy": "raw, departures only; no output clipping",
        }
        write_json(self.path / "manifest.json", self.manifest)

    def complete(self, **evidence):
        self.manifest.update(evidence)
        self.manifest.update(status="complete", runtime_sec=time.monotonic() - self.started, peak_rss_bytes=peak_rss())
        self.manifest["outputs"] = {p.name: sha256(p) for p in self.path.iterdir() if p.is_file() and p.name != "manifest.json"}
        write_json(self.path / "manifest.json", self.manifest)
        return self.manifest
