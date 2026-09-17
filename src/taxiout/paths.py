"""Keep confidential inputs and every generated artifact outside Git checkouts."""

import os
from pathlib import Path


def external_path(path):
    path = Path(path).expanduser().resolve()
    for parent in (path, *path.parents):
        marker = parent / ".git"
        if marker.is_file() or (marker / "HEAD").is_file():
            raise ValueError(f"Private path must be outside every Git repository: {path}")
    return path


def raw_root(config):
    value = os.environ.get("TAXIOUT_RAW_DIR") or config["raw_dir"]
    if not Path(value).is_absolute():
        from taxiout.config import ROOT
        value = ROOT / value
    return external_path(value)


def artifact_path(*parts):
    value = os.environ.get("TAXIOUT_ARTIFACT_ROOT")
    if not value or not Path(value).is_absolute():
        raise ValueError("Set TAXIOUT_ARTIFACT_ROOT to an absolute directory outside the repository")
    root = external_path(value)
    path = external_path(root.joinpath(*parts))
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes TAXIOUT_ARTIFACT_ROOT")
    raw = os.environ.get("TAXIOUT_RAW_DIR")
    if raw:
        raw = external_path(raw)
        if root.is_relative_to(raw) or raw.is_relative_to(root):
            raise ValueError("Raw data and artifacts must use disjoint directories")
    return path
