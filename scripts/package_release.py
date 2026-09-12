"""Build local source/artifact packages without raw challenge data or external writes."""
import json
import zipfile
from pathlib import Path

from taxiout.artifacts import object_hash, read_json, sha256, source_hashes, utc_now, write_json
from taxiout.config import ROOT
from taxiout.submission import validate_submission


def main():
    release = read_json(ROOT / "reports/release_manifest.json")
    run = ROOT / "models" / release["run_id"]
    validation = validate_submission(ROOT / "submissions/local_candidate.parquet", ROOT / "data/raw/prc/submitting.parquet")
    if validation["sha256"] != release["submission"]["sha256"]:
        raise ValueError("Release submission checksum mismatch")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    source_paths = [ROOT / name for name in ["README.md", "LICENSE", "THIRD_PARTY.md", "pyproject.toml", "requirements.lock.txt", ".gitignore", ".gitattributes"]]
    source_paths += [p for folder in ["src", "configs", "tests", "scripts", "reports", "docs"] for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.name != "package_validation.json" and p.suffix in {".py", ".yaml", ".md", ".json", ".csv", ".png", ".pdf"} and "__pycache__" not in p.parts and "runs" not in p.parts]
    source = dist / "prc-2026-source.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(set(source_paths)):
            archive.write(file, file.relative_to(ROOT).as_posix())
    private = dist / "prc-2026-local-artifacts.zip"
    artifacts = [*run.glob("*.cbm"), run / "bundle.json", run / "manifest.json",
                 ROOT / "submissions/local_candidate.parquet", ROOT / "submissions/local_candidate.validation.json"]
    with zipfile.ZipFile(private, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in artifacts:
            archive.write(file, file.relative_to(ROOT).as_posix())
    for package in [source, private]:
        with zipfile.ZipFile(package) as archive:
            if archive.testzip() is not None:
                raise ValueError("Release archive CRC validation failed")
            if any("data/raw/" in name for name in archive.namelist()):
                raise ValueError("Raw data must not enter release archives")
    record = {"created_utc": utc_now(), "public_source": {"file": source.relative_to(ROOT).as_posix(), "sha256": sha256(source)},
              "private_local_artifacts": {"file": private.relative_to(ROOT).as_posix(), "sha256": sha256(private)},
              "source_members": len(set(source_paths)), "archive_crc_passed": True, "raw_data_included": False, "uploaded": False,
              "note": "Model and prediction redistribution terms unverified; local artifacts are not included in public source."}
    write_json(ROOT / "reports/package_validation.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
