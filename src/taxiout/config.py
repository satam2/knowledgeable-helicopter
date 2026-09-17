from pathlib import Path

import yaml

def repository_root():
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("Run taxiout from the repository or a directory beneath it")


ROOT = repository_root()


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "extends" in config:
        config = merge(load_config(path.parent / config.pop("extends")), config)
    if "candidate" in config:
        if config["threads"] < 1 or config["iterations"] < 1:
            raise ValueError("threads and iterations must be positive")
        if config["coverage"] not in {"month-isolated", "continuous-audit-only"}:
            raise ValueError("Unrecognized observation coverage policy")
        if config["rounding"] != "nearest-even-int32":
            raise ValueError("Unsupported output policy")
    return config
