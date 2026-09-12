import argparse
import atexit
import ctypes
import json
import sys

from taxiout.artifacts import read_json
from taxiout.config import ROOT, load_config


def parser():
    result = argparse.ArgumentParser(description="PRC 2026 local taxi-out pipeline")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["audit", "make-splits", "benchmark", "anytime", "evaluate", "confirm", "train-final", "reproduce", "freeze"]:
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/base.yaml" if name != "make-splits" else "configs/folds.yaml")
        if name in {"evaluate", "confirm"}:
            command.add_argument("--fold", required=True, choices=["F1", "F2", "F3", "G1", "C1"])
        if name == "reproduce":
            command.add_argument("--offline", action="store_true", required=True)
        if name == "freeze":
            command.add_argument("--runs", nargs="+", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--reference", default="direct")
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--input", default="data/raw/prc/ranking.parquet")
    predict.add_argument("--output", default="data/processed/ranking_predictions.parquet")
    build = commands.add_parser("build-submission")
    build.add_argument("--bundle", required=True)
    build.add_argument("--template", default="data/raw/prc/submitting.parquet")
    build.add_argument("--output", default="submissions/local_candidate.parquet")
    validate = commands.add_parser("validate-submission")
    validate.add_argument("--file", required=True)
    validate.add_argument("--template", default="data/raw/prc/submitting.parquet")
    return result


def main():
    args = parser().parse_args()
    if sys.platform == "win32":
        previous = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        if previous:
            atexit.register(ctypes.windll.kernel32.SetThreadExecutionState, previous)
    try:
        config = load_config(args.config) if hasattr(args, "config") else None
        if args.command == "audit":
            from taxiout.audit import audit
            result = audit(config)
        elif args.command == "make-splits":
            from taxiout.splits import make_splits
            result = make_splits(args.config)
        elif args.command in {"benchmark", "anytime"}:
            from taxiout.benchmark import anytime, benchmark
            result = (benchmark if args.command == "benchmark" else anytime)(config)
        elif args.command in {"evaluate", "confirm"}:
            from taxiout.train import evaluate_candidate
            result = evaluate_candidate(config, args.fold, confirm=args.command == "confirm")
        elif args.command == "train-final":
            from taxiout.train import train_final
            result = train_final(config)
        elif args.command == "predict":
            from taxiout.predict import predict_candidate
            result = {"rows": len(predict_candidate(args.bundle, args.input, args.output)), "output": args.output}
        elif args.command == "build-submission":
            from taxiout.predict import predict_candidate
            from taxiout.submission import build_submission
            bundle_config = read_json(ROOT / args.bundle / "bundle.json")["config"]
            result = build_submission(ROOT / args.template, predict_candidate(args.bundle, bundle_config["raw_dir"] + "/ranking.parquet"), ROOT / args.output)
        elif args.command == "validate-submission":
            from taxiout.submission import validate_submission
            result = validate_submission(ROOT / args.file, ROOT / args.template)
        elif args.command in {"compare", "freeze", "reproduce"}:
            from taxiout.release import compare, freeze, reproduce
            if args.command == "compare":
                result = compare(args.runs, reference=args.reference)
            elif args.command == "freeze":
                result = freeze(config, args.runs)
            else:
                result = reproduce(config)
        else:
            raise ValueError("Unknown command")
        summary = {k: result[k] for k in ["run_id", "status", "training_rows", "ranking_departures", "template_rows", "rows", "sha256", "passed", "output"] if k in result}
        print(json.dumps(summary or result, indent=2, default=str))
    except (ValueError, FileNotFoundError, RuntimeError, MemoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
