"""Real disjoint development/evaluation classifier-prediction logs replayed through the declared finite thermal/energy model; 300s phases and seed13."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "thermaldebt", "calibrate", "--predictions", args.development, "--output", str(output / "profile.json")],
        ["-m", "thermaldebt", "prepare", "--predictions", args.evaluation, "--output", str(output / "data"), "--phase-duration", "300", "--seed", "13"],
        ["-m", "thermaldebt", "run", "--data", str(output / "data"), "--profile", str(output / "profile.json"), "--policy", "full", "--output", str(output / "full")],
        ["-m", "thermaldebt", "run", "--data", str(output / "data"), "--profile", str(output / "profile.json"), "--policy", "energy_mpc", "--output", str(output / "energy_mpc")],
        ["-m", "thermaldebt", "run", "--data", str(output / "data"), "--profile", str(output / "profile.json"), "--policy", "fixed_low", "--output", str(output / "fixed_low")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--development', required=True, type=absolute_path)
    parser.add_argument('--evaluation', required=True, type=absolute_path)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
