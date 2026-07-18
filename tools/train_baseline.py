"""Run the four baseline recommenders with the standard three random seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import monotonic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "main.py"
BASELINE_MODELS = ("LightGCN", "SimGCL", "NCL", "SGL")
BASELINE_SEEDS = (42, 0, 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run LightGCN, SimGCL, NCL and SGL with seeds 42, 0 and 1."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the dataset for every model (otherwise use each model config).",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Override the GPU identifier for every model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training_epochs for every model.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when a model fails instead of trying the remaining models.",
    )
    return parser.parse_args(argv)


def build_command(model, args):
    command = [
        sys.executable,
        str(MAIN_FILE),
        "--model",
        model,
        "--seeds",
        *(str(seed) for seed in BASELINE_SEEDS),
    ]
    if args.dataset is not None:
        command.extend(("--dataset", args.dataset))
    if args.gpu_id is not None:
        command.extend(("--gpu_id", str(args.gpu_id)))
    if args.epochs is not None:
        command.extend(("--epochs", str(args.epochs)))
    return command


def main(argv=None, run_command=subprocess.run):
    args = parse_args(argv)
    started_at = monotonic()
    failed_models = []

    print("Baseline models: {}".format(", ".join(BASELINE_MODELS)), flush=True)
    print("Seeds per model: {}".format(", ".join(map(str, BASELINE_SEEDS))), flush=True)
    for index, model in enumerate(BASELINE_MODELS, start=1):
        command = build_command(model, args)
        print(flush=True)
        print("[{}/{}] Starting {}".format(index, len(BASELINE_MODELS), model), flush=True)
        result = run_command(command, cwd=str(PROJECT_ROOT))
        if result.returncode == 0:
            print("[{}/{}] Completed {}".format(index, len(BASELINE_MODELS), model), flush=True)
            continue

        failed_models.append(model)
        print(
            "[{}/{}] Failed {} (exit code {})".format(
                index, len(BASELINE_MODELS), model, result.returncode
            ),
            file=sys.stderr,
            flush=True,
        )
        if args.stop_on_error:
            break

    elapsed = monotonic() - started_at
    print(flush=True)
    print("Total baseline time: {:.3f} seconds".format(elapsed), flush=True)
    if failed_models:
        print("Failed models: {}".format(", ".join(failed_models)), file=sys.stderr)
        return 1
    print("All baseline models completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())