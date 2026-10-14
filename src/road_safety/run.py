"""Run the reproducible road-safety analysis pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = [
    ("Structured crash data quality", "road_safety.crash_data", []),
    ("MoRTH statistical baseline", "road_safety.statistics", []),
    ("Spatial concentration analysis", "road_safety.spatial", []),
    ("2022 narrative NLP", "road_safety.nlp", []),
    ("Conditional fatality-band model", "road_safety.crash_models", []),
]


def run_pipeline(*, stats_reps: int, with_vision: bool) -> int:
    for label, module, extra_args in PIPELINE:
        command = [sys.executable, "-m", module, *extra_args]
        if module == "road_safety.statistics":
            command.extend(["--reps", str(stats_reps)])
        print(f"\n=== {label} ===", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            print(f"Pipeline stopped: {module} exited with {completed.returncode}.", file=sys.stderr)
            return completed.returncode

    if with_vision:
        # RDD is a separate optional benchmark because it requires the local
        # image archive and the optional vision dependencies.
        # The CLI default is the documented compact 1-epoch capped run; use
        # --full-run for the full India training/validation split.
        vision_steps = [
            ("Validate local India VOC annotations", "profile"),
            ("Create sequence-proxy grouped split", "split"),
        ]
        for label, subcommand in vision_steps:
            command = [sys.executable, "-m", "road_safety.vision", subcommand]
            print(f"\n=== {label} ===", flush=True)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if completed.returncode:
                print(f"Vision step '{subcommand}' exited with {completed.returncode}.", file=sys.stderr)
                return completed.returncode
        train_command = [
            sys.executable, "-m", "road_safety.vision", "train",
            "--learning-rate", "0.00005",
            "--max-train-images", "1200",
            "--max-val-images", "300",
            "--seed", "42",
        ]
        print("\n=== Train/evaluate compact Faster R-CNN ===", flush=True)
        completed = subprocess.run(train_command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode:
            print(f"Vision step 'train' exited with {completed.returncode}.", file=sys.stderr)
            return completed.returncode
    else:
        print(
            "\nVision training was not requested. To include it, install `uv sync --extra vision`, "
            "obtain the India RDD2022 subset, then rerun with `--with-vision`.",
            flush=True,
        )

    print("\nPipeline completed. Review reports/ and data/processed/ before interpreting results.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the local India road-safety analyses in dependency order. "
            "The public crash dataset and the independent RDD image benchmark remain separate."
        )
    )
    parser.add_argument("--stats-reps", type=int, default=10_000,
                        help="Bootstrap/permutation repetitions for official aggregate analysis")
    parser.add_argument("--with-vision", action="store_true",
                        help="Also train/evaluate the optional RDD2022 India image model")
    args = parser.parse_args()
    if args.stats_reps < 100:
        parser.error("--stats-reps must be at least 100")
    raise SystemExit(run_pipeline(stats_reps=args.stats_reps, with_vision=args.with_vision))


if __name__ == "__main__":
    main()
