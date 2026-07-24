#!/usr/bin/env python3
"""Summarise completed NCDE UEA drop runs.

This script is specialised for the moved NCDE drop outputs living under
``outputs_ncde_drop_uea/<model>``.

The NCDE training code does not write a ``completed.txt`` marker. Instead this
script treats a run directory as completed when the standard saved output files
exist:

- ``steps.npy``
- ``all_train_metric.npy``
- ``all_val_metric.npy``
- ``all_time.npy``
- ``test_metric.npy`` or ``test_acc.npy``
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DROP_RE = re.compile(r"_drop([0-9.]+)")
SEED_RE = re.compile(r"_seed(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Root directory to search. May be passed multiple times. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--model",
        default="ncde",
        help="Model output directory to search under outputs/. Default: ncde",
    )
    parser.add_argument(
        "--show-runs",
        action="store_true",
        help="Print every completed run before the grouped summary.",
    )
    return parser.parse_args()


def default_roots(cli_roots: list[Path]) -> list[Path]:
    if cli_roots:
        return cli_roots
    return [Path.cwd()]


def run_output_files_exist(run_dir: Path) -> bool:
    required = (
        "steps.npy",
        "all_train_metric.npy",
        "all_val_metric.npy",
        "all_time.npy",
    )
    metric_files = ("test_metric.npy", "test_acc.npy")
    return all((run_dir / name).is_file() for name in required) and any(
        (run_dir / name).is_file() for name in metric_files
    )


def find_completed_runs(roots: list[Path], model_name: str) -> list[Path]:
    runs: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        model_root = root / "outputs_ncde_drop_uea" / model_name
        if not model_root.exists():
            continue
        for run_dir in model_root.glob("*/*"):
            if not run_dir.is_dir() or not run_output_files_exist(run_dir):
                continue
            resolved = run_dir.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(run_dir)

    return sorted(runs)


def read_metric(run_dir: Path) -> float:
    for filename in ("test_metric.npy", "test_acc.npy"):
        metric_path = run_dir / filename
        if metric_path.exists():
            return float(np.load(metric_path))
    raise FileNotFoundError(f"No test metric file found in {run_dir}")


def extract_drop(run_name: str) -> float:
    match = DROP_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse drop percentage from {run_name}")
    return float(match.group(1))


def extract_seed(run_name: str) -> str:
    match = SEED_RE.search(run_name)
    if match is None:
        return "?"
    return match.group(1)


def main() -> None:
    args = parse_args()
    roots = default_roots(args.root)
    completed_runs = find_completed_runs(roots, args.model)

    if not completed_runs:
        print("No completed runs found.")
        return

    grouped: dict[tuple[float, str], list[float]] = defaultdict(list)
    per_run_rows: list[tuple[float, str, str, float, Path]] = []

    for run_dir in completed_runs:
        dataset = run_dir.parent.name
        run_name = run_dir.name
        drop = extract_drop(run_name)
        seed = extract_seed(run_name)
        metric = read_metric(run_dir)
        grouped[(drop, dataset)].append(metric)
        per_run_rows.append((drop, dataset, seed, metric, run_dir))

    print(f"Found {len(completed_runs)} completed {args.model} runs.")
    print("Search roots:")
    for root in roots:
        print(f"  {root}")

    if args.show_runs:
        print("\nPer-run results")
        for drop, dataset, seed, metric, run_dir in sorted(
            per_run_rows, key=lambda row: (row[0], row[1], row[2])
        ):
            print(
                f"drop={drop:>4.2f}  dataset={dataset:22s}  "
                f"seed={seed:>4s}  test={metric:.4f}  path={run_dir}"
            )

    print("\nGrouped by drop and dataset")
    macro_by_drop: dict[float, list[float]] = defaultdict(list)
    for (drop, dataset), values in sorted(grouped.items(), key=lambda item: item[0]):
        arr = np.asarray(values, dtype=float)
        macro_by_drop[drop].append(float(arr.mean()))
        print(
            f"drop={drop:>4.2f}  dataset={dataset:22s}  "
            f"n={len(arr)}  mean={arr.mean():.4f}  std={arr.std(ddof=0):.4f}"
        )

    print("\nMacro average by drop")
    for drop, means in sorted(macro_by_drop.items()):
        arr = np.asarray(means, dtype=float)
        print(
            f"drop={drop:>4.2f}  datasets={len(arr)}  "
            f"macro_mean={arr.mean():.4f}  macro_std={arr.std(ddof=0):.4f}"
        )


if __name__ == "__main__":
    main()
