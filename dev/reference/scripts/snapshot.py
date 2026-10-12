"""Evaluate PyChop in the current Mantid environment on the committed reference cases.

The snapshot holds the PyChop resolution (NaN where there is no transmission) for every
evaluated case of the selected instruments, so that two Mantid environments can be compared with
``diff_snapshots.py``. By default only instruments with a single resins model version are
snapshotted: their data come from one Mantid version, so a change in another one means upstream
changed the instrument. Use ``--instruments`` to snapshot others deliberately.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import cases
import numpy as np
from common import (
    BUILD_DATA,
    COMMITTED_DATA,
    SNAPSHOT_DIR,
    load_config,
    load_npz,
    make_metadata,
    pychop_file_stem,
    save_npz,
)
from generate_pychop import PyChopEvaluator

EVALUATED = (cases.OUTCOME_VALUE, cases.OUTCOME_NO_TRANSMISSION)


def default_instruments(config: dict) -> list[str]:
    counts = Counter(entry["instrument"] for entry in config["pychop"])
    return sorted(name for name, n in counts.items() if n == 1)


def snapshot_entry(entry: dict, path: Path, n_points: int) -> np.ndarray:
    arrays, _metadata = load_npz(path)
    cases = arrays["cases"]
    config_map = entry.get("configuration_map", {})

    evaluators: dict[str, PyChopEvaluator] = {}
    resolution = np.full((len(cases), n_points), np.nan)
    for i, case in enumerate(cases):
        if cases.OUTCOMES[case["outcome"]] not in EVALUATED:
            continue
        name = str(case["configuration"])
        if name not in evaluators:
            evaluators[name] = PyChopEvaluator(
                entry["pychop_name"], config_map.get(name, name), n_points
            )
        evaluator = evaluators[name]
        frequencies = [freq for freq in (case["f0"], case["f1"]) if not np.isnan(freq)]
        evaluator.set_frequencies(frequencies)
        resolution[i] = evaluator.resolution(float(case["e_init"]))
    return resolution


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("PIXI_ENVIRONMENT_NAME", "default"),
        help="Snapshot name",
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        help="Instruments to snapshot (default: single-version ones)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Use cases from build/ instead of tests/reference_data",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    config = load_config()
    n_points = config["sampling"]["n_resolution_points"]
    root = BUILD_DATA if args.build else COMMITTED_DATA
    instruments = (
        [inst.upper() for inst in args.instruments]
        if args.instruments
        else default_instruments(config)
    )

    snapshot: dict[str, np.ndarray] = {}
    for entry in config["pychop"]:
        if entry["instrument"] not in instruments:
            continue
        stem = pychop_file_stem(entry)
        path = root / f"pychop/{stem}.npz"
        if not path.exists():
            print(
                f"ERROR: no reference file {path}; run `pixi run update` first",
                file=sys.stderr,
            )
            sys.exit(1)
        snapshot[stem] = snapshot_entry(entry, path, n_points)
        print(f"{stem}: {len(snapshot[stem])} cases")

    out = SNAPSHOT_DIR / f"{args.name}.npz"
    save_npz(
        out,
        snapshot,
        make_metadata(kind="snapshot", name=args.name, instruments=instruments),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
