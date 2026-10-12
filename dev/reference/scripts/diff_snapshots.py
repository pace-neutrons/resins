"""Compare two snapshots (see ``snapshot.py``) and name the instruments whose output differs.

Exit status is non-zero if any instrument present in both snapshots differs.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from common import SNAPSHOT_DIR, load_npz

RTOL = 1e-8


def stem_instrument(stem: str) -> str:
    return stem.split("_", 1)[0]


def compare_arrays(first: np.ndarray, second: np.ndarray) -> tuple[int, float]:
    """Return (number of NaN-mask differences, max relative difference where both finite)."""
    mask_flips = int(np.sum(np.isnan(first) != np.isnan(second)))
    both = ~np.isnan(first) & ~np.isnan(second)
    if not both.any():
        return mask_flips, 0.0
    scale = np.maximum(np.abs(first[both]), np.abs(second[both]))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(scale > 0, np.abs(first[both] - second[both]) / scale, 0.0)
    return mask_flips, float(rel.max())


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("first_snapshot", help="First snapshot name, e.g. mantid611")
    parser.add_argument("second_snapshot", help="Second snapshot name, e.g. mantid615")
    parser.add_argument(
        "--rtol", type=float, default=RTOL, help="Relative tolerance for comparison"
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    arrays = {}
    versions = {}
    for name in (args.first_snapshot, args.second_snapshot):
        path = SNAPSHOT_DIR / f"{name}.npz"
        if not path.exists():
            print(
                f"ERROR: {path} not found; run `pixi run -e {name} snapshot` first",
                file=sys.stderr,
            )
            sys.exit(1)
        arrays[name], metadata = load_npz(path)
        versions[name] = metadata["mantid_version"]

    print(
        f"Comparing {args.first_snapshot} (Mantid {versions[args.first_snapshot]}) with "
        f"{args.second_snapshot} (Mantid {versions[args.second_snapshot]})"
    )
    common = sorted(
        set(arrays[args.first_snapshot]) & set(arrays[args.second_snapshot])
    )
    only = sorted(set(arrays[args.first_snapshot]) ^ set(arrays[args.second_snapshot]))
    for stem in only:
        print(f"  note: {stem} is only in one snapshot")

    differing = []
    for stem in common:
        flips, max_rel = compare_arrays(
            arrays[args.first_snapshot][stem], arrays[args.second_snapshot][stem]
        )
        same = flips == 0 and max_rel <= args.rtol
        print(
            f"  {stem:32s} {'identical' if same else 'DIFFERS':10s} max rel diff {max_rel:.2e}, outcome flips {flips}"
        )
        if not same:
            differing.append(stem_instrument(stem))

    if differing:
        print(
            f"FAIL: output differs between Mantid versions for: {', '.join(sorted(set(differing)))}"
        )
        sys.exit(1)
    print("OK: no differences")


if __name__ == "__main__":
    main()
