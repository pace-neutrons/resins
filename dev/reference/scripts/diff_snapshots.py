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


def compare(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Return (number of NaN-mask differences, max relative difference where both finite)."""
    mask_flips = int(np.sum(np.isnan(a) != np.isnan(b)))
    both = ~np.isnan(a) & ~np.isnan(b)
    if not both.any():
        return mask_flips, 0.0
    scale = np.maximum(np.abs(a[both]), np.abs(b[both]))
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(scale > 0, np.abs(a[both] - b[both]) / scale, 0.0)
    return mask_flips, float(rel.max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", help="First snapshot name, e.g. mantid611")
    parser.add_argument("b", help="Second snapshot name, e.g. mantid615")
    parser.add_argument("--rtol", type=float, default=RTOL)
    args = parser.parse_args(argv)

    arrays = {}
    versions = {}
    for name in (args.a, args.b):
        path = SNAPSHOT_DIR / f"{name}.npz"
        if not path.exists():
            print(f"ERROR: {path} not found; run `pixi run -e {name} snapshot` first", file=sys.stderr)
            return 1
        arrays[name], metadata = load_npz(path)
        versions[name] = metadata["mantid_version"]

    print(f"Comparing {args.a} (Mantid {versions[args.a]}) with {args.b} (Mantid {versions[args.b]})")
    common = sorted(set(arrays[args.a]) & set(arrays[args.b]))
    only = sorted(set(arrays[args.a]) ^ set(arrays[args.b]))
    for stem in only:
        print(f"  note: {stem} is only in one snapshot")

    differing = []
    for stem in common:
        flips, max_rel = compare(arrays[args.a][stem], arrays[args.b][stem])
        same = flips == 0 and max_rel <= args.rtol
        print(f"  {stem:32s} {'identical' if same else 'DIFFERS':10s} max rel diff {max_rel:.2e}, outcome flips {flips}")
        if not same:
            differing.append(stem_instrument(stem))

    if differing:
        print(f"FAIL: output differs between Mantid versions for: {', '.join(sorted(set(differing)))}")
        return 1
    print("OK: no differences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
