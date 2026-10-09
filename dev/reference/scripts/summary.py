"""Print case counts per (file, outcome), file sizes and Mantid versions for reference data.

Reads ``tests/reference_data`` by default, or ``build/reference_data`` with ``--build``.
``--size-only`` prints only the total size (used at the end of ``generate``).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from common import BUILD_DATA, COMMITTED_DATA, load_config, load_npz
from sampling import OUTCOMES


def data_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.npz"))


def total_size(root: Path) -> int:
    return sum(p.stat().st_size for p in data_files(root))


def print_size(root: Path, budget_mb: float) -> bool:
    size = total_size(root)
    within = size <= budget_mb * 1e6
    print(
        f"Total size of {len(data_files(root))} files in {root}: {size / 1e6:.2f} MB "
        f"(budget {budget_mb:g} MB){'' if within else '  ** OVER BUDGET **'}"
    )
    return within


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Summarise build/reference_data")
    parser.add_argument("--size-only", action="store_true")
    args = parser.parse_args(argv)

    root = BUILD_DATA if args.build else COMMITTED_DATA
    budget = load_config()["max_total_size_mb"]

    if args.size_only:
        return 0 if print_size(root, budget) else 1

    header = f"{'file':42s} {'cases':>6s} " + " ".join(f"{o:>18s}" for o in OUTCOMES) + "   size   mantid"
    print(header)
    for path in data_files(root):
        arrays, metadata = load_npz(path)
        cases = arrays["cases"]
        counts = Counter(OUTCOMES[c] for c in cases["outcome"]) if "outcome" in cases.dtype.names else {}
        row = " ".join(f"{counts.get(o, 0):18d}" for o in OUTCOMES)
        print(
            f"{str(path.relative_to(root)):42s} {len(cases):6d} {row}  "
            f"{path.stat().st_size / 1e3:5.0f}k  {metadata.get('mantid_version', '?')}"
        )
    print()
    return 0 if print_size(root, budget) else 1


if __name__ == "__main__":
    sys.exit(main())
