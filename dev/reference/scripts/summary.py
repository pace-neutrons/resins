"""Print case counts per (file, outcome), file sizes and Mantid versions for reference data.

Reads ``tests/reference_data`` by default, or ``build/reference_data`` with ``--build``.
``--size-only`` prints only the total size (used at the end of ``generate``).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from cases import Outcome
from common import BUILD_DATA, COMMITTED_DATA, load_config, load_npz


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


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build", action="store_true", help="Summarise build/reference_data"
    )
    parser.add_argument(
        "--size-only", action="store_true", help="Only check and print total size"
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    root = BUILD_DATA if args.build else COMMITTED_DATA
    budget = load_config()["max_total_size_mb"]

    if args.size_only:
        if not print_size(root, budget):
            sys.exit(1)
        return

    header = (
        f"{'file':42s} {'cases':>6s} "
        + " ".join(f"{outcome:>18s}" for outcome in Outcome)
        + "   size   mantid"
    )
    print(header)
    for path in data_files(root):
        arrays, metadata = load_npz(path)
        cases_arr = arrays["cases"]
        counts = (
            Counter(Outcome.from_index(code) for code in cases_arr["outcome"])
            if "outcome" in cases_arr.dtype.names
            else {}
        )
        row = " ".join(f"{counts.get(outcome, 0):18d}" for outcome in Outcome)
        print(
            f"{path.relative_to(root)!s:42s} {len(cases_arr):6d} {row}  "
            f"{path.stat().st_size / 1e3:5.0f}k  {metadata.get('mantid_version', '?')}"
        )
    print()
    if not print_size(root, budget):
        sys.exit(1)


if __name__ == "__main__":
    main()
