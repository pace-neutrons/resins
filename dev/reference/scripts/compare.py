"""Compare regenerated reference data (``build/``) with the committed data.

Per file: added/removed cases, outcome flips, and the maximum deviation of every value array
over the cases present in both. Also checks the total size budget. Exits non-zero if anything
exceeds the tolerance, unless ``--report-only`` is given.

Cases are matched on all case-table columns except ``outcome`` and ``reason``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from cases import Outcome
from common import BUILD_DATA, COMMITTED_DATA, load_config, load_npz
from summary import data_files, print_size

RTOL = 1e-7
ATOL = 1e-12  # absolute tolerance floor for values that are exactly zero or tiny

NOT_KEY_COLUMNS = {"outcome", "reason"}


def case_keys(table: np.ndarray, exclude=NOT_KEY_COLUMNS) -> list[tuple]:
    names = [name for name in table.dtype.names if name not in exclude]
    return [
        tuple(
            None if (isinstance(val, float) and np.isnan(val)) else val
            for val in (row[name].item() for name in names)
        )
        for row in table
    ]


def max_deviation(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Max relative deviation and max absolute deviation over entries finite in both.

    Entries NaN in only one array are counted by the caller as ``nan_mismatches``.
    """
    both_valid = ~(np.isnan(first) | np.isnan(second))
    if not both_valid.any():
        return 0.0, 0.0
    a, b = first[both_valid], second[both_valid]
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
    return float(rel.max()), float(diff.max())


def compare_aligned(
    name: str,
    new: np.ndarray,
    old: np.ndarray,
    new_keys: list,
    old_keys: list,
    rtol: float,
    report: list[str],
) -> bool:
    """Compare two arrays row-wise after matching rows on their keys."""
    old_index = {k: i for i, k in enumerate(old_keys)}
    pairs = [(i, old_index[k]) for i, k in enumerate(new_keys) if k in old_index]
    if not pairs:
        return True
    new_rows = new[[p[0] for p in pairs]]
    old_rows = old[[p[1] for p in pairs]]
    nan_mismatch = int(np.sum(np.isnan(new_rows) != np.isnan(old_rows)))
    rel, absolute = max_deviation(new_rows, old_rows)
    ok = nan_mismatch == 0 and (rel <= rtol or absolute <= ATOL)
    report.append(
        f"    {name:32s} max rel dev {rel:.2e}  max abs dev {absolute:.2e}  NaN mismatches {nan_mismatch}"
        f"{'' if ok else '  ** EXCEEDS TOLERANCE **'}"
    )
    return ok


def compare_file(new_path: Path, old_path: Path, rtol: float) -> tuple[bool, list[str]]:
    new, new_meta = load_npz(new_path)
    old, old_meta = load_npz(old_path)
    report: list[str] = []
    ok = True

    new_keys, old_keys = case_keys(new["cases"]), case_keys(old["cases"])
    added = set(new_keys) - set(old_keys)
    removed = set(old_keys) - set(new_keys)
    if added or removed:
        ok = False
        report.append(f"    cases added: {len(added)}, removed: {len(removed)}")
        for key in sorted(added, key=str)[:5]:
            report.append(f"      + {key}")
        for key in sorted(removed, key=str)[:5]:
            report.append(f"      - {key}")

    old_index = {k: i for i, k in enumerate(old_keys)}
    flips = [
        (
            k,
            Outcome.from_index(old["cases"]["outcome"][old_index[k]]),
            Outcome.from_index(new["cases"]["outcome"][i]),
        )
        for i, k in enumerate(new_keys)
        if k in old_index
        and old["cases"]["outcome"][old_index[k]] != new["cases"]["outcome"][i]
    ]
    if flips:
        ok = False
        report.append(f"    outcome flips: {len(flips)}")
        for key, was, now in flips[:5]:
            report.append(f"      {key}: {was} -> {now}")

    for name in new:
        if name == "cases" or name not in old:
            continue
        if new[name].shape[1:] != old[name].shape[1:]:
            ok = False
            report.append(
                f"    {name}: shape changed {old[name].shape} -> {new[name].shape}"
            )
            continue
        if new[name].dtype.kind != "f":
            continue
        ok &= compare_aligned(
            name, new[name], old[name], new_keys, old_keys, rtol, report
        )

    for field in ("mantid_version", "model_versions", "n_resolution_points"):
        if new_meta.get(field) != old_meta.get(field):
            report.append(
                f"    metadata {field}: {old_meta.get(field)} -> {new_meta.get(field)}"
            )
    return ok, report


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build", type=Path, default=BUILD_DATA, help="Regenerated data directory"
    )
    parser.add_argument(
        "--committed",
        type=Path,
        default=COMMITTED_DATA,
        help="Committed data directory",
    )
    parser.add_argument(
        "--rtol", type=float, default=RTOL, help="Relative tolerance for comparisons"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Print the report but always exit with status 0",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    new_files = {path.relative_to(args.build): path for path in data_files(args.build)}
    old_files = {
        path.relative_to(args.committed): path for path in data_files(args.committed)
    }
    ok = True

    if not new_files:
        print(
            f"ERROR: no regenerated data in {args.build}; run `pixi run generate`",
            file=sys.stderr,
        )
        sys.exit(1)

    for rel in sorted(set(old_files) - set(new_files)):
        print(f"{rel}: committed but not regenerated")
        ok = False
    for rel in sorted(set(new_files) - set(old_files)):
        print(f"{rel}: new file (not committed)")
        ok = False

    for rel in sorted(set(new_files) & set(old_files)):
        file_ok, report = compare_file(new_files[rel], old_files[rel], args.rtol)
        print(f"{rel}: {'OK' if file_ok else 'DIFFERS'}")
        for line in report:
            print(line)
        ok &= file_ok

    ok &= print_size(args.build, load_config()["max_total_size_mb"])

    print("PASS" if ok else "FAIL: regenerated data differs from committed data")
    if not (ok or args.report_only):
        sys.exit(1)


if __name__ == "__main__":
    main()
