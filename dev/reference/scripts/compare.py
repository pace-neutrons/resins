"""Compare regenerated reference data (``build/``) with the committed data.

For each file, reports:
- added/removed cases
- changed outcomes
- maximum deviations between value arrays
- total size of files.

Exits non-zero if anything exceeds the tolerance.

Cases are identified by their input parameters (e_init, chopper frequencies,
configuration), excluding the outcome and reason result columns.
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
MAX_REPORTED_CASES = 5

NOT_KEY_COLUMNS = frozenset({"outcome", "reason"})


def case_keys(
    table: np.ndarray, exclude: frozenset[str] = NOT_KEY_COLUMNS
) -> list[tuple[object, ...]]:
    """Extract a hashable key tuple for each row in the structured cases table.

    The key consists of all independent parameter columns (such as
    ``configuration``, ``e_init``, ``f0``, ``f1``), excluding dependent outcome
    and diagnostic columns (``outcome`` and ``reason``).

    Parameters
    ----------
    table
        1D NumPy structured cases array from a reference data file.
    exclude
        Column names to exclude from the case identity key.

    Returns
    -------
    list[tuple[object, ...]]
        List of parameter tuples identifying each case, for example:
        ``[('ARCS-100-1.5-AST', 20.0, 300, 0), ...]``.
    """
    key_columns = [name for name in table.dtype.names if name not in exclude]
    return table[key_columns].tolist()


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
    new_keys: list[tuple[object, ...]],
    old_keys: list[tuple[object, ...]],
    rtol: float,
    report: list[str],
) -> bool:
    """Compare two value arrays row-wise after aligning on case parameter keys.

    Case keys are parameter tuples produced by :func:`case_keys`, for example
    ``('ARCS-100-1.5-AST', 20.0, 300, 0)``. Each row in ``new`` corresponds
    to the key at the same index in ``new_keys``; likewise for ``old`` and ``old_keys``.
    Only cases present in both are compared.
    """
    if new_keys == old_keys:
        new_rows, old_rows = new, old
    else:
        new_index = {key: i for i, key in enumerate(new_keys)}
        old_indices = [i for i, key in enumerate(old_keys) if key in new_index]
        if not old_indices:
            report.append(f"    {name:32s} ** NO COMMON CASES TO COMPARE **")
            return False
        new_indices = [new_index[old_keys[i]] for i in old_indices]
        new_rows, old_rows = new[new_indices], old[old_indices]

    nan_mismatches = int(np.sum(np.isnan(new_rows) != np.isnan(old_rows)))
    rel, absolute = max_deviation(new_rows, old_rows)
    ok = not nan_mismatches and (rel <= rtol or absolute <= ATOL)
    report.append(
        f"    {name:32s} max rel dev {rel:.2e}  max abs dev {absolute:.2e}  NaN mismatches {nan_mismatches}"
        f"{'' if ok else '  ** EXCEEDS TOLERANCE **'}"
    )
    return ok


def _report_added_removed(
    new_keys: list[tuple[object, ...]],
    old_keys: list[tuple[object, ...]],
    report: list[str],
) -> bool:
    """Report added and removed cases. Returns True if case sets match."""
    added = set(new_keys) - set(old_keys)
    removed = set(old_keys) - set(new_keys)
    if not (added or removed):
        return True

    report.append(f"    cases added: {len(added)}, removed: {len(removed)}")
    for key in sorted(added, key=str)[:MAX_REPORTED_CASES]:
        report.append(f"      + {key}")
    if len(added) > MAX_REPORTED_CASES:
        report.append("      ...")

    for key in sorted(removed, key=str)[:MAX_REPORTED_CASES]:
        report.append(f"      - {key}")
    if len(removed) > MAX_REPORTED_CASES:
        report.append("      ...")

    return False


def _report_outcome_flips(
    new_cases: np.ndarray,
    old_cases: np.ndarray,
    new_keys: list[tuple[object, ...]],
    old_keys: list[tuple[object, ...]],
    report: list[str],
) -> bool:
    """Report outcome changes for shared cases. Returns True if all outcomes match."""
    old_index = {k: i for i, k in enumerate(old_keys)}
    flips = [
        (
            k,
            Outcome.from_index(old_cases["outcome"][old_index[k]]),
            Outcome.from_index(new_cases["outcome"][new_i]),
        )
        for new_i, k in enumerate(new_keys)
        if k in old_index
        and old_cases["outcome"][old_index[k]] != new_cases["outcome"][new_i]
    ]
    if not flips:
        return True

    report.append(f"    outcome flips: {len(flips)}")
    for key, was, now in flips[:MAX_REPORTED_CASES]:
        report.append(f"      {key}: {was} -> {now}")
    if len(flips) > MAX_REPORTED_CASES:
        report.append("      ...")

    return False


def _report_metadata_changes(
    new_meta: dict, old_meta: dict, report: list[str]
) -> None:
    """Report changes in key metadata fields (informational)."""
    for field in ("mantid_version", "model_versions", "n_resolution_points"):
        if new_meta.get(field) != old_meta.get(field):
            report.append(
                f"    metadata {field}: {old_meta.get(field)} -> {new_meta.get(field)}"
            )


def compare_file(new_path: Path, old_path: Path, rtol: float) -> tuple[bool, list[str]]:
    """Compare one pair of .npz reference files, returning (ok, report_lines)."""
    new, new_meta = load_npz(new_path)
    old, old_meta = load_npz(old_path)
    new_keys, old_keys = case_keys(new["cases"]), case_keys(old["cases"])

    report: list[str] = []
    ok = True

    ok &= _report_added_removed(new_keys, old_keys, report)
    ok &= _report_outcome_flips(new["cases"], old["cases"], new_keys, old_keys, report)

    added_arrays = set(new) - set(old)
    removed_arrays = set(old) - set(new)
    if added_arrays or removed_arrays:
        ok = False
        for name in sorted(added_arrays):
            report.append(f"    array added: {name}")
        for name in sorted(removed_arrays):
            report.append(f"    array removed: {name}")

    for name in sorted(set(new) & set(old)):
        if name != "cases" and new[name].dtype.kind == "f":
            if new[name].shape[1:] != old[name].shape[1:]:
                ok = False
                report.append(
                    f"    {name}: shape changed {old[name].shape} -> {new[name].shape}"
                )
            else:
                ok &= compare_aligned(
                    name, new[name], old[name], new_keys, old_keys, rtol, report
                )

    _report_metadata_changes(new_meta, old_meta, report)
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
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
