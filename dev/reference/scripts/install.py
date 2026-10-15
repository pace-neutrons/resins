"""Copy regenerated reference data (``build/reference_data``) into ``tests/reference_data``.

Files in the destination that were not regenerated are removed, so that the destination ends up
holding exactly the regenerated set.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import BUILD_DATA, COMMITTED_DATA
from summary import data_files


def install(source: Path, destination: Path) -> tuple[list[Path], list[Path]]:
    """Make ``destination`` hold exactly the ``.npz`` files of ``source``.

    Returns the relative paths of copied and of removed files.
    """
    new_files = {path.relative_to(source): path for path in data_files(source)}
    if not new_files:
        raise FileNotFoundError(
            f"No reference data in {source}; run `pixi run generate` first"
        )

    removed = []
    for path in data_files(destination):
        rel = path.relative_to(destination)
        if rel not in new_files:
            path.unlink()
            removed.append(rel)

    copied = []
    for rel, path in sorted(new_files.items()):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(rel)
    return copied, removed


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=BUILD_DATA,
        help="Source directory with regenerated data",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=COMMITTED_DATA,
        help="Destination directory for reference data",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    try:
        copied, removed = install(args.build, args.destination)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    for rel in copied:
        print(f"installed {rel}")
    for rel in removed:
        print(f"removed stale {rel}")
    print(f"{len(copied)} files installed into {args.destination}")


if __name__ == "__main__":
    main()
