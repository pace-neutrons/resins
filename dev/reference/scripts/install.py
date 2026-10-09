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
    new_files = {p.relative_to(source): p for p in data_files(source)}
    if not new_files:
        raise FileNotFoundError(f"No reference data in {source}; run `pixi run generate` first")

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD_DATA)
    parser.add_argument("--destination", type=Path, default=COMMITTED_DATA)
    args = parser.parse_args(argv)

    try:
        copied, removed = install(args.build, args.destination)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    for rel in copied:
        print(f"installed {rel}")
    for rel in removed:
        print(f"removed stale {rel}")
    print(f"{len(copied)} files installed into {args.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
