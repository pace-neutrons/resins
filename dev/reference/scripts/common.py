"""Shared helpers for the reference data scripts (paths, config, npz IO, metadata)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from sampling import OUTCOMES, REPO_ROOT

REFERENCE_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = REFERENCE_DIR / "build"
BUILD_DATA = BUILD_DIR / "reference_data"
SNAPSHOT_DIR = BUILD_DIR / "snapshots"
COMMITTED_DATA = REPO_ROOT / "tests" / "reference_data"

GENERATOR_VERSION = 1

OUTCOME_CODE = {name: i for i, name in enumerate(OUTCOMES)}

PYCHOP_CASE_DTYPE = np.dtype(
    [
        ("configuration", "U32"),
        ("e_init", "f8"),
        ("f0", "f8"),
        ("f1", "f8"),
        ("outcome", "i1"),
        ("reason", "U64"),
    ]
)


def load_config() -> dict:
    with open(REFERENCE_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def pychop_file_stem(entry: dict) -> str:
    return f"{entry['instrument']}_{entry['model_version']}"


def mantid_version() -> str:
    import mantid

    return mantid.__version__


def resins_commit() -> str:
    """Commit of the resins checkout whose instrument YAML was read, or "unknown"."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_metadata(**extra) -> dict:
    metadata = {
        "generator_version": GENERATOR_VERSION,
        "mantid_version": mantid_version(),
        "resins_commit": resins_commit(),
        "units": {
            "energy": "meV",
            "frequency": "Hz",
            "resolution": "FWHM in meV (PyChop getResolution)",
            "width_sq": "time FWHM squared in s^2 (PyChop convention)",
        },
        "outcomes": list(OUTCOMES),
    }
    metadata.update(extra)
    return metadata


def save_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata, sort_keys=True)), **arrays)


def load_npz(path: Path) -> tuple[dict, dict]:
    """Load arrays and parsed metadata from a reference file (no pickles)."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata
