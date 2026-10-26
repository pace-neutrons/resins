"""Generate Abins references: PyChop-based, PANTHER, TOSCA and Lagrange resolution.

Run in the Mantid environment named in ``config.yaml`` (``abins.environment``). Never imports
resins. Energies are given to Abins in meV and converted to cm^-1 with
``abins.constants.MILLI_EV_TO_WAVENUMBER``; results are converted back with the same constant
and stored in meV. Output goes to ``build/reference_data/abins/<name>.npz``.

The PyChop-based cases are read from ``cases.yaml`` (same points as the PyChop reference
generator), so the Abins generator does not import resins or parse its YAML restrictions.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# mantid must be imported first to populate sys.path with Mantid's bundled scripts
import mantid  # noqa: F401

# isort: split

import abins.parameters
import cases
import numpy as np
from abins.constants import MILLI_EV_TO_WAVENUMBER
from abins.instruments.lagrangeinstrument import LagrangeInstrument
from abins.instruments.panther import PantherInstrument
from abins.instruments.pychop import PyChopInstrument
from abins.instruments.toscainstrument import ToscaInstrument
from cases import Outcome
from common import BUILD_DATA, load_config, make_metadata, save_npz
from generate_pychop import PyChopEvaluator, classify_cases

ABINS_PYCHOP_DTYPE = np.dtype(
    [
        ("instrument", "U16"),
        ("configuration", "U32"),
        ("e_init", "f8"),
        ("f0", "i2"),
        ("outcome", "i1"),
        ("reason", "U64"),
    ]
)
PANTHER_DTYPE = np.dtype([("e_init", "f8"), ("outcome", "i1"), ("reason", "U64")])
INDIRECT_DTYPE = np.dtype(
    [
        ("configuration", "U32"),
        ("energy_transfer", "f8"),
        ("outcome", "i1"),
        ("reason", "U64"),
    ]
)

VALUE = int(Outcome.VALUE)
NO_TRANSMISSION = int(Outcome.NO_TRANSMISSION)


def to_meV(wavenumber):
    return wavenumber / MILLI_EV_TO_WAVENUMBER


def to_wavenumber(meV):
    return meV * MILLI_EV_TO_WAVENUMBER


def arange_from(spec: dict) -> np.ndarray:
    return np.arange(spec["start"], spec["stop"], spec["step"], dtype=float)


def abins_pychop_sigma(
    abins: PyChopInstrument, e_init: float, frequency: float, n_points: int
) -> np.ndarray:
    """Sigma in meV of the Abins PyChop polynomial at ``n_points`` energy transfers 0..e_init."""
    abins.set_incident_energy(e_init, "meV")
    abins._chopper_frequency = frequency
    abins._pychop_instrument.chopper_system.setFrequency([frequency])
    abins._polyfits = {}
    energies = np.linspace(0, e_init, n_points)
    return to_meV(abins.calculate_sigma(to_wavenumber(energies)))


def generate_pychop(config: dict, all_cases: dict, out_dir: Path) -> Path:
    n_points = config["sampling"]["n_abins_points"]
    n_resolution_points = config["sampling"]["n_resolution_points"]
    pychop_entries = {
        (entry["instrument"], entry["model_version"]): entry
        for entry in config["pychop"]
    }

    rows = []
    notes: Counter = Counter()
    for item in config["abins"]["pychop"]:
        entry = pychop_entries[(item["instrument"], item["model_version"])]
        config_map = entry.get("configuration_map", {})

        # Load and parse cases from cases.yaml
        key = f"{entry['instrument']}_{entry['model_version']}"
        if key not in all_cases:
            print(f"WARNING: no cases.yaml entry for {key}", file=sys.stderr)
            continue

        instrument_spec = cases.InstrumentSpec.from_dict(all_cases[key])
        all_specs = instrument_spec.parse_configurations()

        # Abins knows only some of the choppers PyChop does (e.g. no MAPS "B").
        supported = abins.parameters.instruments[entry["pychop_name"]]["settings"]

        # Filter to configurations that Abins supports
        config_specs = {
            name: spec
            for name, spec in all_specs.items()
            if config_map.get(name, name) in supported
        }

        evaluators = {
            name: PyChopEvaluator(
                entry["pychop_name"], config_map.get(name, name), n_resolution_points
            )
            for name in config_specs
        }

        classified, entry_notes = classify_cases(entry, config_specs, evaluators)
        notes.update(entry_notes)

        abins_instruments = {
            name: PyChopInstrument(
                name=entry["pychop_name"], setting=config_map.get(name, name)
            )
            for name in config_specs
        }
        for case, outcome, reason in classified:
            # Abins applies no restrictions of its own; only cases PyChop evaluates are compared.
            if outcome not in (Outcome.VALUE, Outcome.NO_TRANSMISSION):
                continue
            sigma = np.full(n_points, np.nan)
            if outcome == Outcome.VALUE:
                sigma = abins_pychop_sigma(
                    abins_instruments[case.configuration],
                    case.e_init,
                    case.frequencies[0],
                    n_points,
                )
            rows.append((entry["instrument"], case, outcome, reason, sigma))

    cases_arr = np.zeros(len(rows), dtype=ABINS_PYCHOP_DTYPE)
    sigma = np.full((len(rows), n_points), np.nan)
    for i, (instrument, case, outcome, reason, row_sigma) in enumerate(rows):
        cases_arr[i] = (
            instrument,
            case.configuration,
            case.e_init,
            case.frequencies[0],
            outcome,
            reason,
        )
        sigma[i] = row_sigma

    metadata = make_metadata(
        kind="abins_pychop",
        abins_class="abins.instruments.pychop.PyChopInstrument",
        n_points=n_points,
        energy_grid="linspace(0, e_init, n_points) in meV",
        sigma="meV, Abins polynomial fit (order 4, 40 points) to PyChop resolution",
        model_versions={
            item["instrument"]: item["model_version"]
            for item in config["abins"]["pychop"]
        },
        notes=dict(notes),
    )
    return write(
        out_dir / "abins/pychop.npz", {"cases": cases_arr, "sigma": sigma}, metadata
    )


def generate_panther(config: dict, out_dir: Path) -> Path:
    n_points = config["abins"]["n_panther_points"]
    energies = [float(energy) for energy in config["abins"]["panther_e_init"]]
    cases_arr = np.zeros(len(energies), dtype=PANTHER_DTYPE)
    sigma = np.zeros((len(energies), n_points))
    instrument = PantherInstrument()
    for i, e_init in enumerate(energies):
        instrument.set_incident_energy(e_init, "meV")
        grid = np.linspace(0, e_init, n_points)
        sigma[i] = to_meV(instrument.calculate_sigma(to_wavenumber(grid)))
        cases_arr[i] = (e_init, VALUE, "")

    metadata = make_metadata(
        kind="abins_panther",
        abins_class="abins.instruments.panther.PantherInstrument",
        n_points=n_points,
        energy_grid="linspace(0, e_init, n_points) in meV",
        sigma="meV",
    )
    return write(
        out_dir / "abins/panther.npz", {"cases": cases_arr, "sigma": sigma}, metadata
    )


def generate_tosca(config: dict, out_dir: Path) -> Path:
    energies = arange_from(config["abins"]["tosca_energy_transfer"])
    cases_arr = np.zeros(len(energies), dtype=INDIRECT_DTYPE)
    for i, energy in enumerate(energies):
        cases_arr[i] = ("", energy, VALUE, "")
    sigma = to_meV(ToscaInstrument.get_sigma(to_wavenumber(energies)))

    metadata = make_metadata(
        kind="abins_tosca",
        abins_class="abins.instruments.toscainstrument.ToscaInstrument",
        sigma="meV",
    )
    return write(
        out_dir / "abins/tosca.npz", {"cases": cases_arr, "sigma": sigma}, metadata
    )


def generate_lagrange(config: dict, out_dir: Path) -> Path:
    spec = config["abins"]["lagrange_energy_transfer"]
    energies = np.unique(
        np.concatenate([arange_from(spec["coarse"]), arange_from(spec["dense"])])
    )
    rows = []
    sigma = []
    for monochromator in config["abins"]["lagrange_monochromators"]:
        instrument = LagrangeInstrument(setting=f"{monochromator} (Lagrange)")
        values = to_meV(instrument.get_sigma(to_wavenumber(energies)))
        for energy, value in zip(energies, values):
            rows.append((monochromator, energy, VALUE, ""))
            sigma.append(value)

    metadata = make_metadata(
        kind="abins_lagrange",
        abins_class="abins.instruments.lagrangeinstrument.LagrangeInstrument",
        sigma="meV",
    )
    return write(
        out_dir / "abins/lagrange.npz",
        {"cases": np.array(rows, dtype=INDIRECT_DTYPE), "sigma": np.array(sigma)},
        metadata,
    )


def write(path: Path, arrays: dict, metadata: dict) -> Path:
    save_npz(path, arrays, metadata)
    cases_arr = arrays["cases"]
    counts = Counter(Outcome.from_index(c) for c in cases_arr["outcome"])
    summary = ", ".join(f"{outcome}={counts[outcome]}" for outcome in Outcome)
    print(
        f"abins/{path.name}: {len(cases_arr)} cases ({summary}); {path.stat().st_size / 1e3:.0f} kB"
    )
    return path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=BUILD_DATA, help="Output directory")
    return parser


def main() -> None:
    args = get_parser().parse_args()

    config = load_config()
    all_cases = cases.load_cases()
    generate_pychop(config, all_cases, args.out)
    generate_panther(config, args.out)
    generate_tosca(config, args.out)
    generate_lagrange(config, args.out)


if __name__ == "__main__":
    main()
