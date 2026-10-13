"""Generate PyChop resolution references (Mantid's bundled ``pychop``).

Run inside a Mantid pixi environment; only entries of ``config.yaml`` whose ``environment``
matches ``--env`` are generated. Never imports resins. Target test points are read from
``cases.yaml``; outcomes are assigned from PyChop and the restrictions declared there.

Output goes to ``build/reference_data/pychop/<INSTRUMENT>_<model_version>.npz`` and contains
only the public resolution FWHM at 40 energy-transfer points -- no internal component arrays
(moderator, chopper, detector, sample widths or chop times).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import cases
import mantid  # noqa: F401  (puts Mantid's bundled pychop on sys.path)
import numpy as np
from cases import Outcome
from common import (
    BUILD_DATA,
    PYCHOP_CASE_DTYPE,
    load_config,
    make_metadata,
    pychop_file_stem,
    save_npz,
)
from pychop.Instruments import Instrument

MAX_FREQUENCIES = 2


class PyChopEvaluator:
    """Thin wrapper over a PyChop Instrument for one chopper configuration.

    Mantid up to 6.11 has a bug: ``Instrument`` binds its child properties (such as
    ``frequency``) on the *class*, to the most recently created instance, and ``getVanVar``
    reads ``self.frequency``. With several instruments alive, the chopper frequency of the wrong
    one is then used. So only the most recently used evaluator keeps its Instrument; any other
    evaluator builds a fresh one on first use (which rebinds the properties), and methods must
    set the frequencies before evaluating.
    """

    _active: PyChopEvaluator | None = None

    def __init__(self, pychop_name: str, chopper: str, n_points: int):
        self.pychop_name = pychop_name
        self.chopper = chopper
        self.n_points = n_points
        self.instrument = None
        self._activate()
        self.is_fermi = bool(self.instrument.isFermi)

    def _activate(self) -> None:
        if PyChopEvaluator._active is not self:
            self.instrument = Instrument(self.pychop_name, chopper=self.chopper)
            self.chopper_system = self.instrument.chopper_system
            PyChopEvaluator._active = self

    def set_frequencies(self, frequencies) -> None:
        """Raises ValueError if PyChop rejects the frequencies."""
        self._activate()
        self.instrument.setFrequency([float(f) for f in frequencies])

    def energy_transfers(self, e_init: float) -> np.ndarray:
        return np.linspace(0, e_init, self.n_points, endpoint=False)

    def resolution(self, e_init: float, energy_transfers=None) -> np.ndarray:
        self._activate()
        if energy_transfers is None:
            energy_transfers = self.energy_transfers(e_init)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                self.instrument.getResolution(
                    Etrans=list(energy_transfers), Ei_in=e_init
                ),
                dtype=float,
            )

    def ei_accepted(self, e_init: float) -> bool:
        """Whether PyChop's ``setEi`` accepts the energy (a GUI-level check, not used by
        ``getResolution``)."""
        self._activate()
        try:
            self.chopper_system.setEi(e_init)
        except ValueError:
            return False
        return True


def classify_cases(
    entry: dict,
    config_specs: dict[str, cases.ConfigurationSpec],
    evaluators: dict[str, PyChopEvaluator],
) -> tuple[list[tuple], Counter]:
    """Assign an outcome to every sampled case.

    Returns rows ``(case, outcome, reason)`` and a counter of PyChop/resins disagreements.
    """
    freq_settings = entry["frequency_settings"]
    rows = []
    notes: Counter = Counter()

    for name, spec in config_specs.items():
        evaluator = evaluators[name]
        for case in spec.expand():
            rejected = spec.check_restrictions(case.e_init, case.frequencies)
            if rejected is not None:
                outcome, setting = rejected
                rows.append((case, outcome, f"resins:{setting}"))
                continue

            try:
                evaluator.set_frequencies(case.frequencies)
            except ValueError:
                notes["pychop rejects a frequency that resins allows"] += 1
                print(
                    f"WARNING {pychop_file_stem(entry)} {name}: PyChop rejects frequencies "
                    f"{case.frequencies} allowed by resins",
                    file=sys.stderr,
                )
                rows.append(
                    (
                        case,
                        Outcome.REJECTED_FREQUENCY,
                        f"pychop:{freq_settings[0]}",
                    )
                )
                continue

            if evaluator.is_fermi and not evaluator.ei_accepted(case.e_init):
                notes["setEi limits narrower than resins' e_init restriction"] += 1

            resolution = evaluator.resolution(case.e_init)
            outcome = (
                Outcome.NO_TRANSMISSION
                if np.any(np.isnan(resolution))
                else Outcome.VALUE
            )
            rows.append((case, outcome, ""))
    return rows, notes


def build_arrays(
    entry: dict,
    evaluators: dict[str, PyChopEvaluator],
    rows: list[tuple],
    n_points: int,
) -> dict[str, np.ndarray]:
    n_cases = len(rows)
    cases_arr = np.zeros(n_cases, dtype=PYCHOP_CASE_DTYPE)
    resolution = np.full((n_cases, n_points), np.nan)

    for i, (case, outcome, reason) in enumerate(rows):
        frequencies = list(case.frequencies) + [0] * (
            MAX_FREQUENCIES - len(case.frequencies)
        )
        cases_arr[i] = (
            case.configuration,
            case.e_init,
            frequencies[0],
            frequencies[1],
            outcome,
            reason,
        )
        if outcome not in (Outcome.VALUE, Outcome.NO_TRANSMISSION):
            continue
        evaluator = evaluators[case.configuration]
        evaluator.set_frequencies(case.frequencies)
        resolution[i] = evaluator.resolution(case.e_init)

    return {"cases": cases_arr, "resolution": resolution}


def generate_entry(entry: dict, config: dict, all_cases: dict, out_dir: Path) -> Path:
    n_points = config["sampling"]["n_resolution_points"]
    freq_settings = entry["frequency_settings"]
    config_map = entry.get("configuration_map", {})

    # Look up the cases.yaml entry for this instrument+model
    key = f"{entry['instrument']}_{entry['model_version']}"
    cases_entry = all_cases.get(key)
    if cases_entry is None:
        print(f"WARNING: no cases.yaml entry for {key}", file=sys.stderr)
        return out_dir  # will be handled by caller

    # Parse specifications per configuration
    instrument_spec = cases.InstrumentSpec.from_dict(cases_entry)
    config_specs = instrument_spec.parse_configurations()

    # Build evaluators for each configuration
    evaluators = {
        name: PyChopEvaluator(
            entry["pychop_name"], config_map.get(name, name), n_points
        )
        for name in config_specs
    }

    rows, notes = classify_cases(entry, config_specs, evaluators)
    arrays = build_arrays(entry, evaluators, rows, n_points)

    metadata = make_metadata(
        kind="pychop",
        instrument=entry["instrument"],
        model_version=entry["model_version"],
        pychop_name=entry["pychop_name"],
        pychop_source=str(Path(sys.modules[Instrument.__module__].__file__).parent),
        environment=entry["environment"],
        frequency_settings=freq_settings,
        configuration_map=config_map,
        n_resolution_points=n_points,
        resolution_grid="linspace(0, e_init, n_resolution_points, endpoint=False)",
        notes=dict(notes),
    )

    path = out_dir / f"pychop/{pychop_file_stem(entry)}.npz"
    save_npz(path, arrays, metadata)

    counts = Counter(Outcome.from_index(c) for c in arrays["cases"]["outcome"])
    summary = ", ".join(f"{outcome}={counts.get(outcome, 0)}" for outcome in Outcome)
    print(
        f"{path.name}: {len(rows)} cases ({summary}); {path.stat().st_size / 1e3:.0f} kB"
    )
    for note, count in notes.items():
        print(f"  note: {count} cases where {note}")
    return path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env", required=True, help="Mantid environment name, e.g. mantid615"
    )
    parser.add_argument("--out", type=Path, default=BUILD_DATA, help="Output directory")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Restrict to these file stems, e.g. MARI_PyChop_fit_v1",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()

    config = load_config()
    all_cases = cases.load_cases()
    entries = [entry for entry in config["pychop"] if entry["environment"] == args.env]
    if args.only:
        entries = [entry for entry in entries if pychop_file_stem(entry) in args.only]
    if not entries:
        print(f"No PyChop entries for environment {args.env}")
        return

    for entry in entries:
        generate_entry(entry, config, all_cases, args.out)


if __name__ == "__main__":
    main()
