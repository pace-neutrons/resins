"""Generate PyChop component and resolution references (Mantid's bundled ``pychop``).

Run inside a Mantid pixi environment; only entries of ``config.yaml`` whose ``environment``
matches ``--env`` are generated. Never imports resins. Output goes to
``build/reference_data/pychop/<INSTRUMENT>_<model_version>.npz``.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import mantid  # noqa: F401  (puts Mantid's bundled pychop on sys.path)
import numpy as np
from pychop.Instruments import Instrument

import sampling
from common import (
    BUILD_DATA,
    OUTCOME_CODE,
    PYCHOP_CASE_DTYPE,
    load_config,
    make_metadata,
    pychop_file_stem,
    save_npz,
)

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
                self.instrument.getResolution(Etrans=list(energy_transfers), Ei_in=e_init), dtype=float
            )

    def transmits(self, e_init: float, frequencies) -> bool:
        self.set_frequencies(frequencies)
        return not np.any(np.isnan(self.resolution(e_init, [0.0])))

    def ei_accepted(self, e_init: float) -> bool:
        """Whether PyChop's ``setEi`` accepts the energy (a GUI-level check, not used by
        ``getResolution``)."""
        self._activate()
        try:
            self.chopper_system.setEi(e_init)
        except ValueError:
            return False
        return True

    def chopper_width_squared(self, e_init: float) -> tuple[float, float]:
        self._activate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first, second = self.chopper_system.getWidthSquared(e_init)
        return float(np.squeeze(first)), (np.nan if second is None else float(second))

    def chop_times(self, e_init: float) -> np.ndarray:
        """Opening windows of the first and the last chopper, as an array (2, windows, 2)."""
        self._activate()
        from pychop.MulpyRep import calcChopTimes

        cs = self.chopper_system
        args = [e_init, cs._long_frequency, cs._instpar, cs.phase]
        if hasattr(cs, "phaseOffset"):
            args.append(cs.phaseOffset)
        windows = calcChopTimes(*args)[1]
        return [np.array(windows[0], dtype=float), np.array(windows[-1], dtype=float)]

    def moderator_width_squared(self, e_init: float) -> tuple[float, float]:
        self._activate()
        moderator = self.instrument.moderator
        return float(moderator.getWidthSquared(e_init)), float(moderator.getAnalyticWidthsSquared(e_init))

    def detector_width_squared(self, e_init: float) -> np.ndarray:
        self._activate()
        # LET has a Detector object without a detector type: no contribution.
        detector = getattr(self.instrument, "detector", None)
        if detector is None or not hasattr(detector, "idet"):
            return np.full(self.n_points, np.nan)
        return np.array([detector.getWidthSquared(e_init, en) for en in self.energy_transfers(e_init)])

    def sample_width_squared(self) -> float:
        self._activate()
        sample = getattr(self.instrument, "sample", None)
        if sample is None:
            return np.nan
        return float(sample.getWidthSquared())

    def distances(self) -> np.ndarray:
        """(moderator->last chopper, aperture->last chopper, chopper->sample, sample->detector,
        moderator->first chopper) in metres."""
        self._activate()
        return np.array(self.chopper_system.getDistances(), dtype=float)


def classify_cases(
    entry: dict,
    evaluators: dict[str, PyChopEvaluator],
    restrictions: dict[str, dict],
    sampled: dict[str, list[sampling.SampledCase]],
) -> tuple[list[tuple], Counter]:
    """Assign an outcome to every sampled case.

    Returns rows ``(case, outcome, reason)`` and a counter of PyChop/resins disagreements.
    """
    frequency_settings = entry["frequency_settings"]
    rows = []
    notes: Counter = Counter()

    for name, cases in sampled.items():
        evaluator = evaluators[name]
        for case in cases:
            rejected = sampling.check_restrictions(
                restrictions[name], case.e_init, frequency_settings, case.frequencies
            )
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
                rows.append((case, sampling.OUTCOME_REJECTED_FREQUENCY, f"pychop:{frequency_settings[0]}"))
                continue

            if evaluator.is_fermi and not evaluator.ei_accepted(case.e_init):
                # PyChop's setEi limits are not applied by getResolution, which is what resins
                # is compared with, so the case keeps its getResolution outcome.
                notes["setEi limits narrower than resins' e_init restriction"] += 1

            resolution = evaluator.resolution(case.e_init)
            outcome = (
                sampling.OUTCOME_NO_TRANSMISSION
                if np.any(np.isnan(resolution))
                else sampling.OUTCOME_VALUE
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
    cases = np.zeros(n_cases, dtype=PYCHOP_CASE_DTYPE)
    resolution = np.full((n_cases, n_points), np.nan)
    chopper_width_sq = np.full((n_cases, 2), np.nan)
    chop_times_list: list = [None] * n_cases

    for i, (case, outcome, reason) in enumerate(rows):
        frequencies = list(case.frequencies) + [np.nan] * (MAX_FREQUENCIES - len(case.frequencies))
        cases[i] = (
            case.configuration,
            case.e_init,
            frequencies[0],
            frequencies[1],
            OUTCOME_CODE[outcome],
            reason,
        )
        if outcome not in (sampling.OUTCOME_VALUE, sampling.OUTCOME_NO_TRANSMISSION):
            continue
        evaluator = evaluators[case.configuration]
        evaluator.set_frequencies(case.frequencies)
        resolution[i] = evaluator.resolution(case.e_init)
        chopper_width_sq[i] = evaluator.chopper_width_squared(case.e_init)
        if not evaluator.is_fermi:
            chop_times_list[i] = evaluator.chop_times(case.e_init)

    arrays: dict[str, np.ndarray] = {
        "cases": cases,
        "resolution": resolution,
        "chopper_width_sq": chopper_width_sq,
    }

    # Non-Fermi: opening windows of first and last chopper, NaN-padded to a common length.
    if any(t is not None for t in chop_times_list):
        n_windows = max(len(w) for t in chop_times_list if t is not None for w in t)
        chop_times = np.full((n_cases, 2, n_windows, 2), np.nan)
        for i, times in enumerate(chop_times_list):
            if times is None:
                continue
            for k, windows in enumerate(times):
                chop_times[i, k, : len(windows)] = windows
        arrays["chop_times"] = chop_times

    # Per (configuration, Ei): quantities that do not depend on chopper frequency.
    evaluated = [
        (c, e)
        for (c, e) in dict.fromkeys(
            (case.configuration, case.e_init)
            for case, outcome, _ in rows
            if outcome in (sampling.OUTCOME_VALUE, sampling.OUTCOME_NO_TRANSMISSION)
        )
    ]
    ei_keys = np.zeros(len(evaluated), dtype=[("configuration", "U32"), ("e_init", "f8")])
    moderator = np.full(len(evaluated), np.nan)
    moderator_analytical = np.full(len(evaluated), np.nan)
    detector = np.full((len(evaluated), n_points), np.nan)
    for k, (configuration, e_init) in enumerate(evaluated):
        evaluator = evaluators[configuration]
        ei_keys[k] = (configuration, e_init)
        moderator[k], moderator_analytical[k] = evaluator.moderator_width_squared(e_init)
        detector[k] = evaluator.detector_width_squared(e_init)
    arrays["ei_keys"] = ei_keys
    arrays["moderator_width_sq"] = moderator
    arrays["moderator_width_sq_analytical"] = moderator_analytical
    arrays["detector_width_sq"] = detector

    # Per configuration
    names = list(evaluators)
    arrays["config_names"] = np.array(names, dtype="U32")
    arrays["sample_width_sq"] = np.array([evaluators[n].sample_width_squared() for n in names])
    arrays["distances"] = np.array([evaluators[n].distances() for n in names])
    return arrays


def generate_entry(entry: dict, config: dict, out_dir: Path) -> Path:
    sampling_config = {**config["sampling"], **entry.get("sampling", {})}
    params = sampling.SamplingParameters.from_config(sampling_config)
    n_points = config["sampling"]["n_resolution_points"]
    frequency_settings = entry["frequency_settings"]

    model = sampling.load_model(entry["instrument"], entry["model_version"])
    options = sampling.configuration_options(model)
    defaults = model["parameters"]["defaults"]
    config_map = entry.get("configuration_map", {})

    evaluators: dict[str, PyChopEvaluator] = {}
    restrictions: dict[str, dict] = {}
    sampled: dict[str, list[sampling.SampledCase]] = {}
    for name, option in options.items():
        evaluator = PyChopEvaluator(entry["pychop_name"], config_map.get(name, name), n_points)
        evaluators[name] = evaluator
        restrictions[name] = sampling.effective_restrictions(model, option)
        sampled[name] = sampling.sample_configuration(
            name,
            restrictions[name],
            defaults,
            frequency_settings,
            evaluator.transmits,
            params,
        )

    rows, notes = classify_cases(entry, evaluators, restrictions, sampled)
    arrays = build_arrays(entry, evaluators, rows, n_points)

    metadata = make_metadata(
        kind="pychop",
        instrument=entry["instrument"],
        model_version=entry["model_version"],
        pychop_name=entry["pychop_name"],
        pychop_source=str(Path(sys.modules[Instrument.__module__].__file__).parent),
        environment=entry["environment"],
        frequency_settings=frequency_settings,
        configuration_map=config_map,
        sampling=sampling_config,
        n_resolution_points=n_points,
        resolution_grid="linspace(0, e_init, n_resolution_points, endpoint=False)",
        notes=dict(notes),
    )

    path = out_dir / "pychop" / f"{pychop_file_stem(entry)}.npz"
    save_npz(path, arrays, metadata)

    counts = Counter(sampling.OUTCOMES[c] for c in arrays["cases"]["outcome"])
    summary = ", ".join(f"{k}={counts.get(k, 0)}" for k in sampling.OUTCOMES)
    print(f"{path.name}: {len(rows)} cases ({summary}); {path.stat().st_size / 1e3:.0f} kB")
    for note, count in notes.items():
        print(f"  note: {count} cases where {note}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Mantid environment name, e.g. mantid615")
    parser.add_argument("--out", type=Path, default=BUILD_DATA, help="Output directory")
    parser.add_argument("--only", nargs="*", help="Restrict to these file stems, e.g. MARI_PyChop_fit_v1")
    args = parser.parse_args(argv)

    config = load_config()
    entries = [e for e in config["pychop"] if e["environment"] == args.env]
    if args.only:
        entries = [e for e in entries if pychop_file_stem(e) in args.only]
    if not entries:
        print(f"No PyChop entries for environment {args.env}")
        return 0

    for entry in entries:
        generate_entry(entry, config, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
