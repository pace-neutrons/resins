"""Case sampling for the reference data generators.

Pure numpy and PyYAML: this module imports neither resins nor Mantid, so that it can run in
every pixi environment. Configurations, defaults and restrictions are read from the resins
instrument YAML files as plain data.

A *case* is one (configuration, incident energy, chopper frequencies) combination. Cases are
chosen by three rules:

* regular: a coarse geometric grid of incident energies at every allowed chopper frequency;
* edge: one point either side of each transmission edge in incident energy;
* limit: points just inside and just outside the incident energy and chopper frequency limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTRUMENT_DATA = REPO_ROOT / "src" / "resins" / "instrument_data"

OUTCOME_VALUE = "value"
OUTCOME_NO_TRANSMISSION = "no_transmission"
OUTCOME_REJECTED_EI = "rejected_ei"
OUTCOME_REJECTED_FREQUENCY = "rejected_frequency"
OUTCOMES = (
    OUTCOME_VALUE,
    OUTCOME_NO_TRANSMISSION,
    OUTCOME_REJECTED_EI,
    OUTCOME_REJECTED_FREQUENCY,
)


@dataclass(frozen=True)
class SamplingParameters:
    n_ei: int = 6
    ei_floor_ratio: float = 0.005
    edge_scan_points: int = 200
    edge_bisect_rtol: float = 1e-7
    edge_delta: float = 1e-3
    limit_delta: float = 1e-3

    @classmethod
    def from_config(cls, config: dict) -> SamplingParameters:
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in config.items() if k in names})


@dataclass(frozen=True)
class SampledCase:
    """A sampled combination of settings, before an outcome has been assigned."""

    configuration: str
    e_init: float
    frequencies: tuple[float, ...]
    origin: str  # "regular", "edge" or "limit"


# ----------------------------------------------------------------------------------------------
# resins instrument YAML as data
# ----------------------------------------------------------------------------------------------


def load_model(instrument: str, model_version: str, instrument_version: str | None = None) -> dict:
    """Return the YAML model entry (with ``parameters`` and ``configurations``) as a dict."""
    with open(INSTRUMENT_DATA / f"{instrument.lower()}.yaml") as f:
        data = yaml.safe_load(f)
    version = instrument_version or data["default_version"]
    models = data["version"][version]["models"]
    model = models[model_version]
    if isinstance(model, str):  # alias
        model = models[model]
    return model


def configuration_options(model: dict, configuration_name: str = "chopper_package") -> dict[str, dict]:
    """All options (e.g. chopper packages) of a model's configuration, keyed by option name."""
    options = model["configurations"][configuration_name]
    return {name: value for name, value in options.items() if name != "default_option"}


def effective_restrictions(model: dict, option: dict) -> dict:
    """Restrictions for one option; an option's own ``restrictions`` replace the model's."""
    if "restrictions" in option:
        return option["restrictions"]
    return model["parameters"].get("restrictions", {})


def allowed_frequencies(restriction: Sequence[float]) -> list[int]:
    """Explicit frequency list for a ``[start, stop, step]`` restriction."""
    if len(restriction) != 3:
        raise ValueError(f"Frequency restriction must be [start, stop, step], got {restriction}")
    return list(range(*restriction))


def check_restrictions(
    restrictions: dict,
    e_init: float,
    frequency_settings: Sequence[str],
    frequencies: Sequence[float],
) -> tuple[str, str] | None:
    """Apply resins' restriction rules, returning ``(outcome, setting)`` on rejection.

    Mirrors ``InstrumentModel._validate_settings``: ``e_init`` is checked first.
    """
    ei_range = restrictions.get("e_init")
    if ei_range is not None and not (ei_range[0] <= e_init <= ei_range[1]):
        return OUTCOME_REJECTED_EI, "e_init"

    for name, frequency in zip(frequency_settings, frequencies):
        restriction = restrictions.get(name)
        if restriction is None:
            continue
        start, stop, step = restriction
        if frequency not in range(start, stop, step):
            return OUTCOME_REJECTED_FREQUENCY, name

    return None


# ----------------------------------------------------------------------------------------------
# Regular part
# ----------------------------------------------------------------------------------------------


def _round_sig(values: np.ndarray, digits: int = 4) -> np.ndarray:
    return np.array([float(f"{v:.{digits - 1}e}") for v in values])


def geometric_ei_grid(ei_range: Sequence[float], n_ei: int, floor_ratio: float) -> np.ndarray:
    """``n_ei`` geometrically spaced energies, clipped away from zero, rounded to 4 digits."""
    lower, upper = ei_range
    lower = max(lower, floor_ratio * upper)
    return _round_sig(np.geomspace(lower, upper, n_ei))


def frequency_combinations(frequency_lists: Sequence[Sequence[int]]) -> list[tuple[float, ...]]:
    """Frequency (tuples) such that every allowed value of each chopper occurs at least once.

    One chopper: every value. Several choppers: a cyclic pairing of the lists instead of the
    full product (e.g. 30 pairs for LET instead of 900).
    """
    if len(frequency_lists) == 1:
        return [(float(f),) for f in frequency_lists[0]]

    n = max(len(values) for values in frequency_lists)
    combinations = []
    for i in range(n):
        # Chopper k is offset by k positions in its list so that pairs are not diagonal.
        combinations.append(
            tuple(float(values[(i + k) % len(values)]) for k, values in enumerate(frequency_lists))
        )
    return combinations


# ----------------------------------------------------------------------------------------------
# Edge part
# ----------------------------------------------------------------------------------------------


def find_transmission_edges(
    transmits: Callable[[float], bool],
    ei_range: Sequence[float],
    n_scan: int = 200,
    rtol: float = 1e-7,
) -> list[float]:
    """Locate the energies at which ``transmits`` changes state.

    The range is scanned on a logarithmic grid of ``n_scan`` points, and each state change is
    bisected (in log energy) until the bracket is narrower than ``rtol`` relative.
    """
    lower, upper = ei_range
    grid = np.geomspace(lower, upper, n_scan)
    states = [bool(transmits(float(e))) for e in grid]

    edges = []
    for i in range(len(grid) - 1):
        if states[i] == states[i + 1]:
            continue
        low, high, low_state = float(grid[i]), float(grid[i + 1]), states[i]
        while high / low - 1.0 > rtol:
            mid = math.sqrt(low * high)
            if bool(transmits(mid)) == low_state:
                low = mid
            else:
                high = mid
        edges.append(math.sqrt(low * high))
    return edges


def edge_points(
    transmits: Callable[[float], bool],
    ei_range: Sequence[float],
    n_scan: int = 200,
    rtol: float = 1e-7,
    delta: float = 1e-3,
) -> list[float]:
    """Points ``edge * (1 - delta)`` and ``edge * (1 + delta)`` for every edge, within the range."""
    lower, upper = ei_range
    points = []
    for edge in find_transmission_edges(transmits, ei_range, n_scan, rtol):
        for point in (edge * (1.0 - delta), edge * (1.0 + delta)):
            if lower <= point <= upper:
                points.append(point)
    return points


# ----------------------------------------------------------------------------------------------
# Limit part
# ----------------------------------------------------------------------------------------------


def ei_limit_points(ei_range: Sequence[float], delta: float = 1e-3) -> list[float]:
    """Energies just inside and just outside both ends of the allowed range.

    The lower end is skipped on the inside if it is zero, where PyChop is undefined; its
    outside point is then a small negative energy.
    """
    lower, upper = ei_range
    points = []
    if lower > 0:
        points += [lower, lower * (1.0 + delta), lower * (1.0 - delta)]
    else:
        points += [-delta * upper if upper else -delta]
    points += [upper, upper * (1.0 - delta), upper * (1.0 + delta)]
    return [float(f"{p:.8g}") for p in points]


def frequency_limit_points(restriction: Sequence[int], delta: float = 1e-3) -> list[float]:
    """Frequencies just outside the allowed set: above the maximum, and one not on the grid."""
    start, stop, step = restriction
    values = range(start, stop, step)
    highest = values[-1]
    above = float(f"{highest * (1.0 + delta):.6g}")
    off_grid = float(start + step // 2 if step > 1 else start + 0.5)
    if off_grid in values:  # pragma: no cover - only with exotic ranges
        off_grid += 0.5
    return [above, off_grid]


# ----------------------------------------------------------------------------------------------
# Putting it together
# ----------------------------------------------------------------------------------------------


def sample_configuration(
    configuration: str,
    restrictions: dict,
    defaults: dict,
    frequency_settings: Sequence[str],
    transmits: Callable[[float, tuple[float, ...]], bool] | None,
    params: SamplingParameters,
) -> list[SampledCase]:
    """All cases for one configuration, without duplicates, in a deterministic order.

    ``transmits(e_init, frequencies)`` tells whether the reference implementation transmits
    neutrons; if None, edge points are skipped.
    """
    ei_range = restrictions["e_init"]
    frequency_lists = [allowed_frequencies(restrictions[name]) for name in frequency_settings]
    nominal_frequencies = tuple(float(defaults[name]) for name in frequency_settings)
    nominal_ei = float(defaults["e_init"])
    cases: dict[tuple, SampledCase] = {}

    def add(e_init: float, frequencies: Iterable[float], origin: str) -> None:
        key = (round(e_init, 9), tuple(frequencies))
        cases.setdefault(key, SampledCase(configuration, e_init, tuple(frequencies), origin))

    ei_grid = geometric_ei_grid(ei_range, params.n_ei, params.ei_floor_ratio)
    if not ei_range[0] <= nominal_ei <= ei_range[1]:
        # The default energy is outside this configuration's range (e.g. MERLIN G): use the
        # middle of the grid so that frequency-limit cases are rejected for frequency only.
        nominal_ei = float(ei_grid[len(ei_grid) // 2])
    combinations = frequency_combinations(frequency_lists)

    # Regular
    for frequencies in combinations:
        for e_init in ei_grid:
            add(float(e_init), frequencies, "regular")

    # Edges
    if transmits is not None:
        scan_range = (max(ei_range[0], params.ei_floor_ratio * ei_range[1]), ei_range[1])
        for frequencies in combinations:
            points = edge_points(
                lambda e, f=frequencies: transmits(e, f),
                scan_range,
                params.edge_scan_points,
                params.edge_bisect_rtol,
                params.edge_delta,
            )
            for e_init in points:
                # Five digits keep the point well inside delta while making it robust to the
                # last digits of the bisection result.
                add(float(f"{e_init:.5g}"), frequencies, "edge")

    # Limits: Ei, at nominal frequencies
    for e_init in ei_limit_points(ei_range, params.limit_delta):
        add(e_init, nominal_frequencies, "limit")

    # Limits: each frequency in turn, other choppers at nominal values
    for i, name in enumerate(frequency_settings):
        for frequency in frequency_limit_points(restrictions[name], params.limit_delta):
            frequencies = list(nominal_frequencies)
            frequencies[i] = frequency
            add(nominal_ei, frequencies, "limit")

    return list(cases.values())
