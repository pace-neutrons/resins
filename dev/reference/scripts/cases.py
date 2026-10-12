"""Static case manifest reader for the reference data generators.

Pure numpy and PyYAML: this module imports neither resins nor Mantid, so it runs in every
pixi environment. Target test points are read directly from ``cases.yaml`` (design D3/D4),
removing the need for a runtime bisection solver or a standalone resins YAML restriction parser.

A *case* is one (configuration, incident energy, chopper frequencies) combination. Cases
come from three sources, all declared in ``cases.yaml``:

* regular: the instrument-level base grid, a coarse Cartesian product of incident energies
  and chopper frequencies (inherited by all configurations, unless overridden);
* edge: per-configuration, per-frequency (Ei, f) points straddling zero-transmission edges;
* limit: incident energies just inside/outside the allowed Ei range at the default frequency,
  and frequencies just above the maximum and off the grid at the default incident energy.

Outcome codes are assigned by the generator from PyChop and the restrictions, not stored here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

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

# Relative offset for points just inside / outside the Ei and frequency limits.
LIMIT_DELTA = 1e-3


@dataclass(frozen=True)
class SampledCase:
    """A case point, before an outcome has been assigned."""

    configuration: str
    e_init: float
    frequencies: tuple[float, ...]


def load_cases() -> dict:
    """Load and return the ``cases.yaml`` content."""
    with open(Path(__file__).resolve().parent.parent / "cases.yaml") as f:
        return yaml.safe_load(f)


def _ei_limit_points(ei_range: list[float]) -> list[float]:
    """Energies just inside and just outside both ends of the allowed range.

    Mirrors the old ``sampling.ei_limit_points``: the lower end is skipped on the inside
    if it is zero (PyChop is undefined there); its outside point is a small negative energy.
    """
    lower, upper = ei_range
    points = []
    if lower > 0:
        points += [lower, lower * (1.0 + LIMIT_DELTA), lower * (1.0 - LIMIT_DELTA)]
    else:
        points += [-LIMIT_DELTA * upper if upper else -LIMIT_DELTA]
    points += [upper, upper * (1.0 - LIMIT_DELTA), upper * (1.0 + LIMIT_DELTA)]
    return [float(f"{p:.8g}") for p in points]


def _frequency_limit_points(restriction: list[int]) -> list[float]:
    """Frequencies just outside the allowed set: above the maximum, and one not on the grid."""
    start, stop, step = restriction
    values = range(start, stop, step)
    highest = values[-1]
    above = float(f"{highest * (1.0 + LIMIT_DELTA):.6g}")
    off_grid = float(start + step // 2 if step > 1 else start + 0.5)
    if off_grid in values:  # pragma: no cover - only with exotic ranges
        off_grid += 0.5
    return [above, off_grid]


def _frequency_limit_points_from_values(values: list[int]) -> list[float]:
    """Frequencies just outside the allowed set, given the explicit list of allowed values."""
    highest = values[-1]
    above = float(f"{highest * (1.0 + LIMIT_DELTA):.6g}")
    step = values[1] - values[0] if len(values) > 1 else 1
    start = values[0]
    off_grid = float(start + step // 2 if step > 1 else start + 0.5)
    if off_grid in values:  # pragma: no cover - only with exotic ranges
        off_grid += 0.5
    return [above, off_grid]


def _frequency_combinations(frequency_lists: list[list[int]]) -> list[tuple[float, ...]]:
    """Frequency tuples such that every allowed value of each chopper occurs at least once.

    One chopper: every value. Several choppers: a cyclic pairing instead of the full product.
    """
    if len(frequency_lists) == 1:
        return [(float(f),) for f in frequency_lists[0]]

    n = max(len(values) for values in frequency_lists)
    combinations = []
    for i in range(n):
        combinations.append(
            tuple(float(values[(i + k) % len(values)]) for k, values in enumerate(frequency_lists))
        )
    return combinations


def expand_entry(entry: dict) -> dict[str, list[SampledCase]]:
    """Expand one instrument-model entry from ``cases.yaml`` into cases per configuration.

    Returns a dict mapping configuration name to a list of :class:`SampledCase`, without
    duplicates, in a deterministic order.
    """
    grid = entry["grid"]
    freq_settings = entry["frequency_settings"]
    defaults = entry["defaults"]
    ei_range = entry["ei_range"]
    configs = entry.get("configurations", {})

    grid_eis = grid["e_init"]
    freq_lists = [grid[s] for s in freq_settings]
    combinations = _frequency_combinations(freq_lists)

    nominal_ei = float(defaults["e_init"])
    nominal_freqs = tuple(float(defaults[s]) for s in freq_settings)

    result: dict[str, list[SampledCase]] = {}

    # Determine the set of configurations: those named in cases.yaml, plus any that the
    # config.yaml maps to this entry (the caller passes the full set).
    for config_name in configs:
        config = configs[config_name] or {}

        # Grid: config override or instrument base
        config_grid = config.get("grid", {}).get("e_init", grid_eis)
        config_ei_range = config.get("ei_range", ei_range)

        # Frequency restrictions: config override or instrument base
        config_freq_lists = []
        for idx, s in enumerate(freq_settings):
            if "restrictions" in config and s in config["restrictions"]:
                start, stop, step = config["restrictions"][s]
                config_freq_lists.append(list(range(start, stop, step)))
            else:
                config_freq_lists.append(freq_lists[idx])

        config_combinations = _frequency_combinations(config_freq_lists)
        # If the config has frequency overrides, recompute default frequencies for limit cases
        config_nominal_freqs = tuple(float(defaults[s]) for s in freq_settings)

        # If the default Ei is outside this config's range, use the middle of the grid
        config_nominal_ei = nominal_ei
        if not (config_ei_range[0] <= config_nominal_ei <= config_ei_range[1]):
            config_nominal_ei = float(config_grid[len(config_grid) // 2])

        cases: dict[tuple, SampledCase] = {}

        def add(e_init: float, frequencies, _cases=cases) -> None:
            key = (round(e_init, 9), tuple(frequencies))
            _cases.setdefault(key, SampledCase(config_name, e_init, tuple(frequencies)))

        # Regular grid
        for frequencies in config_combinations:
            for e_init in config_grid:
                add(float(e_init), frequencies)

        # Edge pairs
        for edge in config.get("edges", []):
            freq_values = tuple(float(edge[s]) for s in freq_settings)
            for e_init in edge["e_init"]:
                add(float(e_init), freq_values)

        # Limit Ei: at nominal (or mid-grid) frequencies
        for e_init in _ei_limit_points(config_ei_range):
            add(e_init, config_nominal_freqs)

        # Limit frequencies: each chopper in turn, others at nominal
        for i, name in enumerate(freq_settings):
            r = config_freq_lists[i]
            for frequency in _frequency_limit_points_from_values(r):
                frequencies = list(config_nominal_freqs)
                frequencies[i] = frequency
                add(config_nominal_ei, frequencies)

        result[config_name] = list(cases.values())

    return result


def configuration_names(entry: dict) -> list[str]:
    """All configuration names declared for an entry."""
    return sorted(entry.get("configurations", {}))


def check_restrictions(
    ei_range: list[float],
    frequency_values: list[list[int] | None],
    e_init: float,
    frequencies: list[float],
) -> tuple[str, str] | None:
    """Apply restriction rules, returning ``(outcome, setting)`` on rejection.

    ``e_init`` is checked first, then each frequency against its list of allowed values.
    """
    if not (ei_range[0] <= e_init <= ei_range[1]):
        return OUTCOME_REJECTED_EI, "e_init"

    for allowed, frequency in zip(frequency_values, frequencies):
        if allowed is None:
            continue
        if frequency not in allowed:
            return OUTCOME_REJECTED_FREQUENCY, "frequency"

    return None
