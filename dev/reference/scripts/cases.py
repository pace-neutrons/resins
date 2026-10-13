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
from enum import Enum
from pathlib import Path

import yaml

# Relative offset for points just inside / outside the Ei and frequency limits.
LIMIT_DELTA = 1e-3
SCALE_UP = 1.0 + LIMIT_DELTA
SCALE_DOWN = 1.0 - LIMIT_DELTA


class Outcome(str, Enum):
    """Expected outcome for a reference case.

    Implemented as ``(str, Enum)`` with explicit ``__str__`` to provide
    ``StrEnum`` behavior while supporting Python 3.10 for ``mantid611`` pixi
    environment.
    """

    VALUE = "value"
    NO_TRANSMISSION = "no_transmission"
    REJECTED_EI = "rejected_ei"
    REJECTED_FREQUENCY = "rejected_frequency"

    def __str__(self) -> str:
        return self.value

    def __index__(self) -> int:
        return _OUTCOME_TO_INDEX[self]

    @classmethod
    def from_index(cls, index: int) -> Outcome:
        return _INDEX_TO_OUTCOME[index]

    @classmethod
    def from_code(cls, code: int) -> Outcome:
        return cls.from_index(code)


_INDEX_TO_OUTCOME = tuple(Outcome)
_OUTCOME_TO_INDEX = {outcome: i for i, outcome in enumerate(_INDEX_TO_OUTCOME)}


@dataclass(frozen=True)
class SampledCase:
    """A case point, before an outcome has been assigned."""

    configuration: str
    e_init: float
    frequencies: tuple[float, ...]


def load_cases() -> dict:
    """Load and return the ``cases.yaml`` content."""
    cases_path = Path(__file__).resolve().parents[1] / "cases.yaml"
    return yaml.safe_load(cases_path.read_text())


def _ei_limit_points(ei_range: list[float]) -> list[float]:
    """Energies just inside and just outside both ends of the allowed range.

    Mirrors the old ``sampling.ei_limit_points``: the lower end is skipped on the inside
    if it is zero (PyChop is undefined there); its outside point is a small negative energy.
    """
    lower, upper = ei_range
    points = []
    if lower > 0:
        points += [lower, lower * SCALE_UP, lower * SCALE_DOWN]
    else:
        points += [-LIMIT_DELTA * upper if upper else -LIMIT_DELTA]
    points += [upper, upper * SCALE_DOWN, upper * SCALE_UP]
    return [round(p, 6) for p in points]


def _frequency_limit_points(values: list[int]) -> list[float]:
    """Frequencies just outside the allowed set, given the explicit list of allowed values."""
    highest = values[-1]
    above = round(highest * SCALE_UP, 4)
    step = values[1] - values[0] if len(values) > 1 else 1
    start = values[0]
    off_grid = float(start + step // 2 if step > 1 else start + 0.5)
    if off_grid in values:  # pragma: no cover - only with exotic ranges
        off_grid += 0.5
    return [above, off_grid]


def _frequency_combinations(
    frequency_lists: list[list[int]],
) -> list[tuple[float, ...]]:
    """Frequency tuples such that every allowed value of each chopper occurs at least once.

    One chopper: every value. Several choppers: a cyclic pairing instead of the full product.
    """
    if len(frequency_lists) == 1:
        return [(float(f),) for f in frequency_lists[0]]

    n = max(len(values) for values in frequency_lists)
    combinations = []
    for i in range(n):
        combinations.append(
            tuple(
                float(values[(i + k) % len(values)])
                for k, values in enumerate(frequency_lists)
            )
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
    freq_lists = [grid[setting] for setting in freq_settings]

    nominal_ei = float(defaults["e_init"])

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
        for idx, setting in enumerate(freq_settings):
            if "restrictions" in config and setting in config["restrictions"]:
                start, stop, step = config["restrictions"][setting]
                config_freq_lists.append(list(range(start, stop, step)))
            else:
                config_freq_lists.append(freq_lists[idx])

        config_combinations = _frequency_combinations(config_freq_lists)
        # If the config has frequency overrides, recompute default frequencies for limit cases
        config_nominal_freqs = tuple(
            float(defaults[setting]) for setting in freq_settings
        )

        # If the default Ei is outside this config's range, use the middle of the grid
        config_nominal_ei = nominal_ei
        if not (config_ei_range[0] <= config_nominal_ei <= config_ei_range[1]):
            config_nominal_ei = float(config_grid[len(config_grid) // 2])

        cases: dict[tuple, SampledCase] = {}

        def add(
            e_init: float, frequencies, _cases=cases, _config_name=config_name
        ) -> None:
            key = (round(e_init, 9), tuple(frequencies))
            _cases.setdefault(
                key, SampledCase(_config_name, e_init, tuple(frequencies))
            )

        # Regular grid
        for frequencies in config_combinations:
            for e_init in config_grid:
                add(float(e_init), frequencies)

        # Edge pairs
        for edge in config.get("edges", []):
            freq_values = tuple(float(edge[setting]) for setting in freq_settings)
            for e_init in edge["e_init"]:
                add(float(e_init), freq_values)

        # Limit Ei: at nominal (or mid-grid) frequencies
        for e_init in _ei_limit_points(config_ei_range):
            add(e_init, config_nominal_freqs)

        # Limit frequencies: each chopper in turn, others at nominal
        for i, name in enumerate(freq_settings):
            freq_values_list = config_freq_lists[i]
            for frequency in _frequency_limit_points(freq_values_list):
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
        return Outcome.REJECTED_EI, "e_init"

    for allowed, frequency in zip(frequency_values, frequencies):
        if allowed is None:
            continue
        if frequency not in allowed:
            return Outcome.REJECTED_FREQUENCY, "frequency"

    return None
