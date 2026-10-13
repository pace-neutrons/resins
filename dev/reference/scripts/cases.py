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


@dataclass(frozen=True)
class ConfigurationSpec:
    """Fully resolved parameter specification for one instrument configuration."""

    name: str
    energies: list[float]
    frequency_combinations: list[tuple[float, ...]]
    frequency_limits: list[list[int]]
    ei_range: tuple[float, float]
    nominal_ei: float
    nominal_frequencies: tuple[float, ...]
    edges: list[tuple[tuple[float, ...], list[float]]]
    frequency_settings: list[str]

    def expand(self) -> list[SampledCase]:
        """Generate all sampled cases for this configuration."""
        cases: dict[tuple, SampledCase] = {}

        def add(energy: float, frequencies: tuple[float, ...]) -> None:
            key = (round(energy, 9), tuple(frequencies))
            cases.setdefault(key, SampledCase(self.name, energy, tuple(frequencies)))

        # 1. Regular grid
        for frequencies in self.frequency_combinations:
            for energy in self.energies:
                add(energy, frequencies)

        # 2. Edge points
        for frequencies, edge_energies in self.edges:
            for energy in edge_energies:
                add(energy, frequencies)

        # 3. Energy limits
        for energy in _ei_limit_points(list(self.ei_range)):
            add(energy, self.nominal_frequencies)

        # 4. Frequency limits
        for index, freq_values in enumerate(self.frequency_limits):
            for frequency in _frequency_limit_points(freq_values):
                perturbed = list(self.nominal_frequencies)
                perturbed[index] = frequency
                add(self.nominal_ei, tuple(perturbed))

        return list(cases.values())


def _parse_configuration_specs(entry: dict) -> list[ConfigurationSpec]:
    """Resolve base grids and configuration overrides into ConfigurationSpec instances."""
    grid = entry["grid"]
    freq_settings = entry["frequency_settings"]
    defaults = entry["defaults"]
    ei_range = entry["ei_range"]

    if "frequencies" in grid:
        base_combinations = [
            tuple(float(f) for f in pair) for pair in grid["frequencies"]
        ]
        base_freq_values = [
            sorted({pair[idx] for pair in grid["frequencies"]})
            for idx in range(len(freq_settings))
        ]
    else:
        setting = freq_settings[0]
        base_combinations = [(float(f),) for f in grid[setting]]
        base_freq_values = [grid[setting]]

    nominal_ei = float(defaults["e_init"])
    nominal_frequencies = tuple(
        float(defaults[setting]) for setting in freq_settings
    )

    specs = []
    for config_name, config in entry.get("configurations", {}).items():
        config = config or {}
        config_grid = [
            float(e)
            for e in config.get("grid", {}).get("e_init", grid["e_init"])
        ]
        config_ei_range = tuple(
            float(e) for e in config.get("ei_range", ei_range)
        )

        if "frequencies" in config.get("grid", {}):
            config_combinations = [
                tuple(float(f) for f in pair)
                for pair in config["grid"]["frequencies"]
            ]
            config_freq_values = [
                sorted({pair[idx] for pair in config["grid"]["frequencies"]})
                for idx in range(len(freq_settings))
            ]
        else:
            config_combinations = base_combinations
            config_freq_values = []
            for idx, setting in enumerate(freq_settings):
                if (
                    "restrictions" in config
                    and setting in config["restrictions"]
                ):
                    start, stop, step = config["restrictions"][setting]
                    config_freq_values.append(list(range(start, stop, step)))
                else:
                    config_freq_values.append(base_freq_values[idx])

        config_nominal_ei = nominal_ei
        if not (config_ei_range[0] <= config_nominal_ei <= config_ei_range[1]):
            config_nominal_ei = float(config_grid[len(config_grid) // 2])

        edges = [
            (
                tuple(float(edge[setting]) for setting in freq_settings),
                [float(e) for e in edge["e_init"]],
            )
            for edge in config.get("edges", [])
        ]

        specs.append(
            ConfigurationSpec(
                name=config_name,
                energies=config_grid,
                frequency_combinations=config_combinations,
                frequency_limits=config_freq_values,
                ei_range=config_ei_range,
                nominal_ei=config_nominal_ei,
                nominal_frequencies=nominal_frequencies,
                edges=edges,
                frequency_settings=freq_settings,
            )
        )

    return specs


def expand_entry(entry: dict) -> dict[str, list[SampledCase]]:
    """Expand one instrument-model entry from ``cases.yaml`` into cases per configuration.

    Returns a dict mapping configuration name to a list of :class:`SampledCase`, without
    duplicates, in a deterministic order.
    """
    return {
        spec.name: spec.expand() for spec in _parse_configuration_specs(entry)
    }


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
