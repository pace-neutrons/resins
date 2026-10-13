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
    frequencies: tuple[int, ...]


def load_cases() -> dict:
    """Load and return the ``cases.yaml`` content."""
    cases_path = Path(__file__).resolve().parents[1] / "cases.yaml"
    return yaml.safe_load(cases_path.read_text())


def _ei_limit_points(ei_limits: tuple[float, float]) -> list[float]:
    """Energies just inside and just outside both ends of the allowed limits.

    Mirrors the old ``sampling.ei_limit_points``: the lower end is skipped on the inside
    if it is zero (PyChop is undefined there); its outside point is a small negative energy.
    """
    lower, upper = ei_limits
    points = []
    if lower > 0:
        points += [lower, lower * SCALE_UP, lower * SCALE_DOWN]
    else:
        points += [-LIMIT_DELTA * upper if upper else -LIMIT_DELTA]
    points += [upper, upper * SCALE_DOWN, upper * SCALE_UP]
    return [round(p, 6) for p in points]


def _frequency_limit_points(values: list[int]) -> list[int]:
    """Integer frequencies just outside the allowed set: above the maximum and off the grid."""
    above = values[-1] + 1
    off_grid = values[0] + 1
    if above in values or off_grid in values:
        raise ValueError(f"Frequency steps of 1 leave no off-grid probe in {values}")
    return [above, off_grid]


@dataclass(frozen=True)
class ConfigurationSpec:
    """Fully resolved parameter specification for one instrument configuration.

    Parameters
    ----------
    default_ei
        Default incident energy in meV.
    default_frequencies
        Default frequency for each chopper (1-tuple for single-chopper
        instruments, 2-tuple for multi-chopper instruments like LET/CNCS).
    ei_limits
        (min, max) allowed incident energy bounds in meV.
    """

    name: str
    energies: list[float]
    frequency_combinations: list[tuple[int, ...]]
    frequency_limits: list[list[int]]
    ei_limits: tuple[float, float]
    default_ei: float
    default_frequencies: tuple[int, ...]
    edges: list[tuple[tuple[int, ...], list[float]]]
    frequency_settings: list[str]

    def expand(self) -> list[SampledCase]:
        """Generate all sampled cases for this configuration."""
        cases: dict[tuple, SampledCase] = {}

        def add(energy: float, frequencies: tuple[int, ...]) -> None:
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
        for energy in _ei_limit_points(self.ei_limits):
            add(energy, self.default_frequencies)

        # 4. Frequency limits
        for index, freq_values in enumerate(self.frequency_limits):
            for frequency in _frequency_limit_points(freq_values):
                perturbed = list(self.default_frequencies)
                perturbed[index] = frequency
                add(self.default_ei, tuple(perturbed))

        return list(cases.values())


def _parse_configuration_specs(entry: dict) -> list[ConfigurationSpec]:
    """Resolve base grids and configuration overrides into ConfigurationSpec instances."""
    grid = entry["grid"]
    freq_settings = entry["frequency_settings"]
    defaults = entry["defaults"]
    ei_limits = tuple(
        float(e) for e in entry.get("ei_limits", entry.get("ei_range", ()))
    )

    if "frequencies" in grid:
        base_combinations = [
            tuple(int(f) for f in pair) for pair in grid["frequencies"]
        ]
        base_freq_values = [
            sorted({pair[idx] for pair in grid["frequencies"]})
            for idx in range(len(freq_settings))
        ]
    else:
        setting = freq_settings[0]
        base_combinations = [(int(f),) for f in grid[setting]]
        base_freq_values = [grid[setting]]

    default_ei = float(defaults["e_init"])
    default_frequencies = tuple(int(defaults[setting]) for setting in freq_settings)

    specs = []
    for config_name, config in entry.get("configurations", {}).items():
        config_grid = [
            float(e) for e in config.get("grid", {}).get("e_init", grid["e_init"])
        ]
        config_ei_limits = tuple(
            float(e) for e in config.get("ei_limits", config.get("ei_range", ei_limits))
        )

        if "frequencies" in config.get("grid", {}):
            config_combinations = [
                tuple(int(f) for f in pair) for pair in config["grid"]["frequencies"]
            ]
            config_freq_values = [
                sorted({pair[idx] for pair in config["grid"]["frequencies"]})
                for idx in range(len(freq_settings))
            ]
        else:
            config_combinations = base_combinations
            config_freq_values = []
            for idx, setting in enumerate(freq_settings):
                if setting in config.get("restrictions", {}):
                    start, stop, step = config["restrictions"][setting]
                    config_freq_values.append(list(range(start, stop, step)))
                else:
                    config_freq_values.append(base_freq_values[idx])

        cfg_default_ei = default_ei
        if not (config_ei_limits[0] <= cfg_default_ei <= config_ei_limits[1]):
            cfg_default_ei = float(config_grid[len(config_grid) // 2])

        edges = [
            (
                tuple(int(edge[setting]) for setting in freq_settings),
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
                ei_limits=config_ei_limits,
                default_ei=cfg_default_ei,
                default_frequencies=default_frequencies,
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
    return {spec.name: spec.expand() for spec in _parse_configuration_specs(entry)}


def configuration_names(entry: dict) -> list[str]:
    """All configuration names declared for an entry."""
    return sorted(entry.get("configurations", {}))


def check_restrictions(
    ei_limits: tuple[float, float],
    frequency_values: list[list[int] | None],
    frequency_settings: list[str],
    e_init: float,
    frequencies: list[int],
) -> tuple[Outcome, str] | None:
    """Apply restriction rules, returning ``(outcome, setting)`` on rejection.

    ``e_init`` is checked first, then each frequency against its list of allowed values.
    ``setting`` is the name of the offending resins setting.
    """
    if not (ei_limits[0] <= e_init <= ei_limits[1]):
        return Outcome.REJECTED_EI, "e_init"

    for allowed, setting, frequency in zip(
        frequency_values, frequency_settings, frequencies
    ):
        if allowed is None:
            continue
        if frequency not in allowed:
            return Outcome.REJECTED_FREQUENCY, setting

    return None
