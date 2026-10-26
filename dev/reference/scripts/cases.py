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

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

import yaml

# Offset for points just inside / outside the Ei limits
ABS_DELTA = 0.1


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


_INDEX_TO_OUTCOME = tuple(Outcome)
_OUTCOME_TO_INDEX = {outcome: i for i, outcome in enumerate(_INDEX_TO_OUTCOME)}


@dataclass(frozen=True)
class SampledCase:
    """A case point, before an outcome has been assigned."""

    configuration: str
    e_init: float
    frequencies: tuple[int, ...]


class ConfigurationDict(TypedDict, total=False):
    """Configuration dictionary from ``cases.yaml``."""

    edges: list[dict[str, Any]]
    grid: dict[str, Any]
    defaults: dict[str, Any]
    ei_limits: list[float]


class InstrumentDict(TypedDict):
    """Instrument-level dictionary from ``cases.yaml``."""

    grid: dict[str, Any]
    ei_limits: list[float]
    defaults: dict[str, float | int]
    frequency_settings: list[str]
    allowed_frequencies: dict[str, list[int]]
    configurations: dict[str, ConfigurationDict]


def load_cases() -> dict[str, InstrumentDict]:
    """Load and return the ``cases.yaml`` content."""
    cases_path = Path(__file__).resolve().parents[1] / "cases.yaml"
    return yaml.safe_load(cases_path.read_text())


def _ei_limit_points(lower: float, upper: float) -> list[float]:
    """Energies straddling both ends of the allowed limits."""
    return [
        round(p, 6)
        for p in (
            lower - ABS_DELTA,
            lower + ABS_DELTA,
            upper - ABS_DELTA,
            upper + ABS_DELTA,
        )
    ]


def _frequency_limit_points(values: set[int]) -> list[int]:
    """Integer frequencies just outside the allowed set: above the maximum and off the grid."""
    above = max(values) + 1
    off_grid = min(values) + 1
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
        Default frequency for each chopper (one entry per setting in ``frequency_settings``).
    ei_limits
        (min, max) allowed incident energy bounds in meV.
    allowed_frequencies
        Allowed discrete frequencies for each chopper setting (one set per setting in
        ``frequency_settings``).
    """

    name: str
    energies: list[float]
    frequency_combinations: list[tuple[int, ...]]
    allowed_frequencies: list[set[int]]
    ei_limits: tuple[float, float]
    default_ei: float
    default_frequencies: tuple[int, ...]
    edge_points: list[tuple[float, tuple[int, ...]]]
    frequency_settings: list[str]

    def expand(self) -> list[SampledCase]:
        """Generate all sampled cases for this configuration."""
        cases: dict[tuple, SampledCase] = {}

        def add(energy: float, frequencies: tuple[int, ...]) -> None:
            key = (round(energy, 9), frequencies)
            cases.setdefault(key, SampledCase(self.name, energy, frequencies))

        # 1. Regular grid
        for freqs, energy in itertools.product(
            self.frequency_combinations, self.energies
        ):
            add(energy, freqs)

        # 2. Edge points
        for energy, freqs in self.edge_points:
            add(energy, freqs)

        # 3. Energy limits
        for energy in _ei_limit_points(*self.ei_limits):
            add(energy, self.default_frequencies)

        # 4. Frequency limits
        for index, allowed in enumerate(self.allowed_frequencies):
            for frequency in _frequency_limit_points(allowed):
                perturbed = list(self.default_frequencies)
                perturbed[index] = frequency
                add(self.default_ei, tuple(perturbed))

        return list(cases.values())

    def check_restrictions(
        self, e_init: float, frequencies: Sequence[int]
    ) -> tuple[Outcome, str] | None:
        """Identify invalid e_init or frequencies.

        Return ``(outcome, setting)`` if invalid or ``None`` if there are no issues.
        """
        min_ei, max_ei = self.ei_limits
        if not (min_ei <= e_init <= max_ei):
            return Outcome.REJECTED_EI, "e_init"

        for allowed, setting, frequency in zip(
            self.allowed_frequencies, self.frequency_settings, frequencies, strict=True,
        ):
            if frequency not in allowed:
                return Outcome.REJECTED_FREQUENCY, setting

        return None


@dataclass(frozen=True)
class InstrumentSpec:
    """Instrument-level specification parsed from cases.yaml."""

    frequency_settings: list[str]
    ei_limits: tuple[float, float]
    default_ei: float
    default_frequencies: tuple[int, ...]
    grid_energies: list[float]
    frequency_combinations: list[tuple[int, ...]]
    allowed_frequencies: list[set[int]]
    configurations: dict[str, ConfigurationDict]

    @classmethod
    def from_dict(cls, data: InstrumentDict) -> InstrumentSpec:
        settings = data["frequency_settings"]
        grid = data["grid"]
        defaults = data["defaults"]
        raw_allowed = data["allowed_frequencies"]
        allowed = [set(raw_allowed[s]) for s in settings]

        min_ei, max_ei = data["ei_limits"]

        combinations: list[tuple[int, ...]]
        if "frequencies" in grid:
            combinations = [tuple(int(f) for f in p) for p in grid["frequencies"]]
        else:
            combinations = [(int(f),) for f in grid[settings[0]]]

        return cls(
            frequency_settings=settings,
            ei_limits=(float(min_ei), float(max_ei)),
            default_ei=float(defaults["e_init"]),
            default_frequencies=tuple(int(defaults[s]) for s in settings),
            grid_energies=[float(e) for e in grid["e_init"]],
            frequency_combinations=combinations,
            allowed_frequencies=allowed,
            configurations=data["configurations"],
        )

    def parse_configurations(self) -> dict[str, ConfigurationSpec]:
        """Produce fully resolved ConfigurationSpec instances per configuration."""
        specs = {}
        for name, cfg in self.configurations.items():
            energies = (
                [float(e) for e in cfg["grid"]["e_init"]]
                if "grid" in cfg and "e_init" in cfg["grid"]
                else self.grid_energies
            )
            if "ei_limits" in cfg:
                cfg_min, cfg_max = cfg["ei_limits"]
                ei_limits = (float(cfg_min), float(cfg_max))
            else:
                ei_limits = self.ei_limits
            # Override exists for MARI G-chopper which doesn't work at the
            # general default Ei
            default_ei = float(cfg.get("defaults", {}).get("e_init", self.default_ei))
            specs[name] = ConfigurationSpec(
                name=name,
                energies=energies,
                frequency_combinations=self.frequency_combinations,
                allowed_frequencies=self.allowed_frequencies,
                ei_limits=ei_limits,
                default_ei=default_ei,
                default_frequencies=self.default_frequencies,
                edge_points=self._parse_edge_points(cfg.get("edges", [])),
                frequency_settings=self.frequency_settings,
            )
        return specs


    def _parse_edge_points(
        self, raw_edges: list[dict]
    ) -> list[tuple[float, tuple[int, ...]]]:
        """Flatten YAML edge declarations into concrete (energy, frequencies) test points.

        Parameters
        ----------
        raw_edges
            List of edge mappings from ``cases.yaml``. Each mapping specifies chopper
            frequency values and a list of incident energies straddling a cutoff:
            ``[{'chopper_frequency': 120, 'e_init': [45.988, 46.08]}, ...]``.

        Returns
        -------
        list[tuple[float, tuple[int, ...]]]
            Flat list of ``(energy, (f0, ...))`` points ready for sampling:
            ``[(45.988, (120,)), (46.08, (120,)), ...]``.
        """
        points = []
        for edge in raw_edges:
            freqs = tuple(int(edge[k]) for k in self.frequency_settings)
            points.extend((float(energy), freqs) for energy in edge["e_init"])
        return points


