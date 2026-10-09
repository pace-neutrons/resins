"""Check that the reference data covers every compared model version and configuration.

This guards against adding an instrument, model version or configuration to resins without
regenerating the reference data (see ``dev/reference/README.md``), and checks that the data
meets the sampling rules it was generated with.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from resins.instrument import Instrument

from ._reference import (
    NO_TRANSMISSION,
    REJECTED_EI,
    REJECTED_FREQUENCY,
    VALUE,
    load,
    pychop_stems,
    stems,
)

# Configurations that Abins' own instrument classes do not support, so are not compared.
ABINS_UNSUPPORTED = {"MAPS": {"B"}, "MARI": {"B", "C"}}

# Largest relative gap between neighbouring cases with different transmission, i.e. the points
# just either side of an edge (offset 1e-3 each, and rounding to five digits).
MAX_EDGE_GAP = 2.3e-3


def resins_pychop_models() -> set[tuple[str, str]]:
    """(instrument, model version) of every PyChop model in resins."""
    found = set()
    for name in Instrument.available_instruments():
        instrument = Instrument.from_default(name)
        for model in instrument.available_unique_models:
            if instrument.get_model_data(model).function.startswith("pychop_fit"):
                found.add((name, model))
    return found


def chopper_packages(instrument: str, model: str) -> list[str]:
    inst = Instrument.from_default(instrument)
    return inst.possible_options_for_model_and_configuration(model, "chopper_package")


def evaluated_by_configuration(ref) -> dict[str, np.ndarray]:
    cases = ref.cases
    return {name: cases[cases["configuration"] == name] for name in np.unique(cases["configuration"])}


def test_every_pychop_model_has_reference_data():
    in_resins = resins_pychop_models()
    in_data = {(load(s).instrument, load(s).model_version) for s in pychop_stems()}

    assert not in_resins - in_data, f"PyChop models without reference data: {sorted(in_resins - in_data)}"
    assert not in_data - in_resins, f"reference data for models not in resins: {sorted(in_data - in_resins)}"


def test_one_file_per_instrument_and_model_version():
    keys = [(load(s).instrument, load(s).model_version) for s in pychop_stems()]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("stem", pychop_stems())
def test_every_configuration_has_value_cases(stem):
    ref = load(stem)
    cases = ref.cases
    expected = set(chopper_packages(ref.instrument, ref.model_version))
    in_file = {str(c) for c in np.unique(cases["configuration"])}

    missing, extra = sorted(expected - in_file), sorted(in_file - expected)
    assert not missing, f"configurations without reference cases: {missing}"
    assert not extra, f"reference cases for configurations not in resins: {extra}"
    for configuration in expected:
        selected = cases[(cases["configuration"] == configuration) & (cases["outcome"] == VALUE)]
        assert len(selected) > 0, f"no value case for configuration {configuration}"


def test_abins_pychop_covers_supported_configurations():
    ref = load("abins/pychop")
    cases = ref.cases
    for instrument, model in ref.metadata["model_versions"].items():
        selected = cases[cases["instrument"] == instrument]
        expected = set(chopper_packages(instrument, model)) - ABINS_UNSUPPORTED.get(instrument, set())
        in_file = {str(c) for c in np.unique(selected["configuration"])}
        assert in_file == expected, f"{instrument}: Abins configurations differ from resins"
        for configuration in expected:
            values = selected[(selected["configuration"] == configuration) & (selected["outcome"] == VALUE)]
            assert len(values) > 0, f"{instrument} {configuration}: no value case"


def test_abins_lagrange_covers_every_monochromator():
    ref = load("abins/lagrange")
    expected = set(
        Instrument.from_default("Lagrange").possible_options_for_model_and_configuration(
            "AbINS", "monochromator"
        )
    )
    assert {str(c) for c in np.unique(ref.cases["configuration"])} == expected


@pytest.mark.parametrize("name", ["panther", "tosca"])
def test_abins_single_configuration_models_have_cases(name):
    assert len(load(f"abins/{name}").cases) > 0


@pytest.mark.parametrize("stem", pychop_stems())
def test_incident_energy_dependence_is_sampled(stem):
    ref = load(stem)
    for configuration, cases in evaluated_by_configuration(ref).items():
        values = cases[cases["outcome"] == VALUE]
        assert len(np.unique(values["e_init"])) >= 4, configuration


@pytest.mark.parametrize("stem", pychop_stems())
def test_every_allowed_frequency_is_sampled(stem):
    ref = load(stem)
    instrument = Instrument.from_default(ref.instrument)
    for configuration, cases in evaluated_by_configuration(ref).items():
        data = instrument.get_model_data(ref.model_version, chopper_package=configuration)
        for column, setting in zip(("f0", "f1"), ref.frequency_settings):
            allowed = set(range(*data.restrictions[setting]))
            sampled = {float(f) for f in cases[column]}
            assert allowed <= sampled, f"{configuration} {setting}: missing {sorted(allowed - sampled)}"


@pytest.mark.parametrize("stem", pychop_stems())
def test_transmission_edges_are_sampled_on_both_sides(stem):
    ref = load(stem)
    for configuration, cases in evaluated_by_configuration(ref).items():
        groups = defaultdict(list)
        for case in cases[np.isin(cases["outcome"], (VALUE, NO_TRANSMISSION))]:
            groups[(case["f0"], case["f1"])].append((float(case["e_init"]), int(case["outcome"])))

        for frequencies, points in groups.items():
            points.sort()
            for (e_low, outcome_low), (e_high, outcome_high) in zip(points, points[1:]):
                if outcome_low != outcome_high:
                    gap = e_high / e_low - 1
                    assert gap <= MAX_EDGE_GAP, (
                        f"{configuration} f={frequencies}: transmission changes between "
                        f"{e_low} and {e_high} without points either side of the edge"
                    )


@pytest.mark.parametrize("stem", pychop_stems())
def test_limits_are_sampled_on_both_sides(stem):
    ref = load(stem)
    instrument = Instrument.from_default(ref.instrument)
    for configuration, cases in evaluated_by_configuration(ref).items():
        data = instrument.get_model_data(ref.model_version, chopper_package=configuration)
        low, high = data.restrictions["e_init"]
        rejected = cases[cases["outcome"] == REJECTED_EI]["e_init"]
        accepted = cases[np.isin(cases["outcome"], (VALUE, NO_TRANSMISSION))]["e_init"]

        assert (rejected < low).any() and (rejected > high).any(), f"{configuration}: Ei limits"
        assert (accepted >= low).all() and (accepted <= high).all()
        assert (accepted <= high).any() and accepted.max() >= high * (1 - 1.1e-3)

        for column, setting in zip(("f0", "f1"), ref.frequency_settings):
            allowed = range(*data.restrictions[setting])
            rejected_f = cases[cases["outcome"] == REJECTED_FREQUENCY][column]
            assert (rejected_f > allowed[-1]).any(), f"{configuration} {setting}: no frequency above maximum"
            assert any(f not in allowed and f < allowed[-1] for f in rejected_f), (
                f"{configuration} {setting}: no off-grid frequency"
            )
