"""Compare each PyChop component of the resins models with committed PyChop reference values.

Component-level comparisons localise a failure to the moderator, chopper, detector, sample or
geometry, which the end-to-end resolution comparison cannot. The reference values were
generated with Mantid's bundled PyChop (see ``dev/reference``); PyChop is not needed here.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from resins.instrument import Instrument
from resins.models import MODELS
from resins.models.pychop import NoTransmissionError

from ._reference import (
    EVALUATED,
    case_params,
    load,
    pychop_stems,
    resolve,
    row_params,
)

STEMS = pychop_stems()


def model_for(ref, configuration: str):
    """Model data and class of the resins model matching a reference file configuration."""
    instrument = Instrument.from_default(ref.instrument)
    data = instrument.get_model_data(ref.model_version, chopper_package=configuration)
    return data, MODELS[data.function]


@pytest.mark.parametrize("param", row_params(STEMS, "config_names"))
def test_distances(param):
    ref, k = resolve(param)
    data, cls = model_for(ref, str(ref["config_names"][k]))
    expected = ref["distances"][k]

    x0, xa, xm = cls._get_distances(data.choppers)
    assert_allclose([x0, xa, xm], expected[[0, 1, 4]], rtol=1e-12)
    assert_allclose(data.d_chopper_sample, expected[2], rtol=1e-12)
    assert_allclose(data.d_sample_detector, expected[3], rtol=1e-12)


@pytest.mark.parametrize("param", row_params(STEMS, "config_names"))
def test_sample_width_squared(param):
    ref, k = resolve(param)
    data, cls = model_for(ref, str(ref["config_names"][k]))
    expected = ref["sample_width_sq"][k]

    if np.isnan(expected):
        assert data.sample is None
    else:
        assert_allclose(cls._get_sample_width_squared(data.sample), expected)


def _ei_key(param):
    ref, k = resolve(param)
    key = ref["ei_keys"][k]
    return ref, k, str(key["configuration"]), float(key["e_init"])


@pytest.mark.parametrize("param", row_params(STEMS, "ei_keys"))
def test_moderator_width_squared(param):
    ref, k, configuration, e_init = _ei_key(param)
    data, cls = model_for(ref, configuration)

    actual = cls._get_moderator_width_squared(data.moderator, e_init)
    assert_allclose(actual, ref["moderator_width_sq"][k])


@pytest.mark.parametrize("param", row_params(STEMS, "ei_keys"))
def test_moderator_width_squared_analytical(param):
    ref, k, configuration, e_init = _ei_key(param)
    data, cls = model_for(ref, configuration)
    moderator = data.moderator

    actual = cls._get_moderator_width_analytical(
        moderator["type"],
        moderator["parameters"],
        moderator["scaling_function"],
        moderator.get("scaling_parameters"),
        e_init,
    )
    assert_allclose(actual, ref["moderator_width_sq_analytical"][k])


@pytest.mark.parametrize("param", row_params(STEMS, "ei_keys"))
def test_detector_width_squared(param):
    ref, k, configuration, e_init = _ei_key(param)
    data, cls = model_for(ref, configuration)
    expected = ref["detector_width_sq"][k]

    if np.all(np.isnan(expected)):
        assert data.detector is None
        return

    energy_transfers = np.linspace(0, e_init, ref.metadata["n_resolution_points"], endpoint=False)
    actual = cls._get_detector_width_squared(data.detector, energy_transfers, e_init)
    assert_allclose(actual, expected)


@pytest.mark.parametrize("param", case_params(STEMS, "components:chopper", EVALUATED))
def test_chopper_width_squared(param):
    ref, i = resolve(param)
    case = ref.cases[i]
    data, cls = model_for(ref, str(case["configuration"]))
    frequencies = ref.frequencies(i)
    expected = ref["chopper_width_sq"][i]

    if np.isnan(expected[0]):
        with pytest.raises(NoTransmissionError):
            cls._get_chopper_width_squared(data, case["e_init"], frequencies)
        return

    actual = cls._get_chopper_width_squared(data, case["e_init"], frequencies)
    # The widths are of order 1e-10 s^2 (squared times), so the tolerance is relative.
    assert_allclose(actual[0], expected[0], rtol=1e-7)
    if np.isnan(expected[1]):  # Fermi chopper: no second chopper
        assert actual[1] is None
    else:
        assert_allclose(actual[1], expected[1], rtol=1e-7)


def _nonfermi_stems() -> list[str]:
    return [s for s in STEMS if "chop_times" in load(s).arrays]


@pytest.mark.parametrize("param", case_params(_nonfermi_stems(), "components:chop_times", EVALUATED))
def test_chop_times(param):
    ref, i = resolve(param)
    case = ref.cases[i]
    data, cls = model_for(ref, str(case["configuration"]))
    expected = ref["chop_times"][i]  # (first/last chopper, windows, open/close), NaN padded

    actual = cls._get_chop_times(data, case["e_init"], ref.frequencies(i))

    assert len(actual) == 2
    for windows, expected_windows in zip(actual, expected):
        n_expected = int(np.sum(~np.isnan(expected_windows[:, 0])))
        assert len(windows) == n_expected
        assert_allclose(np.array(windows), expected_windows[:n_expected], rtol=0, atol=1e-8)
