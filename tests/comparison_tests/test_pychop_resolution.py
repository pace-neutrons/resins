"""Compare the resins PyChop models end to end with committed PyChop reference resolutions.

For each reference case the model is built through the public
``Instrument.get_resolution_function``, and the recorded outcome is asserted: a resolution
within tolerance, a no-transmission error, or an input rejected as invalid.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from resins.instrument import Instrument
from resins.models.model_base import InvalidInputError
from resins.models.pychop import NoTransmissionError

from ._reference import (
    NO_TRANSMISSION,
    REJECTED_EI,
    REJECTED_FREQUENCY,
    VALUE,
    case_params,
    pychop_reference_files,
    resolve,
)

SIGMA2FWHM = 2 * np.sqrt(2 * np.log(2))


def get_model(ref, i: int):
    """Build the resins model for case ``i`` of ``ref`` through the public API."""
    case = ref.cases[i]
    instrument = Instrument.from_default(ref.instrument)
    return instrument.get_resolution_function(
        ref.model_version,
        chopper_package=str(case["configuration"]),
        e_init=float(case["e_init"]),
        **ref.frequency_kwargs(i),
    )


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files=pychop_reference_files(),
        scope="resolution",
        outcomes=(VALUE,),
    ),
)
def test_resolution(param):
    ref, i = resolve(param)
    case = ref.cases[i]
    model = get_model(ref, i)

    instrument = Instrument.from_default(ref.instrument)
    data = instrument.get_model_data(
        ref.model_version, chopper_package=str(case["configuration"])
    )
    energy_transfers, resolution = type(model)._precompute_resolution(
        data, float(case["e_init"]), ref.frequencies(i)
    )
    expected = ref["resolution"][i]

    assert len(energy_transfers) == len(expected)
    assert_allclose(resolution, expected / SIGMA2FWHM, rtol=1e-7)

    # The fitted polynomial is built from these values, so the model's own characteristics
    # follow the reference resolution closely at the fit points too.
    fitted = model.get_characteristics(energy_transfers[:, None])["sigma"]
    assert_allclose(fitted, expected / SIGMA2FWHM, rtol=0.1, atol=1e-3 * expected.max())


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files=pychop_reference_files(),
        scope="resolution",
        outcomes=(NO_TRANSMISSION,),
    ),
)
def test_no_transmission(param):
    ref, i = resolve(param)
    with pytest.raises(NoTransmissionError):
        get_model(ref, i)


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files=pychop_reference_files(),
        scope="resolution",
        outcomes=(REJECTED_EI, REJECTED_FREQUENCY),
    ),
)
def test_rejected_input(param):
    ref, i = resolve(param)
    # The reason column is "<side>:<setting>"; the setting is named in the error message.
    setting = str(ref.cases[i]["reason"]).split(":", 1)[1]

    with pytest.raises(InvalidInputError, match=f'"{setting}" setting'):
        get_model(ref, i)
