from enum import Enum, auto

import numpy as np
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.lagrangeinstrument import LagrangeInstrument

from resins.instrument import Instrument

WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV
LAGRANGE_CONFIGURATIONS = ['Cu(220)', 'Cu(331)', 'Si(311)', 'Si(111)']


@pytest.fixture(scope="module")
def lagrange_rf():
    return Instrument.from_default('Lagrange', 'Lagrange')


@pytest.fixture(scope="module", params=LAGRANGE_CONFIGURATIONS)
def lagrange_abins_plus_rf_abins_resolution_function(lagrange_rf, request):
    abins = LagrangeInstrument(setting=request.param + ' (Lagrange)')
    rf = lagrange_rf.get_resolution_function('AbINS', monochromator=request.param)
    return abins, rf

class FreqRange(Enum):
    LOW = auto()
    HIGH = auto()

@pytest.mark.parametrize(
    "frequencies, label",
    [
        pytest.param(
            np.arange(100, 1000, 50), FreqRange.HIGH,
            id="high frequencies: 100:1000:50",
        ),
        pytest.param(
            np.arange(10, 100, 10), FreqRange.LOW, id="low frequencies: 10:100:10"
        ),
    ],
)
def test_lagrange_against_abins(
        frequencies, label, lagrange_abins_plus_rf_abins_resolution_function
):
    abins, rf = lagrange_abins_plus_rf_abins_resolution_function

    if abins.get_setting() == "Cu(220) (Lagrange)" and label == FreqRange.LOW:
        pytest.xfail(reason="LAGRANGE low frequencies to be fixed in Abins")

    actual = rf.get_characteristics(frequencies[:, None])["sigma"]
    expected = abins.get_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    assert_allclose(actual, expected, rtol=1e-5)
