import numpy as np
import pytest

import mantid
from abins.instruments.toscainstrument import ToscaInstrument

from resolution_functions.instrument import Instrument


WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV


@pytest.fixture(scope="module")
def tosca_rf():
    return Instrument.from_default('TOSCA', 'TOSCA')


@pytest.fixture(scope="module")
def tosca_abins():
    return ToscaInstrument()


@pytest.fixture(scope="module")
def tosca_abins_resolution_function_backward(tosca_rf):
    return tosca_rf.get_resolution_function('AbINS')


@pytest.fixture(scope="module")
def tosca_abins_resolution_function_forward(tosca_rf):
    return tosca_rf.get_resolution_function('AbINS')


@pytest.mark.parametrize('frequencies',
                         [np.arange(0, 400, 10), np.arange(100, 1000, 50)])
def test_tosca_against_abins(frequencies,
                             tosca_abins_resolution_function_backward,
                             tosca_abins_resolution_function_forward,
                             tosca_abins
                             ):
    backward = tosca_abins_resolution_function_backward.get_characteristics(
        frequencies[:, None]
    )["sigma"]
    forward = tosca_abins_resolution_function_forward.get_characteristics(
        frequencies[:, None]
    )["sigma"]
    actual = (backward + forward) * 0.5

    expected = tosca_abins.get_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    np.allclose(actual, expected)
