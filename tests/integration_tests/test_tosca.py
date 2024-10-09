import numpy as np
import pytest

import mantid
from abins.instruments.toscainstrument import ToscaInstrument

from resolution_functions.instruments.indirect_instruments import TOSCA


WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV


@pytest.fixture
def tosca_rf():
    return TOSCA.from_default('TOSCA')


@pytest.fixture
def tosca_abins():
    return ToscaInstrument()


@pytest.fixture
def tosca_abins_resolution_function_backward(tosca_rf):
    return tosca_rf.get_resolution_function('AbINS', 'Backward')


@pytest.fixture
def tosca_abins_resolution_function_forward(tosca_rf):
    return tosca_rf.get_resolution_function('AbINS', 'Forward')


@pytest.mark.parametrize('frequencies',
                         [np.arange(0, 400, 10), np.arange(100, 1000, 50)])
def test_tosca_against_abins(frequencies,
                             tosca_abins_resolution_function_backward,
                             tosca_abins_resolution_function_forward,
                             tosca_abins
                             ):
    backward = tosca_abins_resolution_function_backward(frequencies)
    forward = tosca_abins_resolution_function_forward(frequencies)
    actual = (backward + forward) * 0.5

    expected = tosca_abins.get_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

    assert np.allclose(actual, expected)
