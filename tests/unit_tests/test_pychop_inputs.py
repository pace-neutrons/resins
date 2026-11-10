"""Validation of invalid ``e_init`` and chopper frequencies by the PyChop models.

These need only resins model data, not a reference implementation.
"""
import numpy as np
import pytest

from resins.instrument import Instrument
from resins.models.model_base import InvalidInputError
from resins.models.pychop import PyChopModelCNCS, PyChopModelFermi, PyChopModelLET


@pytest.fixture(scope="module")
def mari_data():
    return Instrument.from_default("MARI").get_model_data("PyChop_fit")


@pytest.fixture(scope="module")
def cncs_data():
    return Instrument.from_default("CNCS").get_model_data("PyChop_fit")


@pytest.fixture(scope="module")
def let_data():
    return Instrument.from_default("LET").get_model_data("PyChop_fit")


@pytest.mark.parametrize(
    "chopper_frequency",
    [
        49.99999,
        -0.048,
        -np.inf,
        600.00017,
        np.inf,
        13554,
        np.nan,
        50.5,
        57.5,
        500.0000001,
        480,
    ],
)
def test_fermi_invalid_chopper_frequency(chopper_frequency, mari_data):
    with pytest.raises(
        InvalidInputError, match='The provided value for the "chopper_frequency" setting'
    ):
        PyChopModelFermi(mari_data, chopper_frequency=chopper_frequency)


@pytest.mark.parametrize("e_init", [-5, -0.00048, -np.inf, 2000.1, np.inf, 13554.1654, np.nan])
def test_fermi_invalid_e_init(e_init, mari_data):
    with pytest.raises(InvalidInputError, match='The provided value for the "e_init" setting'):
        PyChopModelFermi(mari_data, e_init=e_init)


@pytest.mark.parametrize(
    "chopper_frequency,invalid_setting",
    [
        ([59.99999, 60], "resolution_disk_frequency"),
        ([-0.048] * 2, "resolution_disk_frequency"),
        ([-np.inf, 0], "resolution_disk_frequency"),
        ([120, 300.00017], "fermi_frequency"),
        ([np.inf, np.inf], "resolution_disk_frequency"),
        ([300, np.nan], "fermi_frequency"),
        ([60.5, 60], "resolution_disk_frequency"),
        ([180, 67.5], "fermi_frequency"),
        ([600, 600], "resolution_disk_frequency"),
        ([135, 135], "resolution_disk_frequency"),
    ],
)
def test_cncs_invalid_chopper_frequency(chopper_frequency, invalid_setting, cncs_data):
    with pytest.raises(
        InvalidInputError, match=f'The provided value for the "{invalid_setting}" setting'
    ):
        PyChopModelCNCS(
            cncs_data,
            resolution_disk_frequency=chopper_frequency[0],
            fermi_frequency=chopper_frequency[1],
        )


@pytest.mark.parametrize("e_init", [-5, -0.00048, -np.inf, 80.1, np.inf, 13554.1654, np.nan])
def test_cncs_invalid_e_init(e_init, cncs_data):
    with pytest.raises(InvalidInputError, match='The provided value for the "e_init" setting'):
        PyChopModelCNCS(cncs_data, e_init=e_init)


@pytest.mark.parametrize(
    "chopper_frequency,invalid_setting",
    [
        ([59.99999, 60], "resolution_frequency"),
        ([-0.048] * 2, "resolution_frequency"),
        ([-np.inf, 0], "resolution_frequency"),
        ([120, 300.00017], "pulse_remover_frequency"),
        ([np.inf, np.inf], "resolution_frequency"),
        ([300, np.nan], "pulse_remover_frequency"),
        ([60.5, 60], "resolution_frequency"),
        ([180, 67.5], "pulse_remover_frequency"),
        ([600, 600], "resolution_frequency"),
        ([135, 135], "resolution_frequency"),
    ],
)
def test_let_invalid_chopper_frequency(chopper_frequency, invalid_setting, let_data):
    with pytest.raises(
        InvalidInputError, match=f'The provided value for the "{invalid_setting}" setting'
    ):
        PyChopModelLET(
            let_data,
            resolution_frequency=chopper_frequency[0],
            pulse_remover_frequency=chopper_frequency[1],
        )


@pytest.mark.parametrize("e_init", [-5, -0.00048, -np.inf, 30.0001, np.inf, 13554.1654, np.nan])
def test_let_invalid_e_init(e_init, let_data):
    with pytest.raises(InvalidInputError, match='The provided value for the "e_init" setting'):
        PyChopModelLET(let_data, e_init=e_init)
