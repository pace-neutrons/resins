import itertools
import warnings

import numpy as np
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_allclose
import pytest

import mantid
from abins.instruments.pychop import PyChopInstrument

from resolution_functions.instrument import Instrument
from resolution_functions.models.pychop import NoTransmissionError

WAVENUMBER_TO_MEV = 0.12398419843320028
MEV_TO_WAVENUMBER = 1 / WAVENUMBER_TO_MEV

INSTRUMENTS = [[('MAPS', 'MAPS')], [('MARI', 'MARI')], [('MERLIN', 'MERLIN')]]
INSTRUMENT_CONFIGURATIONS = [['A', 'S'], ['A', 'R', 'S'], ['S']]
ENERGIES = np.arange(50, 500, 10)
CHOPPER_FREQUENCIES = np.arange(50, 601, 50)


def get_instrument_matrix(instruments, instrument_configurations):
    instrument_matrix, instrument_ids = [], []
    for instr, configs in zip(instruments, instrument_configurations):
        lst = list(itertools.product(instr, configs))
        instrument_matrix.extend(lst)
        instrument_ids.extend([f'{i[0]}_{s}' for i, s in lst])

    return instrument_matrix, instrument_ids


def get_ef_matrix(energies, chopper_frequencies):
    ef_matrix = list(itertools.product(energies, chopper_frequencies))
    ef_ids = [f'ei={e},f={f}' for e, f in ef_matrix]
    return ef_matrix, ef_ids


def _abins_rf_2d(name, version, config):
    abins = PyChopInstrument(name=name, setting=config)
    rf = Instrument.from_default(name, version)

    return abins, rf, config


INSTRUMENT_MATRIX, INSTRUMENT_IDS = get_instrument_matrix(INSTRUMENTS, INSTRUMENT_CONFIGURATIONS)


@pytest.fixture(scope="module", params=INSTRUMENT_MATRIX, ids=INSTRUMENT_IDS)
def abins_rf_2d(request):
    (name, version), config = request.param
    return _abins_rf_2d(name, version, config)


EF_MATRIX, EF_IDS = get_ef_matrix(ENERGIES, CHOPPER_FREQUENCIES)


@pytest.mark.parametrize('matrix', EF_MATRIX, ids=EF_IDS)
def test_against_abins(matrix, abins_rf_2d,):
    abins, rf_2d, config = abins_rf_2d
    _test_against_abins(abins, rf_2d, config, matrix)


def _test_against_abins(abins, rf_2d, config, matrix):
    energy, chopper_frequency = matrix
    frequencies = np.linspace(0, energy, 1000)

    abins.set_incident_energy(energy, 'meV')
    abins._chopper_frequency = chopper_frequency
    abins._pychop_instrument.chopper_system.setFrequency(chopper_frequency)
    abins._pychop_instrument.frequency = chopper_frequency

    abins._polyfits = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="PyChop: tchop\(\): No transmission.*")
        try:
            expected = abins.calculate_sigma(frequencies * MEV_TO_WAVENUMBER) * WAVENUMBER_TO_MEV

        except Warning:
            # Make sure this library agrees it is a no-transmission situation.
            with pytest.raises(NoTransmissionError):
                rf_2d.get_resolution_function('PyChop_fit',
                                              chopper_package=config,
                                              e_init=energy,
                                              chopper_frequency=chopper_frequency
)
            return

    rf = rf_2d.get_resolution_function('PyChop_fit', chopper_package=config, e_init=energy,
                                       chopper_frequency=chopper_frequency)
    actual = rf.get_characteristics(frequencies[:, None])["sigma"]
    assert_allclose(actual, expected, rtol=1e-5)


@pytest.fixture(scope='module',
                params=[(('MARI', 'MARI'), 'G'), (('MERLIN', 'MERLIN'), 'G')],
                ids=['MARI_G', 'MERLIN_G'])
def g_choppers(request):
    (name, version), config = request.param
    return _abins_rf_2d(name, version, config)


ef, ids = get_ef_matrix(np.arange(10, 181, 5), CHOPPER_FREQUENCIES)


@pytest.mark.parametrize('matrix', ef, ids=ids)
def test_g_choppers_against_abins(matrix, g_choppers,):
    abins, rf_2d, config = g_choppers
    _test_against_abins(abins, rf_2d, config, matrix)

