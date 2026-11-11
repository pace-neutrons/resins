"""Tests of the Lagrange resolution models (AbINS_v1 and AbINS_v2)."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from resins.instrument import Instrument
from resins.models.polynomial import DiscontinuousPolynomialModel1D, FlooredPolynomialModel1D

CU220_POLY = np.polynomial.Polynomial([-3.5961e-2, 2.156e-2, 7.6987e-5])  # 2 sigma, meV
MONOCHROMATORS = ['Cu(220)', 'Cu(331)', 'Si(311)', 'Si(111)']


def get_model(version=None, monochromator='Cu(220)'):
    instrument = Instrument.from_default('Lagrange')
    if version is None:
        return instrument.get_resolution_function(monochromator=monochromator)
    return instrument.get_resolution_function(version, monochromator=monochromator)


def sigma(model, energies) -> np.ndarray:
    return model.get_characteristics(np.asarray(energies, dtype=float)[:, None])['sigma']


# AbINS_v2

def test_v2_cu220_floor_and_polynomial():
    model = get_model('AbINS_v2')
    result = sigma(model, [0, 1, 24.9, 25.1, 30, 40])
    expected_40 = CU220_POLY(40.) / 2
    assert_allclose(result[:5], 0.4)
    assert_allclose(result[5], expected_40)
    assert expected_40 > 0.4


def test_v2_cu220_continuous_at_25_mev():
    below, above = sigma(get_model('AbINS_v2'), [24.999999, 25.000001])
    assert_allclose(below, above)
    assert_allclose(below, 0.4)


@pytest.mark.parametrize('monochromator', ['Cu(331)', 'Si(311)', 'Si(111)'])
def test_v2_equals_v1_for_other_monochromators(monochromator):
    energies = np.linspace(0, 500, 501)
    v1 = sigma(get_model('AbINS_v1', monochromator), energies)
    v2 = sigma(get_model('AbINS_v2', monochromator), energies)
    assert_allclose(v2, v1, rtol=1e-12)


@pytest.mark.parametrize('version', [None, 'AbINS'])
@pytest.mark.parametrize('monochromator', MONOCHROMATORS)
def test_default_and_alias_resolve_to_v2(version, monochromator):
    model = get_model(version, monochromator)
    assert isinstance(model, FlooredPolynomialModel1D)
    energies = np.linspace(0, 100, 101)
    assert_allclose(sigma(model, energies), sigma(get_model('AbINS_v2', monochromator), energies))


# AbINS_v1 (deprecated): documents the existing behaviour

def test_v1_is_discontinuous_polynomial():
    assert isinstance(get_model('AbINS_v1'), DiscontinuousPolynomialModel1D)


def test_v1_cu220_cutoff_formula():
    result = sigma(get_model('AbINS_v1'), [10, 24.9, 25.1, 30])
    assert_allclose(result[:2], 0.4)
    assert_allclose(result[2:], CU220_POLY([25.1, 30.]) / 2)
    assert_allclose(result[3], 0.340, atol=5e-4)
    assert result[2] < 0.4  # the step at 25 meV


@pytest.mark.parametrize('energy', [0., 1., 1.5])
def test_v1_cu220_raises_below_1_6_mev(energy):
    """Known v1 limitation: the polynomial is not positive there."""
    with pytest.raises(AssertionError):
        sigma(get_model('AbINS_v1'), [energy])
