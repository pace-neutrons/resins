import numpy as np
import pytest
from numpy.testing import assert_allclose

from resins.models import MODELS
from resins.models.polynomial import (
    FlooredPolynomialModel1D,
    FlooredPolynomialModelData,
    PolynomialModel1D,
    PolynomialModelData,
)

FIT = [-1., 1.]  # sigma = E - 1, negative below 1 meV


def floored(fit=FIT, minimum_sigma=0.5) -> FlooredPolynomialModel1D:
    return FlooredPolynomialModel1D(FlooredPolynomialModelData(
        function='floored_polynomial', citation=[''], defaults={}, restrictions={},
        fit=fit, minimum_sigma=minimum_sigma,
    ))


def sigma(model, energies) -> np.ndarray:
    return model.get_characteristics(np.asarray(energies, dtype=float)[:, None])['sigma']


def test_registered():
    assert MODELS['floored_polynomial'] is FlooredPolynomialModel1D


def test_clamps_below_floor_and_keeps_values_above():
    # Polynomial is E - 1: 0.2 and 1.0 are below/at the 1.5 floor, 3.0 is above.
    result = sigma(floored(minimum_sigma=1.5), [1.2, 2.0, 3.0, 10.0])
    assert_allclose(result, [1.5, 1.5, 2.0, 9.0])


def test_floor_below_polynomial_matches_polynomial_1d():
    energies = np.linspace(2, 50, 25)
    plain = PolynomialModel1D(PolynomialModelData(
        function='polynomial_1d', citation=[''], defaults={}, restrictions={}, fit=FIT))
    assert_allclose(sigma(floored(minimum_sigma=0.0), energies), sigma(plain, energies))


def test_missing_minimum_sigma_rejected():
    with pytest.raises(TypeError):
        FlooredPolynomialModelData(
            function='floored_polynomial', citation=[''], defaults={}, restrictions={}, fit=FIT)


def test_negative_polynomial_does_not_raise():
    result = sigma(floored(minimum_sigma=0.5), [0.0, 0.5])
    assert_allclose(result, [0.5, 0.5])
