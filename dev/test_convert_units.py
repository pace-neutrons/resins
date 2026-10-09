import numpy as np
import pint
import pytest
from convert_units import *

ureg = pint.UnitRegistry()
np.random.seed(0)


@pytest.mark.parametrize('coefficients,x',
                         [([2., 4., 8.], 1),
                         ([2., 4., 8.], 2),
                         ([2., 4., 8.], 16),
                         ([2., 4., 8.], 0.651654),
                         ([2., 4., 8.], -1),
                         ([2., 4., 8.], -0.45454),
                         ([2., 4., 8.], -28),
                         ([2., 4., 8.], np.random.random(16)),
                         ([2., 4., 8.], np.random.random(16) * np.linspace(-500, 500, 16)),
                         ([515.132, 0.16161, -1.16554], 1),
                         ([515.132, 0.16161, -1.16554], 2),
                         ([515.132, 0.16161, -1.16554], 16),
                         ([515.132, 0.16161, -1.16554], 0.651654),
                         ([515.132, 0.16161, -1.16554], -1),
                         ([515.132, 0.16161, -1.16554], -0.45454),
                         ([515.132, 0.16161, -1.16554], -28),
                         ([515.132, 0.16161, -1.16554], np.random.random(16)),
                         ([515.132, 0.16161, -1.16554], np.random.random(16) * np.linspace(-500, 500, 16))
                          ])
def test_convert_wavenumber_to_meV_polynomial(coefficients: list[float], x: float):
    mev_coefficients = convert_wavenumber_to_meV_polynomial(coefficients)

    x_mev = (x * ureg.reciprocal_centimeter).to('meV', 'spectroscopy').magnitude
    mev_result = mev_coefficients[0] + mev_coefficients[1] * x_mev + mev_coefficients[2] * x_mev ** 2

    old_result = coefficients[0] + coefficients[1] * x + coefficients[2] * x ** 2
    old_result_converted = (old_result * ureg.reciprocal_centimeter).to('meV', 'spectroscopy').magnitude

    assert np.allclose(mev_result, old_result_converted)
