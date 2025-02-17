"""
The AbINS :term:`model` of the PANTHER :term:`instrument`.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
try:
    from warnings import deprecated
except ImportError:
    from typing_extensions import deprecated

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .model_base import InstrumentModel, ModelData, DEPRECATION_MSG
from .mixins import GaussianKernel1DMixin, SimpleConvolve1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PantherAbINSModelData(ModelData):
    """
    Data for the `PantherAbINSModel` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    abs
        Polynomial coefficients for the energy transfer (frequencies) polynomial.
    ei_dependence
        Polynomial coefficients for the initial energy polynomial.
    ei_energy_product
        Polynomial coefficients for the product of initial energy and energy transfer (frequencies)
        polynomial.
    restrictions
    defaults
    """
    abs: list[float]
    ei_dependence: list[float]
    ei_energy_product: list[float]


class PantherAbINSModel(GaussianKernel1DMixin, SimpleConvolve1DMixin, InstrumentModel):
    """
    Model for the PANTHER :term:`instrument` originating from the AbINS software.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output model being a Gaussian. This is done by fitting three power-series polynomials (see
    `numpy.polynomial.polynomial.Polynomial`) to the resolution curve, where the result of the sum
    of the polynomials is the width (sigma) of the Gaussian. Each polynomial can be of any degree
    and is given via the `resolution_functions.models.polynomial.PolynomialModelData`.

    The :term:`resolution` is modelled as::

        resolution = Polynomial(model_data.abs)(frequencies) +
                     Polynomial(model_data.ei_dependence)(e_init) +
                     Polynomial(model_data.ei_energy_product)(e_init * frequencies)

    where ``e_init`` is the initial energy, ``frequencies`` is the energy transfer, and
    ``model_data`` is an instance of `PantherAbINSModelData`.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    e_init
        The incident energy in meV.

    Attributes
    ----------
    input
        The input that the ``__call__`` method expects.
    data_class
        Reference to the `PantherAbINSModelData` type.
    abs : numpy.polynomial.polynomial.Polynomial
        The energy transfer polynomial.
    ei_dependence : float
        The `e_init` contribution to the resolution.
    ei_energy_product : numpy.polynomial.polynomial.Polynomial
        The energy transfer and `e_init` product polynomial.
    citation
    """
    input = ('energy_transfer',)

    data_class = PantherAbINSModelData

    def __init__(self, model_data: PantherAbINSModelData, e_init: float, **_):
        super().__init__(model_data)

        self.e_init = e_init
        self.abs = Polynomial(model_data.abs)
        self.ei_dependence = Polynomial(model_data.ei_dependence)(e_init)
        self.ei_energy_product = Polynomial(model_data.ei_energy_product)

    def get_characteristics(self, omega_q: Float[np.ndarray, 'energy_transfer dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sigma']]:
        """
        Computes the broadening width at each value of energy transfer given by `omega_q`.

        The model approximates the broadening using the Gaussian distribution, so the returned
        widths are in the form of the standard deviation (sigma).

        Parameters
        ----------
        omega_q
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Gaussian width as sigma in meV.
        """
        omega_q = omega_q[:, 0]
        resolution = (self.abs(omega_q) +
                      self.ei_dependence +
                      self.ei_energy_product(self.e_init * omega_q))
        return {'sigma': resolution / (2 * np.sqrt(2 * np.log(2)))}

    @deprecated(DEPRECATION_MSG)
    def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs) -> Float[np.ndarray, 'sigma']:
        """
        Evaluates the model at given energy transfer values (`frequencies`), returning the
        corresponding Gaussian widths (sigma).

        Parameters
        ----------
        frequencies
            Energy transfer in meV. The frequencies at which to return widths.

        Returns
        -------
        sigma
            The Gaussian widths at `frequencies` as predicted by this model.
        """
        return self.get_characteristics(frequencies)['sigma']
