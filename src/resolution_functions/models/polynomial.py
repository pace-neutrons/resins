"""
Collection of models based off polynomials.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from .model_base import InstrumentModel, ModelData
from .mixins import GaussianKernel1DMixin, SimpleConvolve1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class PolynomialModelData(ModelData):
    """
    Data for the `PolynomialModel1D` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PolynomialModel1D`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    fit
        Polynomial coefficients.
    restrictions
    defaults
    """
    fit: list[float]


class PolynomialModel1D(GaussianKernel1DMixin, SimpleConvolve1DMixin, InstrumentModel):
    """
    Model using a 1D polynomial to model an :term:`instrument`.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output :term:`model` being a Gaussian. This is done by fitting a single power-series polynomial
    (see `numpy.polynomial.polynomial.Polynomial`) to the resolution curve, where the result of the
    polynomial is the width (sigma) of the Gaussian. The polynomial can be of any degree and is
    given via the `PolynomialModelData`.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.

    Attributes
    ----------
    input
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `PolynomialModelData` type.
    polynomial : numpy.polynomial.polynomial.Polynomial
        The polynomial representing the resolution function.
    citation
    """
    input = ('energy_transfer',)

    data_class: ClassVar[type[PolynomialModelData]] = PolynomialModelData

    def __init__(self, model_data: PolynomialModelData, **_):
        super().__init__(model_data)
        self.polynomial = Polynomial(model_data.fit)

    def get_characteristics(self, omega_q: Float[np.ndarray, 'energy_transfer dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sigma']]:
        """
        Computes the broadening width at each value of energy transfer (`omega_q`).

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
            The characteristics of the broadening function, i.e. the Gaussian width as sigma.
        """
        return {'sigma': self.polynomial(omega_q[:, 0])}


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class DiscontinuousPolynomialModelData(ModelData):
    """
    Data for the `DiscontinuousPolynomialModel1D` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `DiscontinuousPolynomialModel1D`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    fit
        Polynomial coefficients.
    low_energy_cutoff
        The lower bound (in meV) for the energy transfer (frequencies), below which the ``sigma``
        values are set to the value of `low_energy_resolution`.
    low_energy_resolution
        The value (in meV) to which ``sigma`` is set when the energy transfer is lower than
        `low_energy_cutoff`.
    high_energy_cutoff
        The upper bound (in meV) for the energy transfer (frequencies), above which the ``sigma``
        values are set to the value of `high_energy_resolution`.
    high_energy_resolution
        The value (in meV) to which ``sigma`` is set when the energy transfer is higher than
        `high_energy_cutoff`.
    restrictions
    defaults
    """
    fit: list[float]
    low_energy_cutoff: float = - np.inf
    low_energy_resolution: float = 0.
    high_energy_cutoff: float = np.inf
    high_energy_resolution: float = 0.


class DiscontinuousPolynomialModel1D(GaussianKernel1DMixin, SimpleConvolve1DMixin, InstrumentModel):
    """
    Model using a 1D polynomial to model an :term:`instrument`, but with values above and below
    certain energy transfer set to constant values.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output :term:`model` being a Gaussian. This is done by fitting a single power-series polynomial
    (see `numpy.polynomial.polynomial.Polynomial`) to the resolution curve, where the result of the
    polynomial is the width (sigma) of the Gaussian. The polynomial can be of any degree and is
    given via the `PolynomialModelData`. However, all ``sigma`` values below
    `DiscontinuousPolynomialModelData.low_energy_cutoff` are set to the value of
    `DiscontinuousPolynomialModelData.low_energy_resolution` and similarly all ``sigma`` values
    above `DiscontinuousPolynomialModelData.high_energy_cutoff` are set to the value of
    `DiscontinuousPolynomialModelData.high_energy_resolution`.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.

    Attributes
    ----------
    input
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `DiscontinuousPolynomialModelData` type.
    polynomial : numpy.polynomial.polynomial.Polynomial
        The polynomial representing the resolution function.
    low_energy_cutoff
        The lower bound (in meV) for the energy transfer (frequencies), below which the ``sigma``
        values are set to the value of `low_energy_resolution`.
    low_energy_resolution
        The value (in meV) to which ``sigma`` is set when the energy transfer is lower than
        `low_energy_cutoff`.
    high_energy_cutoff
        The upper bound (in meV) for the energy transfer (frequencies), above which the ``sigma``
        values are set to the value of `high_energy_resolution`.
    high_energy_resolution
        The value (in meV) to which ``sigma`` is set when the energy transfer is higher than
        `high_energy_cutoff`.
    citation
    """
    input = ('energy_transfer',)

    data_class: ClassVar[type[DiscontinuousPolynomialModelData]] = DiscontinuousPolynomialModelData

    def __init__(self, model_data: DiscontinuousPolynomialModelData, **_):
        super().__init__(model_data)

        self.polynomial = Polynomial(model_data.fit)

        self.low_energy_cutoff = model_data.low_energy_cutoff
        self.low_energy_resolution = model_data.low_energy_resolution

        self.high_energy_cutoff = model_data.high_energy_cutoff
        self.high_energy_resolution = model_data.high_energy_resolution

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
        result = self.polynomial(omega_q)

        assert np.all(result > 0)

        result[omega_q < self.low_energy_cutoff] = self.low_energy_resolution
        result[omega_q > self.high_energy_cutoff] = self.high_energy_resolution

        return {'sigma': result * 0.5}
