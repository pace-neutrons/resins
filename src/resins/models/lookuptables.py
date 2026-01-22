"""
A model based on interpolated lookup tables

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resolution_functions.instrument.Instrument.get_resolution_function` method.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import ClassVar, TYPE_CHECKING

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import RegularGridInterpolator

from .model_base import InstrumentModel, ModelData
from .mixins import SimpleBroaden1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ScaledTabulatedModelData(ModelData):
    """
    Data for the `ScaledTabulatedModel` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `ScaledTabulatedModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    npz
        Relative path from Instrument yaml files to lookup table file
    restrictions
    defaults
    """

    npz: str


class ScaledTabulatedModel(SimpleBroaden1DMixin, InstrumentModel):
    """
    Model using a lookup table to model a 1D :term:`instrument`.

    This allows non-Gaussian shapes to be produced. For smooth interpolation
    and data efficiency, the x-axis is scaled in proportion to an approximated
    standard deviation. This standard deviation is fitted to a polynomial
    function of energy.

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
    npz
        The .npz file containing the model data
    citation
    """

    input = ("energy_transfer",)

    data_class: ClassVar[type[ScaledTabulatedModelData]] = ScaledTabulatedModelData

    def __init__(self, model_data: ScaledTabulatedModelData, **_):
        super().__init__(model_data)
        self.data = np.load(
            importlib.resources.files("resins.instrument_data") / model_data.npz
        )

        self.polynomial = Polynomial(
            coef=self.data["coef"],
            domain=self.data["domain"],
            window=self.data["window"],
        )
        self._interp = RegularGridInterpolator(
            (self.data["energy_transfer"], self.data["kernel_energies"]),
            self.data["table"],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def get_characteristics(
        self, omega_q: Float[np.ndarray, "energy_transfer dimension=1"]
    ) -> dict[str, Float[np.ndarray, "sigma"]]:
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
        return {"sigma": self.polynomial(omega_q[:, 0])}

    def get_kernel(
        self,
        points: Float[np.ndarray, "sample dimension=1"],
        mesh: Float[np.ndarray, "mesh"],
    ) -> Float[np.ndarray, "sample mesh"]:
        assert len(omega_q.shape) == 2 and omega_q.shape[1] == 1
        energy = omega_q

        scale_factors = self.polynomial(energy)
        scaled_x_values = mesh / scale_factors

        # Clip lookup energies to known maximum; width scaling should give a
        # reasonable extrapolation from there
        energy = np.minimum(energy, max(self.data["energy_transfer"]))

        energy_expanded = np.meshgrid(energy[:, None], mesh, indexing="ij")[0]
        lookup_mesh = np.stack([energy_expanded, scaled_x_values], axis=-1)
        interp_kernels = self._interp(lookup_mesh) / scale_factors
        return interp_kernels

    def get_peak(
        self,
        points: Float[np.ndarray, "sample dimension=1"],
        mesh: Float[np.ndarray, "mesh"],
    ) -> Float[np.ndarray, "sample mesh"]:
        shifted_meshes = [mesh - energy for energy in omega_q[:, 0]]

        shifted_kernels = [
            self.get_kernel(np.array([point]), shifted_mesh)
            for point, shifted_mesh in zip(points, shifted_meshes)
        ]

        return np.array(np.vstack(shifted_kernels))
