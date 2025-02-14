from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from jaxtyping import Float
    from .model_base import InstrumentModel


class GaussianKernel1DMixin:
    """
    A mixin providing the implementation for the Gaussian kernel ``get_kernel`` method.

    Implements `resolution_functions.models.model_base.InstrumentModel.get_kernel` method of models
    whose broadening can be represented by a 1D Gaussian distribution. Any model that satisfies this
    condition should inherit the ``get_kernel`` method from this mixin instead of writing its own
    implementation.

    Technically, any model that implements the
    `resolution_functions.models.model_base.InstrumentModel.get_characteristics` method and which
    returns the ``sigma`` parameter in its dictionary can use this mixin to inherit the Gaussian
    ``get_kernel`` method. However, it is recommended that only models that actually model a
    Gaussian kernel should use this mixin.
    """
    def get_kernel(self: InstrumentModel,
                   mesh: Float[np.ndarray, 'energy_mesh'],
                   omega_q: Float[np.ndarray, 'energy_transfer dimension=1']
                   ) -> Float[np.ndarray, 'energy_transfer energy_mesh']:
        """
        Computes the Gaussian kernel on the provided `mesh` at each value of the `omega_q` energy
        transfer.

        Parameters
        ----------
        mesh
            The mesh on which to evaluate the kernel.
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.

        Returns
        -------
        kernel
            The Gaussian kernel as given by this model, computed on the `mesh` and for each value
            of `energy_transfer`.
        """
        new_mesh = np.zeros((len(omega_q), len(mesh)))
        new_mesh[:, :] = mesh

        sigma = self.get_characteristics(omega_q)['sigma']
        return norm.pdf(new_mesh, scale=sigma[:, np.newaxis])
