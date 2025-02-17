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
                   mesh: Float[np.ndarray, 'mesh'],
                   omega_q: Float[np.ndarray, 'sample dimension=1']
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Gaussian kernel on the provided `mesh` at each value of the `omega_q` energy
        transfer.

        Parameters
        ----------
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `omega_q` transfer space of interest.
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.

        Returns
        -------
        kernel
            The Gaussian kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer.
        """
        new_mesh = np.zeros((len(omega_q), len(mesh)))
        new_mesh[:, :] = mesh

        sigma = self.get_characteristics(omega_q)['sigma']
        return norm.pdf(new_mesh, loc=omega_q, scale=sigma[:, np.newaxis])


class SimpleConvolve1DMixin:
    def convolve(self: InstrumentModel,
                 mesh: Float[np.ndarray, 'mesh'],
                 omega_q: Float[np.ndarray, 'sample dimension=1'],
                 data: Float[np.ndarray, 'data']
                 ) -> Float[np.ndarray, 'spectrum']:
        """
        Broadens the `data` on the full `mesh` using the straightforward scheme.

        Parameters
        ----------
        mesh
            The mesh to use for the broadening. This is a 1D array which *must* span the entire
            `omega_q` space of interest.
        omega_q
            The independent variable (energy transfer or momentum scalar) whose `data` to broaden.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of w/Q values
            for which there is `data`. Therefore, the ``sample`` dimension *must* match the length
            of the `data` array.
        data
            The intensities at the `omega_q` points.

        Returns
        -------
        spectrum
            The broadened spectrum.
        """
        kernels = self.get_kernel(mesh, omega_q)
        return np.dot(kernels.T, data)
