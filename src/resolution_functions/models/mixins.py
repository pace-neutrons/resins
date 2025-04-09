"""
Mixins providing generic implementations for
`~resolution_functions.models.model_base.InstrumentModel` methods.

The classes defined here are mixins to be used by specific models via multiple inheritance, allowing
common code to be shared between models. Please note, however, that when doing this, the mixin
**must** be the first base class (i.e. ``class Foo(Mixin, InstrumentModel)``) so that its
implementation of a method overrides the abstract declaration in ``InstrumentModel``.
"""
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
    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Gaussian kernel centered on zero on the provided `mesh` at each (energy
        transfer or momentum scalar) point in `points`.

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array which must encompass 0.

        Returns
        -------
        kernels
            The Gaussian kernel at w/Q point as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, 0.)

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh'],
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Gaussian broadening peak on the provided `mesh` at each (energy transfer or
        momentum scalar) point in `points`.

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the peak. This *must* be a
            Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `points` w\Q space of interest.

        Returns
        -------
        peaks
            The Gaussian peak at each w/Q point in `points` as given by this model, computed on the
            `mesh` and centered on the corresponding w/Q. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, points)

    def _get_kernel(self: InstrumentModel,
                    points: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.
                    ) -> Float[np.ndarray, 'sample mesh']:
        """Computes the kernel using the specified `displacement`."""
        new_mesh = np.zeros((len(points), len(mesh)))
        new_mesh[:, :] = mesh

        sigma = self.get_characteristics(points)['sigma']
        return norm.pdf(new_mesh, loc=displacement, scale=sigma[:, np.newaxis])


class SimpleBroaden1DMixin:
    """
    A mixin providing the most simple implementation for the ``convolve`` method.

    Implements `resolution_functions.models.model_base.InstrumentModel.convolve` method in the
    most simple and basic way - the dot product between the matrix of kernels (obtained from the
    ``get_kernel`` method) and the intensities.

    This implementation should be mostly used as a reference method given that it is correct but
    inefficient. It should be able to work with any model, so it may be used when other
    implementations are unavailable.
    """
    def broaden(self: InstrumentModel,
                points: Float[np.ndarray, 'sample dimension=1'],
                data: Float[np.ndarray, 'data'],
                mesh: Float[np.ndarray, 'mesh'],
                ) -> Float[np.ndarray, 'spectrum']:
        """
        Broadens the `data` on the full `mesh` using the straightforward scheme.

        Parameters
        ----------
        points
            The independent variable (energy transfer or momentum scalar) whose `data` to broaden.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of w/Q values
            for which there is `data`. Therefore, the ``sample`` dimension *must* match the length
            of the `data` array.
        data
            The intensities at the w/Q `points`.
        mesh
            The mesh to use for the broadening. This is a 1D array which *must* span the entire
            `points` space of interest.

        Returns
        -------
        spectrum
            The broadened spectrum. This is a 1D array of the same length as `mesh`.
        """
        kernels = self.get_peak(points, mesh)
        return np.einsum('i,ij...->j...', data, kernels)
