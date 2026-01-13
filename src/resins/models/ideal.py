from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import cauchy, trapezoid, triang, uniform

from .model_base import InstrumentModel, ModelData
from .mixins import GaussianKernel1DMixin, SimpleBroaden1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


class StaticSnappedPeaksMixin:
    """Mixin providing a get_peak() by copying kernel to nearest bins

    This results in 'snapping' to the nearest bin rather than a more accurate evaluation at the true
    point position. However it also eliminates surprising changes to the kernel shape based on the
    sub-bin position, and gives similar results to convolution with pre-binned data.

    get_kernel() must be provided by the inheriting class or another Mixin.

    """

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, ' mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Apply the kernel at the nearest `mesh` point for at each value of `points` energy transfer.

        The kernel is obtained by calling get_kernel() for the first item of ``points``: this is
        intended for use with energy-independent ("static") broadening functions.

        Note that peak positions are quantized to the nearest mesh point. As a result, the position
        of peaks in this approach does not vary smoothly with the input parameters.

        Parameters
        ----------
        points
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `points` transfer space of interest.

        Returns
        -------
        peaks
            Broadening kernel aligned to nearest ``mesh`` point for each input point. This is a 2D N
            x M array where N is the length of ``points`` (i.e. number of ω values) and M is the
            length of the ``mesh`` array.
        """
        assert points.shape[0] == points.size  # i.e. Nx1 array
        assert mesh.ndim == 1

        bin_width = mesh[1] - mesh[0]
        mesh_length = len(mesh)

        # Set up kernel mesh: 0-centered with range equal to output mesh on each side
        kernel_range = (mesh_length - 1) * bin_width
        kernel_length = 2 * mesh_length - 1
        kernel_mesh = np.linspace(-kernel_range, kernel_range, kernel_length)

        assert kernel_mesh[mesh_length - 1] == 0.

        kernel = self.get_kernel(points[:1], kernel_mesh)[0]

        # Create output mesh with additional padding for kernel overlapping edges
        output = np.zeros([len(points), mesh_length * 3 - 2])

        positions = np.searchsorted(mesh, points.flatten() - bin_width * 0.5)
        for i, (point, position) in enumerate(zip(points, positions)):
            output[i, position:(position + kernel_length)] = kernel

        return output[:, (mesh_length - 1):(2 * mesh_length - 1)]


class GenericBoxcar1DModel(StaticSnappedPeaksMixin, SimpleBroaden1DMixin, InstrumentModel):
    """
    A generic Boxcar model.

    Models the :term:`resolution` as a Boxcar (square) function.

    A useful relationship: the standard deviation of a width-1 boxcar is √(1/12).
    So to produce crudely "equivalent" broadening to a Gaussian of known σ,
    use a boxcar width = σ √12 .

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    width
        The width of the Boxcar function in meV. This width is used for all values of [w, Q].

    Attributes
    ----------
    input
        The names of the columns in the ``points`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `ModelData` type.
    width
        The width of the Boxcar function in meV. This width is used for all values of [w, Q].
    citation

    Warnings
    --------
    This model is for testing purposes - it does not do any computation and instead uses the
    user-provided width for all values of [w, Q]. It should not be normally used to model
    instruments.
    """
    input = ('energy_transfer',)

    data_class = ModelData

    def __init__(self, model_data: ModelData, width: float = 1., **_):
        super().__init__(model_data)
        self.width = width

    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, ' sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `points`.

        This model is a static test model, so it returns the same width for each value of `points`,
        which is in the form of the width of a Boxcar kernel.

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Boxcar width in meV and derived standard deviation (sigma).
        """
        characteristics = {'width': np.ones(len(points)) * self.width}
        characteristics['sigma'] = np.full_like(characteristics['width'], np.sqrt(1/12))
        return characteristics

    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, ' mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Boxcar (square) kernel centered on zero on the provided `mesh` at each value of
        `points` (energy transfer or momentum scalar).

        Note that these kernels generated with scipy.signal.uniform are rounded
        to an odd-integer width and edges move directly from full-height to
        zero. Better but more complicated algorithms exist.

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Boxcar kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        kernel = uniform(loc=(-self.width / 2), scale=self.width).pdf(mesh)
        kernel /= np.trapezoid(kernel, mesh)

        out_kernel = np.tile(kernel, (len(points), 1))

        return out_kernel


class GenericTriangle1DModel(SimpleBroaden1DMixin, StaticSnappedPeaksMixin, InstrumentModel):
    """
    A generic Triangle model.

    Models the :term:`resolution` as an isosceles Triangle function.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    fwhm
        The width (in Full-Width Half-Maximum) of the Triangle function. This width is used for all
        values of [w, Q]. When realised on a user mesh, the width is rounded to an integer number of
        bins to create straight lines from the peak to zero.

    Attributes
    ----------
    input
        The names of the columns in the ``points`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `ModelData` type.
    fwhm
        The width (in Full-Width Half-Maximum) of the Triangle function. This width is used for all
        values of [w, Q].
    citation

    Warnings
    --------
    This model is for testing purposes - it does not do any computation and instead uses the
    user-provided width for all values of [w, Q]. It should not be normally used to model
    instruments.
    """
    input = ('energy_transfer',)

    data_class = ModelData

    def __init__(self, model_data: ModelData, fwhm: float = 1., **_):
        super().__init__(model_data)
        self.fwhm = fwhm

    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, ' sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `points`.

        This model is a static test model, so it returns the same width for each value of `points`,
        which is in the form of the Full-Width Half-Maximum of a Triangle model.

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Triangle width as FWHM.
        """
        return {'fwhm': np.ones(len(points)) * self.fwhm}

    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, ' mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Triangle kernel centered on zero on the provided `mesh` at each value of
        `points` (energy transfer or momentum scalar).

        Note that shape and area are only exactly correct when FHWM equals an integer number of
        bins.

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Triangle kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = triang.pdf(mesh, 0.5, loc=-self.fwhm, scale=self.fwhm * 2)

        return kernel


class GenericTrapezoid1DModel(SimpleBroaden1DMixin, StaticSnappedPeaksMixin, InstrumentModel):
    """
    A generic Trapezoid model.

    Models the :term:`resolution` as an isosceles Trapezoid function.

    Note that shape and area are only exactly correct when base lengths correspond to an even number
    of bin widths. The get_peak() and broaden() methods will snap input points to the nearest mesh
    value; this results in a consistent peak shape.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    long_base
        The length of the longer (bottom) base of the Trapezoid function. This width is used for all
        values of [w, Q].
    short_base
        The length of the shorter (top) base of the Trapezoid function. This width is used for all
        values of [w, Q].

    Attributes
    ----------
    input
        The names of the columns in the ``points`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `ModelData` type.
    long_base
        The length of the longer (bottom) base of the Trapezoid function. This width is used for all
        values of [w, Q].
    short_base
        The length of the shorter (top) base of the Trapezoid function. This width is used for all
        values of [w, Q].
    citation

    Warnings
    --------
    This model is for testing purposes - it does not do any computation and instead uses the
    user-provided width for all values of [w, Q]. It should not be normally used to model
    instruments.
    """
    input = ('energy_transfer',)

    data_class = ModelData

    def __init__(self, model_data: ModelData, long_base: float = 1., short_base: float = 0.5, **_):
        super().__init__(model_data)
        self.long_base = long_base
        self.short_base = short_base

    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, ' sample']]:
        """
        Returns the characteristics of a Trapezoid function for each value of energy transfer given
        by `points`.

        This model is a static test model, so it returns the same characteristics for each value
        of `points`. A Trapezoid model has two characteristics:

        * ``long_base`` - the length of the longer (bottom) base of a trapezoid
        * ``short_base`` - the length of the shorter (top) base of a trapezoid.

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function.
        """
        return {
            'long_base': np.ones(len(points)) * self.long_base,
            'short_base': np.ones(len(points)) * self.short_base,
        }

    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, ' mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Trapezoid kernel centered on zero on the provided `mesh` at each value of
        `points` (energy transfer or momentum scalar).

        Note that shape and area are only exactly correct when base lengths correspond to an even
        number of bin widths.

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Trapezoid kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        slope_length = 0.5 * (self.long_base - self.short_base) / self.long_base

        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = trapezoid.pdf(mesh, slope_length, 1 - slope_length,
                                     loc=(-0.5 * self.long_base), scale=self.long_base)
        return kernel


class GenericGaussian1DModel(SimpleBroaden1DMixin, GaussianKernel1DMixin, InstrumentModel):
    """
    A generic Gaussian model.

    Models the :term:`resolution` as a Gaussian function.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    sigma
        The width (in sigma) of the Gaussian function. This width is used for all values of [w, Q].

    Attributes
    ----------
    input
        The names of the columns in the ``points`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `ModelData` type.
    sigma
        The width (in sigma) of the Gaussian function. This width is used for all values of [w, Q].
    citation

    Warnings
    --------
    This model is for testing purposes - it does not do any computation and instead uses the
    user-provided width for all values of [w, Q]. It should not be normally used to model
    instruments.
    """
    input = ('energy_transfer',)

    data_class = ModelData

    def __init__(self, model_data: ModelData, sigma: float = 1., **_):
        super().__init__(model_data)
        self.sigma = sigma

    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, ' sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `points`.

        This model is a static test model, so it returns the same width for each value of `points`,
        which is in the form of the standard deviation (sigma) of a Gaussian model.

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Gaussian width as sigma in meV.
        """
        return {'sigma': np.ones(len(points)) * self.sigma}


class GenericLorentzian1DModel(SimpleBroaden1DMixin, InstrumentModel):
    """
    A generic Lorentzian model.

    Models the :term:`resolution` as a Lorentzian function.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    fwhm
        The width (in Full-Width Half-Maximum) of the Lorentzian function. This width is used for all
        values of [w, Q].

    Attributes
    ----------
    input
        The names of the columns in the ``points`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        Reference to the `ModelData` type.
    fwhm
        The width (in Full-Width Half-Maximum) of the Lorentzian function. This width is used for all
        values of [w, Q].
    citation

    Warnings
    --------
    This model is for testing purposes - it does not do any computation and instead uses the
    user-provided width for all values of [w, Q]. It should not be normally used to model
    instruments.
    """
    input = ('energy_transfer',)

    data_class = ModelData

    def __init__(self, model_data: ModelData, fwhm: float = 1., **_):
        super().__init__(model_data)
        self.fwhm = fwhm

    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, ' sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `points`.

        This model is a static test model, so it returns the same width for each value of `points`,
        which is in the form of the Full-Width Half-Maximum of a Lorentzian model.

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Lorentzian width as FWHM.
        """
        return {'fwhm': np.ones(len(points)) * self.fwhm}

    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, ' mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Lorentzian kernel centered on zero on the provided `mesh` at each value of
        `points` (energy transfer or momentum scalar).

        Parameters
        ----------
        points
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Lorentzian kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, 0.)

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, ' mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Lorentzian kernel on the provided `mesh` at each value of the `points`
        energy transfer.

        Parameters
        ----------
        points
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `points` transfer space of interest.

        Returns
        -------
        kernel
            The Lorentzian kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, points)

    def _get_kernel(self,
                    points: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, ' mesh'],
                    displacement: float | Float[np.ndarray, ' sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = cauchy.pdf(mesh, loc=displacement, scale=self.fwhm * 0.5)
        return kernel

