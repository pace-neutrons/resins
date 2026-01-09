from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import convolve
from scipy.stats import cauchy, norm, trapezoid, triang, uniform

from .model_base import InstrumentModel, ModelData
from .mixins import GaussianKernel1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


class StaticConvolveBroadenMixin:
    def broaden(self: InstrumentModel,
                points: Float[np.ndarray, 'sample dimension'],
                data: Float[np.ndarray, 'data'],
                mesh: Float[np.ndarray, '...'],
                ) -> Float[np.ndarray, '...']:
        """
        Broadens the `data` on the full `mesh` using a convolution of a single kernel with data.

        Parameters
        ----------
        points
            The independent variable (energy transfer or momentum scalar) whose `data` to broaden.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of w/Q values
            for which there is `data`. Therefore, the ``sample`` dimension *must* match the length
            of the `data` array.
        data
            The intensities at the points.
        mesh
            The mesh to use for the broadening. This is a 1D array which *must* span the entire
            `points` space of interest.

        Returns
        -------
        spectrum
            The broadened spectrum. This is a 1D array of the same length as `mesh`.
        """
        bin_width = mesh[1] - mesh[0]

        kernel = self.get_kernel(np.array([[points[0, 0]]]), mesh)[0] * bin_width
        return convolve(data, kernel, mode="same")


class StaticSnappedPeaksMixin:
    """Mixin providing a get_peak() based on application of broaden() to quantised delta functions

    In the composed class broaden() must not use get_peak()
    """
    @staticmethod
    def _get_snapped_dirac_peaks(points: Float[np.ndarray, 'sample dimension=1'],
                                mesh: Float[np.ndarray, 'mesh']
                                ) -> Float[np.ndarray, 'sample mesh']:
        """Compute digitized delta functions on the energy-mesh

        These are one-sample peaks with area 1, at the nearest mesh point. No interpolation is
        performed; this is intended to be convolved to produce a "snapped" spectrum for ease of
        comparison and interpolation, rather than the best possible representation of the data.

        Mesh is always interpreted as energy-transfer, corresponding to the first column of
        ``points``.

            Parameters
            ----------
            points
                The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
                array where N is the number of energy transfers.
            mesh
                A regular mesh on which to evaluate the kernel. This is a 1D array which *must* span
                the `points` transfer space of interest.

            Returns
            -------
            binned_points
                Intensity values corresponding to ``mesh`` for each item in ``points``

        """
        bin_width = mesh[1] - mesh[0]
        edges = np.linspace(mesh[0] - (bin_width / 2), mesh[-1] + (bin_width / 2), len(mesh) + 1)

        peaks = np.asarray([
            np.histogram(point[0], bins=edges)[0] for point in points
        ])

        return peaks / bin_width

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Apply the kernel at the nearest `mesh` point for at each value of `points` energy transfer.

        Note that:
        - peak positions are quantized to the nearest mesh point

        As a result the position of peaks in this approach does not vary smoothly with the input
        parameters.

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
            The Boxcar kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """

        delta_functions = self._get_snapped_dirac_peaks(points, mesh)

        peaks = [self.broaden(points[:1], delta_function, mesh)
                 for delta_function in delta_functions]

        return np.asarray(peaks)


class GenericBoxcar1DModel(StaticSnappedPeaksMixin, StaticConvolveBroadenMixin, InstrumentModel):
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
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
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
                   mesh: Float[np.ndarray, 'mesh'],
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
        indices = np.flatnonzero(kernel)

        if len(indices) > 1:
            first, *_, last = indices
        else:
            first = last = indices[0]

        kernel /= np.trapezoid(kernel, mesh)

        out_kernel = np.tile(kernel, (len(points), 1))

        return out_kernel

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Compute the Boxcar (square) kernel on the provided `mesh` at each value of the `points`
        energy transfer.

        Note that:
        - peak positions are quantized to the nearest mesh point
        - the boxcar kernel is quantized to an odd number of samples
        - the boxcar kernel is normalised to total value 1 based on the actual number of samples

        As a result the width, height and position of peaks in this approach do
        not vary smoothly with the input parameters.

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
            The Boxcar kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return super().get_peak(points, mesh)


class GenericTriangle1DModel(StaticConvolveBroadenMixin, StaticSnappedPeaksMixin, InstrumentModel):
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
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
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
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Triangle kernel centered on zero on the provided `mesh` at each value of
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
            The Triangle kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """

        bin_width = mesh[1] - mesh[0]
        quantized_fwhm = np.round(self.fwhm / bin_width) * bin_width

        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = triang.pdf(mesh, 0.5, loc=-quantized_fwhm, scale=quantized_fwhm * 2)

        return kernel

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Compute set of triangle functions at ``points`` on ``mesh``.

        Note that:
        - peak positions are quantized to the nearest mesh point
        - the triangle width kernel is quantized to run straight from peak to zeros

        As a result the width, height and position of peaks in this approach do
        not vary smoothly with the input parameters.

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
            Triangle kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered near the corresponding energy transfer. This is a 2D N x M array
             where N is the number of ``points and M is the length of ``mesh``.
        """
        return super().get_peak(points, mesh)


class GenericTrapezoid1DModel(StaticConvolveBroadenMixin, InstrumentModel):
    """
    A generic Trapezoid model.

    Models the :term:`resolution` as an isosceles Trapezoid function.

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
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
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
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Trapezoid kernel centered on zero on the provided `mesh` at each value of
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
            The Trapezoid kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, - 0.5 * self.long_base)

    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Trapezoid kernel on the provided `mesh` at each value of the `points`
        energy transfer.

        Parameters
        ----------
        points
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            energy-transfer space of interest.

        Returns
        -------
        kernel
            The Trapezoid kernel at each value of `points` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(points, mesh, points - 0.5 * self.long_base)

    def _get_kernel(self,
                    points: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        slope_length = 0.5 * (self.long_base - self.short_base)

        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = trapezoid.pdf(mesh, slope_length, 1 - slope_length,
                                     loc=displacement, scale=self.long_base)
        return kernel


class GenericGaussian1DModel(StaticConvolveBroadenMixin, GaussianKernel1DMixin, InstrumentModel):
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
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
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


class GenericLorentzian1DModel(StaticConvolveBroadenMixin, InstrumentModel):
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
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
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
                   mesh: Float[np.ndarray, 'mesh'],
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
                 mesh: Float[np.ndarray, 'mesh']
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
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        kernel = np.zeros((len(points), len(mesh)))
        kernel[:, :] = cauchy.pdf(mesh, loc=displacement, scale=self.fwhm * 0.5)
        return kernel

