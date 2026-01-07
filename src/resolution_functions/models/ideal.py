from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import convolve
from scipy.stats import cauchy, norm, trapezoid, triang

from .model_base import InstrumentModel, ModelData
from .mixins import GaussianKernel1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


class StaticConvolveBroadenMixin:
    def broaden(self: InstrumentModel,
                omega_q: Float[np.ndarray, 'sample dimension'],
                data: Float[np.ndarray, 'data'],
                mesh: Float[np.ndarray, '...'],
                ) -> Float[np.ndarray, '...']:
        """
        Broadens the `data` on the full `mesh` using a convolution of a single kernel with data.

        Parameters
        ----------
        omega_q
            The independent variable (energy transfer or momentum scalar) whose `data` to broaden.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of w/Q values
            for which there is `data`. Therefore, the ``sample`` dimension *must* match the length
            of the `data` array.
        data
            The intensities at the `omega_q` points.
        mesh
            The mesh to use for the broadening. This is a 1D array which *must* span the entire
            `omega_q` space of interest.

        Returns
        -------
        spectrum
            The broadened spectrum. This is a 1D array of the same length as `mesh`.
        """
        kernel = self.get_kernel(np.array([[omega_q[0, 0]]]), mesh)
        return convolve(kernel, data)


class GenericBoxcar1DModel(StaticConvolveBroadenMixin, InstrumentModel):
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
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
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

    def get_characteristics(self, omega_q: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `omega_q`.

        This model is a static test model, so it returns the same width for each value of `omega_q`,
        which is in the form of the width of a Boxcar kernel.

        Parameters
        ----------
        omega_q
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Boxcar width in meV and derived standard deviation (sigma).
        """
        characteristics = {'width': np.ones(len(omega_q)) * self.width}
        characteristics['sigma'] = np.full_like(characteristics['width'], np.sqrt(1/12))
        return characteristics

    def get_kernel(self,
                   omega_q: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, 'mesh']
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Boxcar (square) kernel centered on zero on the provided `mesh` at each value of
        `omega_q` (energy transfer or momentum scalar).

        Parameters
        ----------
        omega_q
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Boxcar kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        radius = self.width * 0.5
        indices = np.logical_or(mesh <= - radius, mesh >= radius)

        kernel = np.zeros((len(omega_q), len(mesh)))
        kernel[:, indices] = 1 / self.width

        return kernel

    def get_peak(self,
                 omega_q: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Boxcar (square) kernel on the provided `mesh` at each value of the `omega_q`
        energy transfer.

        Parameters
        ----------
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `omega_q` transfer space of interest.

        Returns
        -------
        kernel
            The Boxcar kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        radius = self.width * 0.5

        kernel = np.zeros((len(omega_q), len(mesh)))
        for value in omega_q:
            value = value[0]
            indices = np.logical_or(mesh >= (value - radius), mesh <= (value + radius))
            kernel[:, indices] = 1 / self.width

        return kernel


class GenericTriangle1DModel(StaticConvolveBroadenMixin, InstrumentModel):
    """
    A generic Triangle model.

    Models the :term:`resolution` as an isosceles Triangle function.

    Parameters
    ----------
    model_data
        The data associated with the model for a given version of a given instrument.
    fwhm
        The width (in Full-Width Half-Maximum) of the Triangle function. This width is used for all
        values of [w, Q].

    Attributes
    ----------
    input
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
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

    def get_characteristics(self, omega_q: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `omega_q`.

        This model is a static test model, so it returns the same width for each value of `omega_q`,
        which is in the form of the Full-Width Half-Maximum of a Triangle model.

        Parameters
        ----------
        omega_q
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Triangle width as FWHM.
        """
        return {'fwhm': np.ones(len(omega_q)) * self.fwhm}

    def get_kernel(self,
                   omega_q: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Triangle kernel centered on zero on the provided `mesh` at each value of
        `omega_q` (energy transfer or momentum scalar).

        Parameters
        ----------
        omega_q
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Triangle kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, -self.fwhm)

    def get_peak(self,
                 omega_q: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Triangle kernel on the provided `mesh` at each value of the `omega_q`
        energy transfer.

        Parameters
        ----------
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `omega_q` transfer space of interest.

        Returns
        -------
        kernel
            The Triangle kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, omega_q-self.fwhm)

    def _get_kernel(self,
                    omega_q: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        kernel = np.zeros((len(omega_q), len(mesh)))
        kernel[:, :] = triang.pdf(mesh, 0.5, loc=displacement, scale=self.fwhm * 2)
        return kernel


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
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
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

    def get_characteristics(self, omega_q: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Returns the characteristics of a Trapezoid function for each value of energy transfer given
        by `omega_q`.

        This model is a static test model, so it returns the same characteristics for each value
        of `omega_q`. A Trapezoid model has two characteristics:

        * ``long_base`` - the length of the longer (bottom) base of a trapezoid
        * ``short_base`` - the length of the shorter (top) base of a trapezoid.

        Parameters
        ----------
        omega_q
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function.
        """
        return {
            'long_base': np.ones(len(omega_q)) * self.long_base,
            'short_base': np.ones(len(omega_q)) * self.short_base,
        }

    def get_kernel(self,
                   omega_q: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Trapezoid kernel centered on zero on the provided `mesh` at each value of
        `omega_q` (energy transfer or momentum scalar).

        Parameters
        ----------
        omega_q
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Trapezoid kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, - 0.5 * self.long_base)

    def get_peak(self,
                 omega_q: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Trapezoid kernel on the provided `mesh` at each value of the `omega_q`
        energy transfer.

        Parameters
        ----------
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `omega_q` transfer space of interest.

        Returns
        -------
        kernel
            The Trapezoid kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, omega_q - 0.5 * self.long_base)

    def _get_kernel(self,
                    omega_q: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        slope_length = 0.5 * (self.long_base - self.short_base)

        kernel = np.zeros((len(omega_q), len(mesh)))
        kernel[:, :] = trapezoid.pdf(mesh, slope_length, 1 - slope_length,
                                     loc=displacement, scale=self.long_base)
        return kernel


class GenericGaussian1DModel(StaticConvolveBroadenMixin, GaussianKernel1DMixin, InstrumentModel):
    """
    A generic Boxcar model.

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
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
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

    def get_characteristics(self, omega_q: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `omega_q`.

        This model is a static test model, so it returns the same width for each value of `omega_q`,
        which is in the form of the standard deviation (sigma) of a Gaussian model.

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
        return {'sigma': np.ones(len(omega_q)) * self.sigma}


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
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
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

    def get_characteristics(self, omega_q: Float[np.ndarray, 'sample dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Returns the broadening width at each value of energy transfer given by `omega_q`.

        This model is a static test model, so it returns the same width for each value of `omega_q`,
        which is in the form of the Full-Width Half-Maximum of a Lorentzian model.

        Parameters
        ----------
        omega_q
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Lorentzian width as FWHM.
        """
        return {'fwhm': np.ones(len(omega_q)) * self.fwhm}

    def get_kernel(self,
                   omega_q: Float[np.ndarray, 'sample dimension=1'],
                   mesh: Float[np.ndarray, 'mesh'],
                   ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Lorentzian kernel centered on zero on the provided `mesh` at each value of
        `omega_q` (energy transfer or momentum scalar).

        Parameters
        ----------
        omega_q
            The energy transfer or momentum scalar for which to compute the kernel. This *must* be
            a Nx1 2D array where N is the number of w/Q values.
        mesh
            The mesh on which to evaluate the kernel. A 1D array.

        Returns
        -------
        kernel
            The Lorentzian kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on zero. This is a 2D N x M array where N is the number of w/Q
            values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, 0.)

    def get_peak(self,
                 omega_q: Float[np.ndarray, 'sample dimension=1'],
                 mesh: Float[np.ndarray, 'mesh']
                 ) -> Float[np.ndarray, 'sample mesh']:
        """
        Computes the Lorentzian kernel on the provided `mesh` at each value of the `omega_q`
        energy transfer.

        Parameters
        ----------
        omega_q
            The energy transfer in meV for which to compute the kernel. This *must* be a Nx1 2D
            array where N is the number of energy transfers.
        mesh
            The mesh on which to evaluate the kernel. This is a 1D array which *must* span the
            `omega_q` transfer space of interest.

        Returns
        -------
        kernel
            The Lorentzian kernel at each value of `omega_q` as given by this model, computed on the
            `mesh` and centered on the corresponding energy transfer. This is a 2D N x M array where
            N is the number of w/Q values and M is the length of the `mesh` array.
        """
        return self._get_kernel(omega_q, mesh, omega_q)

    def _get_kernel(self,
                    omega_q: Float[np.ndarray, 'sample dimension=1'],
                    mesh: Float[np.ndarray, 'mesh'],
                    displacement: float | Float[np.ndarray, 'sample'] = 0.,
                    ) -> Float[np.ndarray, 'sample mesh']:
        kernel = np.zeros((len(omega_q), len(mesh)))
        kernel[:, :] = cauchy.pdf(mesh, loc=displacement, scale=self.fwhm * 0.5)
        return kernel
