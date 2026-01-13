"""
Model for TOSCA-like :term:`instruments<instrument>` from the [VISION-paper]_.

All classes within are exposed for reference only and should not be instantiated directly. For
obtaining the :term:`resolution function` of an :term:`instrument`, please use the
`resins.instrument.Instrument.get_resolution_function` method.

.. [VISION-paper] https://doi.org/10.1016/j.nima.2009.03.204
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .model_base import InstrumentModel, ModelData, InvalidPointsError
from .mixins import GaussianKernel1DMixin, SimpleBroaden1DMixin

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class VisionPaperModelData(ModelData):
    """
    Data for the `VisionPaperModel` :term:`model`.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for `PantherAbINSModel`.
    citation
        The citation for the model. Please use this to look up more details and cite the model.
    restrictions
        All constraints that the model places on the :term:`settings<setting>`. If the value is a
        `list`, this signifies the `range` style (start, stop, step) tuple, and if it is a `set`, it
        is a set of explicitly allowed values.
    defaults
        The default values for the :term:`settings<setting>`, used when a value is not provided when
        creating the model.
    primary_flight_path
        Distance between the :term:`moderator` and the :term:`sample` in meters (m).
    primary_flight_path_uncertainty
        The uncertainty associated with the `primary_flight_path`, in meters (m).
    sample_thickness
        Thickness of the :term:`sample` in meters (m).
    detector_thickness
        Thickness of the :term:`detector` in meters (m).
    crystal_plane_spacing
        Distance between the layers of atoms making up the :term:`detector`, in meters (m).
    d_r
        Uncertainty associated with the :term:`detector` offset, in meters (m).
    d_t
        Uncertainty associated with the :term:`source` pulse shape, in ? (?).
    angles
        Angle between the :term:`sample` and the analyser, in degrees.
    distance_sample_analyzer
        Distance between the :term:`sample` and the analyser, in meters (m).
    average_bragg_angle_graphite
        Average Bragg angle of the graphite analyser, in degrees.
    """
    primary_flight_path: float
    primary_flight_path_uncertainty: float
    sample_thickness: float
    detector_thickness: float
    crystal_plane_spacing: float
    d_r: float
    d_t: float
    angles: list[float]
    distance_sample_analyzer: float
    average_bragg_angle_graphite: float


class VisionPaperModel(GaussianKernel1DMixin, SimpleBroaden1DMixin, InstrumentModel):
    """
    Model for TOSCA-like :term:`instruments<instrument>` from the [VISION paper]_.

    Models the :term:`resolution` as a function of energy transfer (frequencies) only, with the
    output model being an Ikeda-Carpenter distribution. This is done by taking into account the
    contributions from the various parts of the :term:`instrument` (for more information, please see
    the reference).

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
        Reference to the `VisionPaperModelData` type.
    citation
    """
    input = ('energy_transfer',)

    data_class = VisionPaperModelData

    PLANCK = 6.626068e-34  # J s
    REDUCED_PLANCK = 1.054571817e-34  # J s
    NEUTRON_MASS = 1.67492749804e-27  # kg

    def __init__(self, model_data: VisionPaperModelData, **_):
        super().__init__(model_data)

        self.l1 = model_data.primary_flight_path
        self.d_t = model_data.d_t

        self.e0 = self.PLANCK ** 2 * 0.5 / self.NEUTRON_MASS * (0.5 / model_data.crystal_plane_spacing) ** 2
        self.nu0 = 0.5 * self.PLANCK / (self.NEUTRON_MASS * model_data.crystal_plane_spacing)
        self.one_over_l1 = 1 / self.l1
        self.distance_ratio = model_data.primary_flight_path_uncertainty * self.one_over_l1

        self.theta = np.deg2rad(model_data.average_bragg_angle_graphite)
        self.capital_t = 0.5 * 1 / np.tan(self.theta)

        self.z2 = model_data.distance_sample_analyzer
        self.capital_t_over_z2 = self.capital_t / self.z2

        self.d_a = model_data.sample_thickness ** 2 / 12
        d_b = 0.7e-6
        d_c = model_data.detector_thickness ** 2 / 12
        self.db_dc_factor = (2 * d_b + d_c)

        self.final_term = self.e0 / np.tan(self.theta) / self.z2 * model_data.d_r

    def get_characteristics(self, points: Float[np.ndarray, 'energy_transfer dimension=1']
                            ) -> dict[str, Float[np.ndarray, 'sigma']]:
        """
        Computes the broadening width at each value of energy transfer given by `points`.

        The model approximates the broadening using the Gaussian distribution, so the returned
        widths are in the form of the standard deviation (sigma).

        Parameters
        ----------
        points
            The energy transfer in meV at which to compute the width in sigma of the kernel.
            This *must* be a ``sample`` x 1 2D array where ``sample`` is the number of energy
            transfers.

        Returns
        -------
        characteristics
            The characteristics of the broadening function, i.e. the Gaussian width as sigma.
        """
        try:
            points = points[:, 0]
        except IndexError as e:
            raise InvalidPointsError(
                f'The provided array of points (shape={points.shape}) is not valid. The points '
                f'array must be a Nx1 2D array where N is the number of energy transfers.'
            ) from e
        e1 = points * self.REDUCED_PLANCK + self.e0 * (1 / np.sin(self.theta))
        z0 = self.l1 * (self.e0 / e1) ** 0.5
        one_over_z0 = 1 / z0

        sigma = self.distance_ratio - self.nu0 * self.d_t / z0
        sigma += (self.one_over_l1 + one_over_z0 + self.capital_t_over_z2) * self.d_a
        sigma += (one_over_z0 + self.capital_t_over_z2) * self.db_dc_factor
        sigma *= 2 * e1
        sigma -= self.final_term

        return {'sigma': sigma}
