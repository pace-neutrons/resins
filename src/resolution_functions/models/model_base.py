"""
Abstract base classes for all instrument models.

The classes defined within provide a common interface that all models must follow. A model consists
of two objects:

1. Data required by that model (subclassed from `ModelData`)
2. The model itself (subclassed from `InstrumentModel`)

Any new models must implement a subclass of each of these base classes (see their individual
documentation for details about how to use them). Additionally, the model must be added to the
`models.MODELS` mapping (found in ``models/__init__.py``) or they won't be found from the
`Instrument` class.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Float


class InvalidInputError(Exception):
    """
    A custom Exception, common to all models, signalling invalid user input.

    This exception should be raised whenever a user-provided parameter to a model is outside
    the valid bounds for that model of a particular instrument. For example, it is raised when the
    incident energy (``e_init``) provided to a model of a direct instrument is outside the range
    available to that instrument.
    """


class InvalidPointsError(Exception):
    """
    A custom Exception, common to all models, signaling that the user-provided points are invalid.
    """


@dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
class ModelData(ABC):
    """
    Abstract base class for defining the data required by a model.

    Subclasses of this base class define all the static data (e.g. if an instrument is modelled by a
    polynomial, this class will contain the coefficients) that the associated `InstrumentModel` will
    have access to - the only things it will have at its disposal are the
    attributes defined here and user-choice parameters. Notably, though, this
    dataclass is used as a transient data storage - the model itself does not
    keep a reference to it and stores as little of its information as possible.

    Therefore, it is used largely as a data verification for the data contained in the yaml files.
    Subclasses of this class must mirror the data defined in the yaml files that use the model
    - this class is the programming encoding of that data. In other words, this class defines which
    data a model uses while the yaml files provide the concrete values for a specific version of a
    specific instrument. If there is any discrepancy between the two (either the yaml file missing
    or having extra data), an error will be raised.

    This base class is a `dataclasses.dataclass`, so all subclasses have to be written accordingly.
    Furthermore, it is created with ``slots=True`` and ``frozen=True`` to prevent tampering with the
    static data. These points should be kept in mind when writing a subclass of `InstrumentModel`,
    since they mean that the attributes of this class cannot be edited at runtime.

    Additionally, this class is only constructed via the `instrument.Instrument._get_model_data`
    method, which means that no complex types are supported inside this class, only the basic Python
    types supported by the yaml format. Please do not use ``dataclass`` magic; if the information
    required for the model to work requires more complex structuring, the use of `typing.TypedDict`
    is recommended.

    Lastly, this class provides an implementation for the `restrictions` and `defaults` properties,
    but one that is only valid for models that do not have restrictions or defaults. Any subclasses
    should overwrite these as appropriate for the model.

    Attributes
    ----------
    function
        The name of the function, i.e. the alias for the corresponding `InstrumentModel`.
    citation
        The citation for a particular model. Please use this to look up more details and cite the
        model.
    restrictions
    defaults
    """
    function: str
    citation: list[str]

    @property
    def restrictions(self) -> dict[str, list[int | float]]:
        """
        The restrictions that the corresponding :term:`model` places on each :term:`setting`.

        This is provided via a dictionary where the keys are the name of the :term:`setting`,
        and the values are the restriction for that :term:`setting`. The restrictions can be the
        upper or lower bounds, or even a list of allowed values. If an invalid input is passed in, a
        `InvalidInputError` will be raised.

        Returns
        -------
        restrictions
            A mapping of :term:`setting` to its corresponding restriction.
        """
        return {}

    @property
    def defaults(self) -> dict[str, Any]:
        """
        The defaults for each :term:`setting` for the corresponding :term:`model`.

        This is provided as a dictionary where the keys are the name of the :term:`setting`,
        and the values are the default for that :term:`setting`. The default will be used when a
        particular :term:`setting` is not passed in to the :term:`model`

        Returns
        -------
        defaults
            A mapping of :term:`settings<setting>` to their corresponding default values.
        """
        return {}


class InstrumentModel(ABC):
    """
    Abstract base class for containing the code that defines a resolution function.

    In other words, the code representation of a mathematical function that models the resolution of
    an INS instrument. E.g., if the resolution of an instrument can be modelled using a polynomial,
    a subclass of this class will be the implementation of a polynomial, and `ModelData` will be the
    set of coefficient values.

    This base class provides the basic template for all subclasses but does not impose significant
    restrictions. However, there are some expectations that are not expressed with code:

    - The ``__init__`` method must be implemented, and it must take the corresponding `ModelData`
      subclass as its first positional argument.
      - The ``__init__`` method of this class must be called via ``super()``
      - The subclass ``__init__`` method should take all user-choice parameters as arguments, i.e.
        anything that was decided at the time of the experiment, such as the initial energy or
        chopper frequency.
        - These parameters should be ``Optional`` wherever possible, with the defaults for each
          instrument in the corresponding yaml files.
      - Any number of any other parameters is allowed, though ``__init__`` must not accept the
        energy transfer/momentum ([w, Q]) parameter used in the other methods.
      - The ``__init__`` method should perform as much of the computation as possible, i.e. any
        computation that does not involve the [w, Q] parameter.
      - No reference to the `ModelData` should be kept.
    - The ``get_characteristics``, ``get_kernel``, and ``broaden`` methods must be implemented.
      - Some or all of these may come as reusable code (as appropriate) via the use of the mixin
        pattern (see `resolution_functions.models.mixins`).
      - Each must take the ``omega_q`` argument, which must be a ``sample`` x ``dimension`` 2D
        array, where ``sample`` is the number of [w, Q] values provided by the user and
        ``dimension`` are the [w, Q] variables required by the model, as defined in the ``input``
        class variable. These can be any combination of the energy transfer (i.e. frequencies),
        the momentum value (q), and the momentum vector (q-vectors).
        - For example, a model that uses the energy transfer and momentum scalar would have
          ``dimension=2`` and ``input = ('energy transfer', 'momentum')``.
      - It must also take ``*args`` and ``**kwargs``.
    - The `input` class variable must be given a specific value.
    - The `data_class` class variable must be assigned to the corresponding `ModelData` subclass.
    - Any additional defined methods should be private, but feel free to use your discretion.

    Parameters
    ----------
    model_data
        The data associated with the model

    Attributes
    ----------
    input
        The names of the columns in the ``omega_q`` array expected by all computation methods, i.e.
        the names of the independent variables ([Q, w]) that the model models.
    data_class
        The `ModelData` subclass associated with this particular model.
    citation
    """
    input: ClassVar[tuple[str, ...]]

    data_class: ClassVar[type[ModelData]]

    def __init__(self, model_data: ModelData, **_):
        self._citation = model_data.citation

    @abstractmethod
    def get_characteristics(self, points: Float[np.ndarray, 'sample dimension']
                            ) -> dict[str, Float[np.ndarray, 'sample']]:
        """
        Computes the characteristics of the broadening function at each point in [w, Q] space
        provided.

        Parameters
        ----------
        points
            The points in [w, Q] space at which to compute the characteristics of the broadening
            kernel. These have to enumerate all the desired combinations of the independent
            variables [w, Q]. This *must* be a ``sample`` x ``dimension`` 2D array where
            ``sample`` is the number of [w, Q] points and ``dimension`` is the number of
            independent variables as specified by the ``input`` class variable.

        Returns
        -------
        characteristics
            The characteristics of the broadening function at each combination of independent
            variables.
        """

    @abstractmethod
    def get_kernel(self,
                   points: Float[np.ndarray, 'sample dimension'],
                   *meshes: list[Float[np.ndarray, 'mesh']],
                   ) -> Float[np.ndarray, '...']:
        """
        Computes the kernel centered on zero on the provided `meshes` at each point in [w, Q] space
        (`points`) provided.

        Parameters
        ----------
        points
            The points in [w, Q] space at which to compute the kernels. These must be all the
            combinations of the independent variables [w, Q] at whose values to compute the
            kernels. This *must* be a ``sample`` x ``dimension`` 2D array where ``sample`` is the
            number of [w, Q] points and ``dimension`` is the number of independent variables
            as specified by the ``input`` class variable.
        *meshes
            The collection of meshes on which to evaluate each kernel. Each of these
            must be a 1D array specifying the points along a direction in the [w, Q] space on which
            to compute the kernels. All of the meshes are expanded into an (N+1)D results array that
            contains the value of the kernel at that combination of points from each mesh. Each mesh
            must contain a zero point so that the kernel can be centred on zero.

        Returns
        -------
        kernel
            The normalised kernel representing the broadening, centered on zero, produced for each
            [w, Q] point provided via the `points` array. This is an (N+1)D array, where N
            is the number of independent variables.
        """

    @abstractmethod
    def get_peak(self,
                 points: Float[np.ndarray, 'sample dimension'],
                 *meshes: Float[np.ndarray, 'mesh'],
                 ) -> Float[np.ndarray, '...']:
        """
        Computes the broadening peak on the provided `meshes` at each point in the [w, Q] space
        (`points`) provided, centered on that point.

        Parameters
        ----------
        points
            The points in [w, Q] space at which to compute the broadening peaks. These must be all
            the combinations of the independent variables [w, Q] at whose values to compute the
            peaks. This *must* be a ``sample`` x ``dimension`` 2D array where ``sample`` is the
            number of [w, Q] points and ``dimension`` is the number of independent variables
            as specified by the ``input`` class variable.
        *meshes
            The collection of meshes on which to evaluate each peak. Each of these
            must be a 1D array specifying the points along a direction in the [w, Q] space on which
            to compute the kernels. All of the meshes are expanded into an (N+1)D results array that
            contains the value of the kernel at that combination of points from each mesh. Each mesh
            must span enough space to include all of the provided [w, Q] `points`.

        Returns
        -------
        kernel
            The normalised peak representing the broadening, centered on its corresponding [w, Q]
            value on the mesh, produced for each [w, Q] point provided via the `points`
            array. This is an (N+1)D array, where N is the number of independent variables.
        """

    @abstractmethod
    def broaden(self,
                points: Float[np.ndarray, 'sample dimension'],
                data: Float[np.ndarray, 'data'],
                *meshes: Float[np.ndarray, 'mesh'],
                ) -> Float[np.ndarray, '...']:
        """
        Broadens the `data` on the `meshes`.

        Parameters
        ----------
        points
            The points in [w, Q] space at whose `data` to broaden. This *must* be a ``sample`` x
            ``dimension`` 2D array where ``sample`` is the number of [w, Q] combinations and
            ``dimension`` is the number of independent variables as specified by the ``input``
            class variable. The ``sample`` dimension *must* match the length of the `data` array.
        data
            The intensities at the [w, Q] `points`.
        *meshes
            The collection of meshes to use for the broadening. Each of these must be a 1D array
            specifying the points along a direction in the [w, Q] space on which to compute the
            kernels. All of the meshes are expanded into an ND results array that
            contains the value of the kernel at that combination of points from each mesh. Each mesh
            must span enough space to include all of the provided [w, Q] `points`.

        Returns
        -------
        spectrum
            The broadened spectrum. This an ND array, where N is the number of independent variables
            (this is also the number of `meshes` and the ``dimension`` axis of `points`) and the
            length along each axis is the length of the corresponding mesh.
        """

    def __call__(self,
                 points: Float[np.ndarray, 'sample dimension'],
                 data: Float[np.ndarray, 'data'],
                 *meshes: Float[np.ndarray, 'mesh'],
                 ) -> Float[np.ndarray, '...']:
        """
        Broadens the `data` on the `meshes`.

        Parameters
        ----------
        points
            The points in [w, Q] space at whose `data` to broaden. This *must* be a ``sample`` x
            ``dimension`` 2D array where ``sample`` is the number of [w, Q] combinations and
            ``dimension`` is the number of independent variables as specified by the ``input``
            class variable. The ``sample`` dimension *must* match the length of the `data` array.
        data
            The intensities at the [w, Q] `points`.
        *meshes
            The collection of meshes to use for the broadening. Each of these must be a 1D array
            specifying the points along a direction in the [w, Q] space on which to compute the
            kernels. All of the meshes are expanded into an ND results array that
            contains the value of the kernel at that combination of points from each mesh. Each mesh
            must span enough space to include all of the provided [w, Q] `points`.

        Returns
        -------
        spectrum
            The broadened spectrum. This an ND array, where N is the number of independent variables
            (this is also the number of `meshes` and the ``dimension`` axis of `points`) and the
            length along each axis is the length of the corresponding mesh.
        """
        return self.broaden(points, data, *meshes)

    def __str__(self) -> str:
        return f'{type(self).__name__}(citation={self.citation})'

    @property
    def citation(self) -> list[str]:
        """
        The citation for this model. Please use this to look up more details and cite the model.

        Returns
        -------
        citation
            The citation.
        """
        return self._citation
