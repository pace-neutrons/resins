How To Add A New Model
======================

ResINS strives to be a repository for all :term:`models<model>` of :term:`INS`
:term:`instruments<instrument>`, but it is inevitable that there will be some
that have not been included *yet*. Fortunately, ResINS was designed to make
adding new models as straightforward as possible, though depending on how
unique the model is, it may require significant amount of work:

.. contents::
    :backlinks: entry
    :depth: 2
    :local:


.. _howto-model:

How to add a new parameter-only model
-------------------------------------

Adding a new :term:`model` which only alters the parameters of another, existing
model is the simpler task; all that is required is to edit the YAML data file
of the corresponding :term:`instrument`. Now, there *is* going to be some
difference in how to do this when creating a :term:`model` for personal use only
and when contributing to ResINS, but the overall process is identical to
:doc:`adding a new version<add_version>`, so see that guide for details. Here,
going forward, it is assumed that you have a YAML data file ready for editing.

A :term:`model` is a "property" of a :term:`version` of an :term:`instrument`
and so it must be added to a particular version (though
:ref:`YAML magic<spec-yaml-magic>` can be used to avoid repetition). Therefore,
to add a new :term:`model`, up to two new entries must be added inside
``models`` key of a particular :term:`version` of an :term:`instrument`:

* If adding a new version of an existing model:

  1. A new entry must be added whose key has the same name as the previous
     versions, but whose version number is incremented by one. E.g., given a
     model called ``model1`` and which has versions ``model1_v1`` and
     ``model1_v2``, the new model should be called ``model1_v3``.

     * The corresponding value should be a dictionary containing the data for
       the model, :ref:`see below<howto-model-params>` for details.

  2. If this new version should become the new default version for the model
     (for example in the case of a bugfix), please edit the associated alias
     entry to point to the new version. For example, using the above names,
     there should be an entry ``model1: "model1_v2"`` which should be changed to
     ``model1: "model1_v3"``.

     * In this case, the key-value pair should already exist and only needs to
       be changed.

     * The value of this entry must be kept as a string which must correspond to
       a valid key in the ``models`` dictionary.

* If adding a new model, unrelated to the others:

  1. A new entry must be added whose key has the name following the schema:
     ``{model_name}_v1`` (since this is a new model, its first version should
     have the number ``1``), e.g. ``model1_v1``.

     * The corresponding value should be a dictionary containing the data for
       the model, :ref:`see below<howto-model-params>` for details.

  2. A new entry must be added whose key is only the ``model_name`` from above.

     * The corresponding value must be a string whose value must be the key
       added previously, so (using the above example) ``model1: "model1_v1"``.


.. _howto-model-params:

How to specify model parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the model keys added, the next step is to add the data associated with the
versioned model. The data must follow the
:doc:`YAML file spec<../dev/yaml_spec>`, where the guidance on what belongs
where can also be found, but the general points to keep in mind are as follows:

* The ``function`` entry must have a value that corresponds to one of the
  existing models. See :py:const:`resins.models.MODELS` for a
  dictionary that maps these ``function`` values to ResINS model objects. The
  created model will use the corresponding object.

  * To use a ``function`` not listed in ``MODELS``, new code will have to be
    written, see :ref:`howto-model-algorithm`.

* The ``parameters`` dictionary must contain all the parameters that the
  relevant model expects (see the associated API documentation, especially that
  of the ``ModelData`` subclass) and that is not included in ``configurations``.

  * It might be useful to look at the existing YAML files that use the same
    model as the ``configurations`` are likely to be the same. Normally, only
    the values of some of the parameters are likely to be different between
    different use-cases of the same model.

  * Ultimately, the ``parameters`` and ``configurations`` depend on the
    mathematics/physics of the :term:`model` and the physical :term:`INS`
    :term:`instrument`. If unsure, and especially when contributing to ResINS,
    do not hesitate to contact us on
    `our GitHub <https://github.com/pace-neutrons/resins>`_.


.. _howto-model-algorithm:

How to add a new algorithm for a new model
------------------------------------------

Creating a new :term:`model` that uses new physics/mathematics - ones that are
not yet implemented in ResINS - is significantly more work than the case above,
since Python code will have to be written. Though, before we start, some notes
on the procedures for different outcomes:

* For personal use, the new code can be placed wherever - it will
  have to be registered with ResINS as explained below.

* If contributing to ResINS, please use a new file in
  ``resins/src/resins/models``.


.. _howto-model-dataclass:

How to add a new model data class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step should be creating a new class that will hold all the data
accessible to the new :term:`model`. It does not have to be completed
immediately - it is ok to continue adding to it as the model is being developed
- but it is a good starting point since the model class will need this class.

The data class **must be** a subclass of
:py:class:`resins.models.model_base.ModelData` (please read its
documentation for details on how it and its subclasses are supposed to work)
**and** it must be decorated with the :py:func:`dataclasses.dataclass` decorator
with the following arguments:

* ``init=True``
* ``repr=True``
* ``frozen=True``
* ``slots=True``
* ``kw_only=True``

Thus, for example:

.. code-block:: Python

    from dataclasses import dataclass
    from resins.models.model_base import ModelData

    @dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
    class TestModelData(ModelData):
        param1: int
        param2: bool

The parameters in this class should be specified as is normal for a
``dataclass`` and should represent *all* the parameters required by the model.
I.e., these should be the combination of the values from the YAML file
:ref:`parameters<spec-parameters>` and from the chosen
:ref:`options<spec-configurations>`. How these required values will be split
between the two places does not matter for the Python code as
:py:meth:`~resins.instrument.Instrument.get_resolution_function`
combines the two before creating the data object (i.e. ``TestModelData``).

.. note::

    If the YAML file is going to contain default values and/or restrictions on
    some arguments for the model (e.g. the model takes ``e_init`` and YAML file
    specifies that the default value is ``100`` and that only values between
    ``10`` and ``1000`` are allowed), you will need to reimplement the
    :py:attr:`resins.models.model_base.ModelData.defaults` and/or
    :py:attr:`resins.models.model_base.ModelData.restrictions`
    properties (the documentation does not need to be overwritten, just the
    code). For example:

.. code-block:: Python

    from dataclasses import dataclass
    from typing import Any
    from resins.models.model_base import ModelData

    @dataclass(init=True, repr=True, frozen=True, slots=True, kw_only=True)
    class TestModelData(ModelData):
        default_e_init: int
        e_init_restrictions: list[int]

        @property
        def defaults(self) -> dict[str, list[int | float]]:
            return {'e_init': self.default_e_init}

        @property
        def restrictions(self) -> dict[str, Any]:
            return {'e_init': self.e_init_restrictions}


How to add a new model class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With the data class in place, it is possible to create the model, which is a
subclass of
:py:class:`resins.models.model_base.InstrumentModel` (see its
documentation for detailed specification of how to inherit from it). This
**must** specify three class-level variables:

* ``input`` - an integer specifying the number of arguments that the `
  `__call__`` method takes

* ``output`` - an integer specifying the number of outputs the ``__call__``
  method returns

* ``data_class`` - a reference to the data class
  :ref:`created above<howto-model-dataclass>`, for example:

.. code-block:: Python

    class TestModel(InstrumentModel):
        input = 1
        output = 1
        data_class = TestModelData


Next, the ``__init__`` method should be defined. The function signature:

* *Must* take ``model_data`` (an instance of the data class above) as its
  first argument.
* *May* take any number of other argument (usually representing
  :term:`settings<setting>`, but may be others).
* *Must not* expect the independent variables (i.e. the variables that the
  model is a function of such as energy transfer or momentum).
* *Must* take ``**kwargs``.

The body of ``__init__()``:

* *Must* call ``super().__init__``.
* *Should* (if necessary) perform any validation of the arguments, e.g. that
  the ``e_init`` is within the allowed range etc.
* *Should* perform as much of the calculation as possible without the
  independent variables. These pre-computed, intermediate values should be
  stored as instance variables.

  * For more complex calculations, it might be advisable to break them up into
    multiple methods. Any such methods should be private (i.e. start with ``_``)
    and, if possible, should be made ``@staticmethod`` or ``@classmethod``.

* It *should not* keep a reference to ``model_data``.

For example:

.. code-block:: Python

    from resins.models.model_base import InstrumentModel, InvalidInputError

    class TestModel(InstrumentModel):
        def __init__(self, model_data: TestModelData, e_init: float | None = None, **_):
            super().__init__(model_data)

            if e_init is None:
                e_init = model_data.default_e_init
            elif not model_data.e_init_restrictions[0] <= e_init <= model_data.e_init_restrictions[1]:
                raise InvalidInputError('Good message')

            self.useful_value = 0.5 * e_init ** 3

Lastly, the ``__call__`` method must be implemented:

* It *must* take as arguments all the independent variables that the model
  models the :term:`resolution` as a function of.
* It *must* accept ``*args`` and ``**kwargs``.
* It *should* perform the remaining computation of the resolution, using the
  instance variables.
* It *must* return the resolution at the values of independent variables
  provided via the arguments.

For example:

.. code-block:: Python

    from jaxtyping import Float
    import numpy as np
    from resins.models.model_base import InstrumentModel

    class TestModel(InstrumentModel):
        def __call__(self, frequencies: Float[np.ndarray, 'frequencies'], *args, **kwargs
                     ) -> Float[np.ndarray, 'sigma']:
            return frequencies * self.useful_value

Then, with the model code complete, all that remains is to register it with
ResINS:

* If the above code is outside the ResINS repo (i.e. for personal use),
  somewhere in your program (before the model is intended to be used) a
  new key-value pair has to be inserted into
  :py:const:`resins.models.MODELS`:

.. code-block:: Python

    from resins.models import MODELS
    from resins import Instrument

    from custom_model_source import TestModel

    MODELS['test_model'] = TestModel

    instr = Instrument.from_file('path/to/data.yaml', 'version')
    model = instrument.get_resolution_function(model_name='model1', e_init=200)
    assert isinstance(model, TestModel)

* If the above code is inside the ResINS repo (i.e. to be submitted to the
  code-base), the :py:const:`resins.models.MODELS` dictionary
  (found at ``resins/src/resins/models/__init__.py``)
  has to be modified by adding a new key-value pair, where the key is the
  "name" of the function and the value is a reference to the above-created
  model.

  * The key can be anything as it is not exposed to the user - it is only
    present here and in the YAML data files, but it should be somewhat relevant.
    The only thing that matters is that it is globally unique.

.. code-block:: Python

    from resins.models.test_model import TestModel

    MODELS = {
        ...
        'test_model': TestModel,
    }


How to add the data
^^^^^^^^^^^^^^^^^^^

Now, with all the code in place, only the data that the model will use has to be
added. Since the code is written, the case has effectively become the same as
:ref:`adding a parameter-only model<howto-model>` (see the guide for more
details). The only difference is that, in this case, it is possible to tweak the
code (if necessary) to make everything nicer. Along with that, though, comes the
responsibility of structuring the data appropriately - there is no example among
the other YAML files to look to. How to do this is up to you, but some points of
advice are:

* The ``configurations`` section should reflect the physical INS instrument,
  see :term:`configuration`.

  * All parameters that change depending on which :term:`option` for a
    :term:`configuration` is chosen should be in the ``configurations`` section.
    The should not be any parameters that are present in both the ``parameters``
    and ``configurations`` sections.

    * There is no advice for parameters that depend on a combination of multiple
      different :term:`configurations<configuration>` -
      `contact the maintainers <https://github.com/pace-neutrons/resins>`_

* If the ``parameters`` section contains a large number of parameters, it can be
  a good idea to group some of these parameters into dictionaries.

  * This is especially recommended if there is a logical reason for the
    grouping, for example grouping all the moderator parameters together.

  * To maintain type hinting, the corresponding fields in the Python (the model
    data class) can be made in :py:class:`typing.TypedDict`:

.. code-block:: yaml

    model1_v1:
        parameters:
            param1: 1
            param2: 2
            param3: 3
            param4: 4

and

.. code-block:: Python

    class TestModelData(ModelData):
        param1: int
        param2: int
        param3: int
        param4: int

can be changed into:

.. code-block:: yaml

    model1_v1:
        parameters:
            param1: 1
            group1:
                param2: 2
                param3: 3
                param4: 4

and

.. code-block:: Python

    class TestModelData(ModelData):
        param1: int
        group1: Group1

    class Group1(TypedDict):
        param2: int
        param3: int
        param4: int

