YAML Data File Specification
============================

This file constitutes the official specification for the YAML data files used
to store the data for an :term:`instrument`.

Spec
----

The data must be stored in a YAML format file with the following structure:

.. parsed-code-block:: yaml
    :emphasize-lines: 1,2,3,5,8,9,10,15,17

    :ref:`name<spec-name>`: :iref:target:`"instrument_name"<spec-name-targ>`
    :ref:`default_version<spec-default-version>`: :iref:target:`"version1"<spec-default-version-targ>`
    :ref:`version<spec-version>`\ :iref:target:`:<spec-version-targ>`
        version1:
            :ref:`default_model<spec-default-model>`: :iref:target:`"model1"<spec-default-model-targ>`
            :ref:`models<spec-models>`\ :iref:target:`:<spec-models-targ>`
                model1: "model1_v1"
                model1_v1:
                    :ref:`function<spec-function>`: :iref:target:`"model_function_name"<spec-function-targ>`
                    :ref:`citation<spec-citation>`: ["citation1", :iref:target:`"citation2"<spec-citation-targ>`]
                    :ref:`parameters<spec-parameters>`\ :iref:target:`:<spec-parameters-targ>`
                        parameter1: "value1"
                        parameter2: 2
                        parameter3: 3.14
                        parameter4: [1, 2, 3, 4]
                    :ref:`configurations<spec-configurations>`\ :iref:target:`:<spec-configurations-targ>`
                        configuration1:
                            :ref:`default_option<spec-default-option>`: :iref:target:`"option1"<spec-default-option-targ>`
                            option1:
                                parameter5: "value2"
                                parameter6: 56.1687


The highlighted lines contain keys with **preset names** that **must not** be
changed. The names of the remaining keys as well as the values can and should be
changed appropriately. Further, these freeform keys are free to have multiples
of (e.g. `version1`, `version2`, see :ref:`spec-example`), given that their
substructure is kept.

.. _spec-name:

name
^^^^

This key (:iref:ref:`see in spec<spec-name-targ>`) specifies the name of the
:term:`instrument`, set as a string (`"instrument_name"`). This *should* be the
public, official name that *should* be used everywhere else in ResINS. This is
the name that :py:attr:`resolution_functions.instrument.Instrument.name` is set
to and that will be shown when printing ``Instrument``, i.e.
``print(instrument)``.

*This* name does *not* have to be unique, but since it is recommended to match
the other uses of the same instrument in ResINS, it **should** be globally
unique.

.. _spec-default-version:

default_version
^^^^^^^^^^^^^^^

This key (:iref:ref:`see in spec<spec-default-version-targ>`) specifies the name
of the :term:`version` that will be used by default for this :term:`instrument`
when user does not specify which :term:`version` they want to use, i.e. calling
:py:meth:`~resolution_functions.instrument.Instrument.from_default` with only
one argument, e.g. ``Instrument.from_default('TOSCA')``.

The value of this key, specified as a string, **must** match one of the version
keys (see :ref:`version<spec-version>`).

.. _spec-version:

version
^^^^^^^

This key (:iref:ref:`see in spec<spec-version-targ>`) contains all the data for
all the :term:`versions<version>`. It must be a (YAML) dictionary in which each
key is the name of an :term:`instrument` :term:`version` and its corresponding
value is another dictionary with the associated data.

.. warning::

    All of the entries in this dictionary **will** be interpreted as
    :term:`versions<version>` - no other data is permissible in this section. If
    anything not following the below guidelines is placed in the dictionary, it
    will lead to errors.

All the subkeys (:term:`version` names) must be mutually unique, but none has
to be globally unique, though it is recommended, if possible. Regardless,
though, each of the subkeys *must not* be arbitrary - it should represent an
official name for the given :term:`version`.

Each value for the subkey (:term:`version` name) in the dictionary **must** be a
correctly formatted data for an :term:`instrument` :term:`version` in the form
of a (YAML) dictionary. That said, though, this inner dictionary has less strict
specification - the only requirement is that it contains a key called
:ref:`models<spec-models>`. In fact, this space is encouraged to be used for
storing shared data (see :ref:`spec-yaml-magic`).


.. _spec-default-model:

default_model
^^^^^^^^^^^^^

This key (:iref:ref:`see in spec<sspec-default-model-targ>`), found inside the
(YAML) dictionary corresponding to a particular :term:`instrument`
:term:`version` (see the :ref:`version key<spec-version>`), specifies the name
of the :term:`model` that will be used by default when the user does not specify
which :term:`model` they want to use, e.g. when calling
:py:meth:`resolution_functions.instrument.Instrument.get_resolution_function`.

The value of this key, specified as a string, **must** match one of the model
keys (see :ref:`version<spec-models>`).


.. _spec-models:

models
^^^^^^

This key (:iref:ref:`see in spec<spec-models-targ>`), found inside the (YAML)
dictionary corresponding to a particular :term:`instrument` :term:`version` (see
the :ref:`version key<spec-version>`), contains all the data for all the
:term:`models<model>`. Its value must be a (YAML) dictionary in which each key
is the name of a :term:`model` and its corresponding value is either:

* Another dictionary with the associated data

  * In this case, the key (:term:`model` name) **must** include a version number
    in the form ``{model_name}_v{version_number}``, e.g. ``PyChop_fit_v1``,
    where the ``version_number`` is an integer.

* A string whose value matches one of the keys *whose value is a dictionary*.
  Chaining *will* lead to errors.

  * In this case, the key (:term:`model` name) **must not** include a
    version number.


.. warning::

    All of the entries in this inner dictionary **will** be interpreted as
    :term:`models<model>` - no other data is permissible in this section. If
    anything not following the below guidelines is placed in the dictionary, it
    will lead to errors.

All the subkeys (:term:`model` names) must be mutually unique, but none has
to be globally unique - in fact, if a model is applicable to multiple
:term:`instruments<instrument>` or :term:`versions<version>`, it is recommended
that the same name is used for that :term:`model` in each YAML file. Regardless,
though, each of the subkeys *must not* be arbitrary - it should represent an
official name for the given :term:`model`.

Each value for the subkey (:term:`model` name) in the dictionary **must** be a
correctly formatted data for a :term:`model` in the form of a (YAML) dictionary.
That said, though, this inner dictionary has less strict specification - the
only requirement is that it *must* contain the following keys:

* :ref:`function<spec-function>`
* :ref:`citation<spec-citation>`
* :ref:`parameters<spec-parameters>`
* :ref:`configurations<spec-configurations>`

Otherwise, other entries for the dictionary are not defined and may similarly
be used for storing shared data (see :ref:`spec-yaml-magic`), so long as they do
not clash with the names above.

.. _spec-function:

function
^^^^^^^^

This key (:iref:ref:`see in spec<spec-function-targ>`), found inside the (YAML)
dictionary corresponding to a particular :term:`model`, (see the
:ref:`model key<spec-models>`), specifies the exact ResINS :term:`model` object
that will be instantiated when a user wants to use the particular :term:`model`.
The value for this key is a string.


.. important::

    The value for this key **must** correspond to one of the keys in
    :py:data:`resolution_functions.models.MODELS` (and therefore must be
    globally unique. For creating a new model, see :doc:`../howtos/add_model`.


.. _spec-citation:

citation
^^^^^^^^

This key (:iref:ref:`see in spec<spec-citation-targ>`), found inside the (YAML)
dictionary corresponding to a particular :term:`model`, (see the
:ref:`model key<spec-models>`), specifies the citations/references associated
with the particular :term:`model` of the particular :term:`instrument`. These
are exposed to the user as-is via ``ModelData.citation`` and
``InstrumentModel.citation``.

The value corresponding to this key must be a list of strings, where each string
is a shortened citation (only initials and last name, no paper title, etc.).
There is no requirement for citation style beyond that, though the DOI should
be included if there is one.


.. _spec-parameters:

parameters
^^^^^^^^^^

This key (:iref:ref:`see in spec<spec-parameters-targ>`), found inside the (YAML)
dictionary corresponding to a particular :term:`model`, (see the
:ref:`model key<spec-models>`), specifies all the parameters required by the
particular :term:`model`. Its value must be a (YAML) dictionary in which each
key is the name of a parameter of that model, and the value is a valid value for
that parameter of that model.

There are no intrinsic restrictions on this dictionary, but it must contain
**exactly** the parameters required by the ResINS model specified by the
:ref:`function value<spec-function>`. There can be no missing or extra
parameters, though please note that some of the parameters required by the model
may be stored in the :ref:`configurations dictionary<spec-configurations>`. The
values must match the arguments expected by the associated ``ModelData``
subclass, which means that the type of each parameter could be anything -
``int``, ``float``, ``string``, ``list``, ``dict`` - as long as the
``ModelData`` expects it. In fact, when
:doc:`creating new models<../howtos/add_model>`, it is encouraged to further
structure the data if there are many parameters.


.. _spec-configurations:

configurations
^^^^^^^^^^^^^^

This key (:iref:ref:`see in spec<spec-configurations-targ>`), found inside the
(YAML) dictionary corresponding to a particular :term:`model`, (see the
:ref:`model key<spec-models>`), specifies all the
:term:`configurations<configuration>` available to the particular :term:`model`.
Its value must be a (YAML) dictionary in which each key is the name of a
:term:`configuration`, and the corresponding value is the data associated with
the :term:`configuration`. This data consists of two different things:

* The :ref:`default_option<spec-default-option>` key
* The various :term:`options<option>` associated with the :term:`configuration`.

Besides the special :ref:`default_option<spec-default-option>` entry, all the
other entries in this inner dictionary **will** be interpreted as
:term:`options<option>` - no other data is permissible in this section. If
anything not following the below guidelines is placed in the dictionary, it
will lead to errors.

All the subkeys (:term:`option` names) must be mutually unique, but none needs
to be globally unique. The only thing that matters is that they *must not* be
arbitrary - each subkey should represent an official name for the given
:term:`option`.

Each value for the subkey (:term:`option` name) in the dictionary **must** be a
correctly formatted data for an :term:`option` in the form of a (YAML)
dictionary. Each key in *this* dictionary must be a parameter of the associated
model and its value a valid value for that parameter of that model. Each
entry must contain **all** the parameters that :term:`configuration` can change;
shared values should be handled via :ref:`spec-yaml-magic`.

Similar to :ref:`parameters<spec-parameters>`, there are no restrictions on the
values for the entries in this dictionary except those placed by the relevant
``ModelData``. The parameters in the :ref:`parameters section<spec-parameters>`
and those in this section must together make up **exactly** the parameters
required by the ``ModelData``.

.. important::

    While, in the
    :py:meth:`~resolution_functions.instrument.Instrument.get_resolution_function`
    method, the :ref:`configurations<spec-configurations>` override the
    :ref:`parameters<spec-parameters>`, using this fact is **heavily discouraged**
    because *it is not guaranteed*.


.. _spec-default-option:

default_option
^^^^^^^^^^^^^^

This key (:iref:ref:`see in spec<spec-default-option-targ>`), found inside the
(YAML) dictionary corresponding to a particular :term:`configuration`, (see the
:ref:`configurations key<spec-configurations>`), specifies the name of the
:term:`option` that will be used by default for this :term:`configuration`
when user does not specify which :term:`option` they want to use, i.e. calling
:py:meth:`~resolution_functions.instrument.Instrument.get_resolution_function`
without specifying the configuration, e.g.
``maps.get_resolution_function('PyChop_fit')``.

The value of this key, specified as a string, **must** match one of the option
keys (see :ref:`configurations<spec-configurations>`).


.. _spec-yaml-magic:

YAML magic
----------

To avoid repetition and prevent errors, the use of
`anchors and aliases <https://yaml.org/spec/1.1/current.html#id863390>`_
is encouraged. This allows for data to be set only once and used in multiple
places, keeping the files smaller and hopefully avoiding bugs. That said, the
shared data has to be placed somewhere where it will not clash with the
expectations that ResINS has, as it still remains in its original location
when expanded by the YAML parser. There are multiple such places:

* At the top level of the file
* Inside the dictionary of a specific :ref:`version<spec-version>`
* Inside the dictionary of a specific :ref:`model<spec-models>`




.. _spec-example:

Example
-------

.. code-block:: yaml

    name: "instrument"
    default_version: "new_version"
    version:
        old_version:
            default_model: "model3"
            models:
                model3: "model3_v1"
                model3_v1:
                    function: "model3_function"
                    citation: ["https://mantid.org/docs/relevant-page.html"]
                    parameters:
                        fit: [0.6546, 2.10548, -9.5, -0.00004]
                    configurations: {}
                old_model: "old_model_v1"
                old_model_v1:
                    function: "old_function"
                    citation: ["A. Doof et. al., Sci. Mag., 1975, 1, 1-6."]
                    parameters:
                        distance: 1.5
                        length: 2e-2
                    configurations:
                        chopper_package:
                            default_option: "G"
                            G:
                                value1: 1
                            H:
                                value1: 2
                        analyzer:
                            default_option: "Forward"
                            Forward:
                                value2: 3
                            Backward:
                                value2: 4

        new_version:
            constants: &version1_constants
                distance: 2.0
                length: 1e-3
                allowed_e_init: [10, 1000]
                kind: "kind1"
                matrix:
                    [[1, 0],
                     [0, 1]]
                sample:
                    width: 1.0
                    height: 2.0
            choppers: &version1_choppers
                chopper: &version1_chopper
                    chopper1:
                        number: 2
                        size: 2.25
                    chopper2:
                        number: 1
                        size: 9.1
                    chopper3: &version1_chopper3
                        number: 4
                        size: 0.2

            configurations: &version1_configurations
                chopper_package:
                    default_option: "A"
                    A:
                        slit: 3.14e-3
                        <<: *version1_choppers
                    B:
                        slit: 1.88e-3
                        <<: *version1_choppers
                    C:
                        slit: 1.88e-3
                        chopper:
                            <<: *version1_chopper
                            chopper3:
                                <<: *version1_chopper3
                                size: 0.3

            default_model: "model1"
            models:
                model1: "model1_v3"
                model1_v1:
                    function: "model1_function"
                    citation: ["A. Yi, H. Wells, and Y. Li, Sci. Mag., 2009, 42, 700-706. https://doi.org/164648"]
                    parameters: *version1_constants
                    configurations: *version1_configurations
                model1_v2:
                    function: "model1_function_modified"
                    citation: ["A. Yi, H. Wells, and Y. Li, Sci. Mag., 2010, 44, 700-706. https://doi.org/164648"]
                    parameters: *version1_constants
                    configurations: *version1_configurations
                model1_v3:
                    function: "model1_function_modified"
                    citation: ["A. Yi, H. Wells, and Y. Li, Sci. Mag., 2015, 69, 700-706. https://doi.org/164648"]
                    parameters:
                        <<: *version1_constants
                        kind: "kind2"
                    configurations: *version1_configurations
                model2: "model2_v1"
                model2_v1:
                    function: "model2_function"
                    citation: ["Z. Zun et. al., Book On The Topic, Publisher, 1999. ISBN 000-000-000-0", "J. Adams et. al., Sci. Mag., 2000, 27, 1-12."]
                    parameters: *version1_constants
                    configurations: {}
                model3: "model3_v1"
                model3_v1:
                    function: "model3_function"
                    citation: ["https://mantid.org/docs/relevant-page.html"]
                    parameters:
                        fit: [1.6546, 0.10548, -99.5, 0.00004]
                    configurations: {}

Validation
----------

Validation of data files can be performed using a script found in the GitHub
repository at ``resolution_functions/dev/validate_data_file.py``.
