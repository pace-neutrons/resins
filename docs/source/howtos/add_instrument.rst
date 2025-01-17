How To Add an Instrument
========================

ResINS strives to model all :term:`INS` :term:`instruments<instrument>` past,
present, and future, but it is inevitable that some will not be natively
supported *yet*. Fortunately, ResINS was designed to make adding new instruments
relatively straightforward. Though, there are two ways of doing this:

.. contents::
    :backlinks: entry
    :depth: 2
    :local:

.. _add-personal-instrument:

How To Add an Instrument for Personal Use
-----------------------------------------

It is possible to add an :term:`instrument` only locally, outside the ResINS
ecosystem. Technically, all that is required is a dictionary, but the most
readable way is to construct a YAML file with the instrument details. This file
**must** follow the :doc:`data file spec<../dev/yaml_spec>` and include all the
necessary information. Then, ResINS can be leveraged to do the rest without
additional code:

>>> from resolution_functions import Instrument
>>> new_instrument_path = '~/instrument/instrument.yaml'
>>> version = 'version'  # If the created YAML file contains multiple versions
>>> new_instrument = Instrument.from_file(new_instrument_path, version)
>>> new_instrument.name
'new_instrument'
>>> new_instrument.version
'version'

Given that the YAML file specifies a model already implemented in ResINS, and
that its parameters have been all included, the resolution function can be
computed:

>>> model = new_instrument.get_resolution_function('PyChop_fit', chopper_package='A', e_init=100, chopper_frequency=300)
>>> print(model)
PyChopModelFermi(citation=[''])

However, if a new model was created for the new instrument, more work will be
required, see :doc:`add_model`.

How to register an instrument with ResINS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

??????????????????

How to add an instrument without creating a yaml file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For personal use, it is possible to also specify an instrument without creating
a YAML data file for it. All that is required is a Python dictionary, though the
dictionary still **must** follow the :doc:`data file spec<../dev/yaml_spec>`
(though only the dict inside :ref:`models: key<spec-models>`):

>>> from resolution_functions import Instrument
>>> new_instrument_name = 'name'
>>> new_instrument_version = 'v1'
>>> new_instrument_default_model = 'AbINS'
>>> new_instrument_data = {'AbINS': {'function': 'polynomial_1d', 'citation': [''], 'parameters': [0, 2], 'settings': {}}}
>>> new_instrument = Instrument(new_instrument_name, new_instrument_version, new_instrument_data, new_instrument_default_model)
>>> print(new_instrument)
Instrument(name=name, version=v1)
>>> model = new_instrument.get_resolution_function()
>>> print(model)
PolynomialModel1D(citation=[''])
>>> model(100)
200.0

How To Add an Instrument to ResINS
----------------------------------

If you would like to contribute a new :term:`instrument` to ResINS (which we do
appreciate!), do open an issue on
`our GitHub <https://github.com/pace-neutrons/resolution_functions>`_
so that we can help. Otherwise, the process will start out similar to when you
would :ref:`create an instrument for personal use<add-personal-instrument>` in
that you will need to create a new YAML file for the instrument, following the
:doc:`spec<../dev/yaml_spec>`. Then, the file will have to be placed in
``resolution_functions/src/resolution_functions/instrument_data`` (of course
working on a new branch of your own fork, see Contributing Guidelines). Lastly,
to be able to use the new instrument with
:py:meth:`~resolution_functions.instrument.Instrument.from_default`,
it has to be added to
:py:data:`resolution_functions.instrument.INSTRUMENT_MAP`;
create a new entry in the dictionary with the format::

    INSTRUMENT_MAP = {
        'instrument_name': ('yaml_file_name', None)
    }

where `instrument_name` is the official name of the instrument that you would
like users to use when creating the instrument, and `yaml_file_name` is the
name of the YAML file without the `.yaml` extension, e.g. `arcs`.

.. note::

    The `None` in the example above is used for creating an alias and represents
    a version name for one of the versions in the `yaml_file_name.yaml` file.
    For example, the TFXA instrument is an alias for the TFXA :term:`version` of
    the TOSCA :term:`instrument` and is specified as
    `'TFXA': ('tosca', 'TFXA')`.
