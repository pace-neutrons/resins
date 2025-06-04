How To Add a Version To an Instrument
=====================================

ResINS strives to model all :term:`versions<version>` of any given :term:`INS`
:term:`instrument` - old, new, and planned - but it is inevitable that some will
not be natively supported *yet*. Fortunately, while not as straightforward as
:doc:`creating a new instrument<add_instrument>`, ResINS aims to make adding
new :term:`versions<version>` as simple as possible:

.. contents::
    :backlinks: entry
    :depth: 2
    :local:


How to add a version for personal use
-------------------------------------

Adding a :term:`version` to an :term:`instrument` only locally (without
:ref:`submitting to ResINS<howto-version-resins>`) is the more difficult task
since ResINS does not have an extension/add-ons mechanism. Therefore, to create
a version, it is necessary to resort to either creating an entire instrument or
working with Python dictionaries instead of YAML files:

.. _howto-version-yaml:

Using YAML files
^^^^^^^^^^^^^^^^

It is currently impossible to dynamically append a new :term:`version` to an existing
:term:`instrument`. It is necessary to create a new instrument YAML file into
which the new :term:`version` can be placed. This can be done in two ways:

* Copy the YAML data file of the relevant :term:`instrument` from ResINS

  * The ResINS data files can be found at ``resins/instrument_data``
  * The file should be copied to a different location and the new path saved

* Create a new YAML file (see :doc:`add_instrument`)

  * The top-level information can be the same as in the original
    :term:`instrument` or it can be gibberish

.. warning::

    While it is possible to add a new version by modifying the instrument's YAML
    data file that comes with ResINS in-place (without copying), this file
    may be overwritten when updating ResINS with ``pip``, resulting in
    **data loss**.

With an instrument file present, all that is required is to add the data for
the new :term:`version` to the the YAML file, which can be done by adding a new
key to the ``version`` dictionary of the YAML file. The corresponding value to
this key should be a dictionary containing the data for the :term:`version`. See
:doc:`../dev/yaml_spec` for more information on how to structure the YAML data
file. Then, ResINS can be leveraged to do the rest without additional code:

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


Using a dictionary
^^^^^^^^^^^^^^^^^^

An :py:cls:`~resolution_functions.instrument.Instrument` can be constructed
directly from a dictionary containing the data for a particular :term:`version`.
In this case, the process is identical to
:ref:`creating a new instrument using a dictionary<howto-instrument-dict>`.


.. _howto-version-resins:

How to add a version to ResINS
------------------------------

If you would like to contribute a new :term:`version` to ResINS (which we do
appreciate!), do open an issue on
`our GitHub <https://github.com/pace-neutrons/resolution_functions>`_
so that we can help. Either way, the process to do so is similar to but simpler
than :ref:`creating a version using YAML files<howto-version-yaml>`. The crux of
the difference is that now there is no need to worry about creating files - all
that is required is to edit the relevant :term:`instrument's<instrument>` YAML
file (found at ``resins/src/resins/instrument_data``) in-place (of course
working on a new branch of your own fork, see Contributing Guidelines). The data
will have to be correctly formatted according to the
:doc:`spec<../dev/yaml_spec>` by adding a new entry to the ``version``
dictionary with a unique (version) name and the correct data.

There is no need to edit any Python code.
