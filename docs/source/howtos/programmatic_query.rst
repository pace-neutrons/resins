How To Programmatically Find Data
*********************************

The :term:`instrument`/:term:`model` data used by ResINS is available in the
:doc:`documentation</instruments>` and comes packaged together with the library
via the YAML files in ``resins/instrument_data``. However,
one may also obtain the data programmatically at runtime; this could be useful
when accessing ResINS through other software to ensure the information is
up-to-date with the code. This guide shows how to to this:

.. contents::
    :backlinks: entry
    :depth: 3
    :local:


How To Find Data In Computer-compatible Format
==============================================

Most of the methods shown here return Python data objects such as lists, dictionaries, or strings.
These are generally human-readable,
but for some larger sets of data additional methods have been
provided that format the information into nice-looking tables (see
:ref:`howto-query-human`).


.. _how-to-instrument:

How To Find the Available Instruments
-------------------------------------

The :py:class:`~resins.instrument.Instrument` provides a method,
:py:meth:`~resins.instrument.Instrument.available_instruments`,
which lists all the available :term:`instruments<instrument>`:

>>> from resins import Instrument
>>> Instrument.available_instruments()
['ARCS', 'CNCS', 'HYSPEC', 'Lagrange', 'LET', 'MAPS', 'MARI', 'MERLIN', 'PANTHER', 'TFXA', 'TOSCA', 'VISION', 'SEQUOIA', 'TOSCA1']


Advanced
^^^^^^^^

An advanced way of interacting with available instruments is to access the
``INSTRUMENT_MAP`` dictionary directly:

>>> from resins.instrument import INSTRUMENT_MAP
>>> INSTRUMENT_MAP
{'ARCS': ('arcs.yaml', None), 'CNCS': ('cncs.yaml', None), 'HYSPEC': ('hyspec.yaml', None), 'Lagrange': ('lagrange.yaml', None), 'LET': ('let.yaml', None), 'MAPS': ('maps.yaml', None), 'MARI': ('mari.yaml', None), 'MERLIN': ('merlin.yaml', None), 'PANTHER': ('panther.yaml', None), 'TFXA': ('tosca.yaml', 'TFXA'), 'TOSCA': ('tosca.yaml', None), 'VISION': ('vision.yaml', None), 'SEQUOIA': ('sequoia.yaml', None), 'TOSCA1': ('tosca.yaml', 'TOSCA1')}

which maps the user-input instrument name to the instrument data file and version
name. There should be no reason to access this object for normal use.


.. _how-to-version:

How To Find the Versions Available for an Instrument
----------------------------------------------------

Once again, the :py:class:`~resins.instrument.Instrument` provides
a method,
:py:meth:`~resins.instrument.Instrument.available_versions`,
which lists all the :term:`versions<version>` available for an
:term:`instruments<instrument>`:

>>> from resins import Instrument
>>> available_versions, default_version = Instrument.available_versions('TOSCA')
>>> available_versions
['TFXA', 'TOSCA1', 'TOSCA']
>>> default_version
'TOSCA'

This method returns both the list of versions available for the instrument, and
the default version for that instrument. If the instrument being
searched for is an alias for a particular version of an instrument, that version
is returned instead of the default version:

>>> available_versions, default_version = Instrument.available_versions('TFXA')
>>> available_versions
['TFXA', 'TOSCA1', 'TOSCA']
>>> default_version
'TFXA'

.. _how-to-model:

How To Find the Models Available for an Instrument
--------------------------------------------------

Given an :ref:`instrument<how-to-instrument>` and its
:ref:`version<how-to-version>`, it is possible to query the list of available
:term:`models<model>` by using the
:py:meth:`~resins.instrument.Instrument.available_models`
property:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.available_models
['AbINS', 'book', 'vision']

The default model can similarly be accessed via an attribute:

>>> tosca.default_model
'AbINS'

Advanced
^^^^^^^^

The above property returns only the recommended (usually latest) version of each
model. These "models" are actually aliases that each point to a versioned model
which holds the data. These versioned models can be listed by using
:py:meth:`~resins.instrument.Instrument.available_unique_models`:

>>> tosca.available_unique_models
['AbINS_v1', 'book_v1', 'vision_v1']

Then, to round out the options, *all* models can be listed via
:py:meth:`~resins.instrument.Instrument.all_available_models`:

>>> tosca.all_available_models
['AbINS', 'AbINS_v1', 'book', 'book_v1', 'vision', 'vision_v1']

.. important::

    Older versions of models may contain known bugs and/or inaccuracies - use at own
    risk.


.. _how-to-config:

How To Find the Configurations that Must be Chosen for a Model
--------------------------------------------------------------

The :term:`configurations<configuration>` required by a given :term:`model` can
be retrieved using the
:py:meth:`~resins.instrument.Instrument.possible_configurations_for_model`
method:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_configurations_for_model('AbINS')
[]
>>> tosca.possible_configurations_for_model('book')
['detector_bank']

.. seealso::

    :ref:`howto-query-models-configs` shows how to list all
    :term:`configurations<configuration>` for all :term:`models<model>` in the
    form of a nicely formatted table.


.. _howto-query-options:

How To Find the Options Available for a Configuration
-----------------------------------------------------

To obtain the :term:`options<option>` as Python data, the
:term:`model` must already be known.
(Humans can read the formatted table from :ref:`otherwise<howto-query-models-configs-options>`.)

If the model and setting are known
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To list the possible :term:`options<option>` for a given :term:`configuration`
(:ref:`how to find configurations<how-to-config>`) of a given :term:`model`
(:ref:`how to find models<how-to-model>`), the
:py:meth:`~resins.instrument.Instrument.possible_options_for_model_and_configuration`
method is provided:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_options_for_model_and_configuration('book', 'detector_bank')
['Backward', 'Forward']

If only the model is known
^^^^^^^^^^^^^^^^^^^^^^^^^^

To list all the :term:`options<option>` for all :term:`configurations<configuration>` of a
given :term:`model` (:ref:`how to find models<how-to-model>`), the
:py:meth:`~resins.instrument.Instrument.possible_options_for_model`
method is provided:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.possible_options_for_model('book')
{'detector_bank': ['Backward', 'Forward']}


How To Find the Default Option for a Configuration
--------------------------------------------------

Given the :term:`model` name (:ref:`how to find models<how-to-model>`) and the
:term:`configuration` (:ref:`how to find configurations<how-to-config>`),
the default option can be retrieved using the
:py:meth:`~resins.instrument.Instrument.default_option_for_configuration`
method:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> tosca.default_option_for_configuration('book', 'detector_bank')
'Backward'


How To Find the Default Values for a Setting
--------------------------------------------

The default values for all :term:`settings<setting>` associated with a
:term:`model` can be found using the ``default`` attribute of the model, which
can be retrieved using the
:py:meth:`~resins.instrument.Instrument.get_model_data` method:

>>> from resins import Instrument
>>> merlin = Instrument.from_default('MERLIN', 'MERLIN')
>>> model = merlin.get_model_data('PyChop_fit')
>>> type(model)
<class 'resins.models.pychop.PyChopModelDataFermi'>
>>> model.defaults
{'e_init': 400, 'chopper_frequency': 400}

.. warning::

    For some :term:`models<model>`, the :term:`configurations<configuration>`
    may affect the default values of the :term:`settings<setting>`.


How To Find the Restrictions on the Values of a Setting
-------------------------------------------------------

The restrictions on the values for all :term:`settings<setting>` associated with a
:term:`model` can be found using the ``restrictions`` attribute of the model, which
can be retrieved using the
:py:meth:`~resins.instrument.Instrument.get_model_data` method:

>>> from resins import Instrument
>>> merlin = Instrument.from_default('MERLIN', 'MERLIN')
>>> model = merlin.get_model_data('PyChop_fit')
>>> type(model)
<class 'resins.models.pychop.PyChopModelDataFermi'>
>>> model.restrictions
{'chopper_frequency': [50, 601, 50], 'e_init': [0, 181]}

.. warning::

    For some :term:`models<model>`, the :term:`configurations<configuration>`
    may affect the restrictions on the :term:`settings<setting>`:

>>> model = merlin.get_model_data('PyChop_fit', chopper_package='S')
>>> type(model)
<class 'resins.models.pychop.PyChopModelDataFermi'>
>>> model.restrictions
{'chopper_frequency': [50, 601, 50], 'e_init': [7, 2000]}


.. _howto-query-human:

How To Find Data In Human-readable Format
=========================================

Unlike the cases above, in some cases it might be more desirable to obtain
comprehensive information in an easily legible format. The following are provided
for that purpose:

.. _howto-query-models-configs:

How To Find the Models and Their Configurations
-----------------------------------------------

All :term:`configurations<configuration>` for all the :term:`models<model>` can
be displayed via the
:py:meth:`~resins.instrument.Instrument.format_available_models_and_configurations`
method:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> print(tosca.format_available_models_and_configurations())
|--------------|--------------|-------------------|
| MODEL        | ALIAS FOR    | CONFIGURATIONS    |
|==============|==============|===================|
| AbINS        | AbINS_v1     |                   |
|--------------|--------------|-------------------|
| AbINS_v1     |              |                   |
|--------------|--------------|-------------------|
| book         | book_v1      |                   |
|--------------|--------------|-------------------|
| book_v1      |              | detector_bank     |
|--------------|--------------|-------------------|
| vision       | vision_v1    |                   |
|--------------|--------------|-------------------|
| vision_v1    |              | detector_bank     |
|--------------|--------------|-------------------|
<BLANKLINE>


.. _howto-query-models-configs-options:

How To Find the Models and Their Configurations and Their Options
-----------------------------------------------------------------

All the :term:`options<option>` for all the
:term:`configurations<configuration>` of all :term:`models<model>` can be listed
by the
:py:meth:`~resins.instrument.Instrument.all_available_models_options`
method:

>>> from resins import Instrument
>>> tosca = Instrument.from_default('TOSCA', 'TOSCA')
>>> print(tosca.format_available_models_options())
|--------------|--------------|-------------------|-----------------------|
| MODEL        | ALIAS FOR    | CONFIGURATIONS    | OPTIONS               |
|==============|==============|===================|=======================|
| AbINS        | AbINS_v1     |                   |                       |
|--------------|--------------|-------------------|-----------------------|
| AbINS_v1     |              |                   |                       |
|--------------|--------------|-------------------|-----------------------|
| book         | book_v1      |                   |                       |
|--------------|--------------|-------------------|-----------------------|
| book_v1      |              | detector_bank     | Backward (default)    |
|              |              |                   | Forward               |
|--------------|--------------|-------------------|-----------------------|
| vision       | vision_v1    |                   |                       |
|--------------|--------------|-------------------|-----------------------|
| vision_v1    |              | detector_bank     | Backward (default)    |
|              |              |                   | Forward               |
|--------------|--------------|-------------------|-----------------------|
<BLANKLINE>
