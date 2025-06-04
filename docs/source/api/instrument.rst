resolution\_functions.instrument
--------------------------------

.. automodule:: resolution_functions.instrument
   :members:
   :show-inheritance:


.. autodata:: INSTRUMENT_MAP
    :no-value:

    A mapping of the name of an :term:`instrument` to its data file and the
    :term:`version` it's aliasing, for all available instruments.

    Has the form ``{'instrument_name': ('file_name.yaml', alias_for)}`` where
    ``instrument_name`` is the public, official name of the instrument that is
    used when creating it, ``file_name.yaml`` is the name of the YAML data file
    containing the data for that instrument (without the full path), and
    ``alias_for`` is either

    * a `str` containing the name of a version in that file; or

      * In this case, when ``instrument_name`` is accessed, the **specified**
        version is returned.

    * a `None`.

      * In this case,when ``instrument_name`` is accessed, the **default**
        version is returned.