Testing and reference data
==========================

The test suite in ``tests/`` uses ``pytest``; the requirements are
specified in the ``test`` group::

   pip install . --group test
   pytest -n auto

or::

   uv run --group test pytest

Comparison tests
----------------

The tests in ``tests/comparison_tests/`` compare the resins models
with other implementations of the same resolution functions: Mantid's
bundled PyChop, and Abins (PyChop-based, PANTHER, TOSCA and
Lagrange). The reference data for these is located in
``tests/reference_data/``, defining both the test cases and the
expected result of each:

- a reference value, compared within a stated tolerance;
- conditions allowing no neutron transmission, which resins must report with a
  :py:class:`~resins.models.pychop.NoTransmissionError`;
- or an incident energy or chopper frequency outside the allowed range, which resins
  must reject with an :py:class:`~resins.models.model_base.InvalidInputError`.

Points on both sides of each transmission edge and of each input limit are
included, as is every allowed chopper frequency of every configuration.
When adding an instrument, model version or configuration to resins,
reference data must be generated and committed alongside the changes
(see ``dev/reference/README.md``).

Known disagreements with the reference are listed in
``tests/comparison_tests/reference.py`` and are strict expected failures, so
fixing one makes the suite fail until its entry is removed.

Regenerating the reference data
-------------------------------

Reference data is generated with Mantid in locked `pixi
<https://pixi.sh>`_ environments under ``dev/reference/``, and
committed to the Git history.  A non-destructive sanity check
(``pixi run validate``) is provided as well as the destructive
refresh operation (``pixi run update``).
See `dev/reference/README.md <https://github.com/pace-neutrons/resins/blob/main/dev/reference/README.md>`_ for more information about these maintenance workflows.
