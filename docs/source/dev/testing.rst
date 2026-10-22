Testing and reference data
==========================

The test suite in ``tests/`` runs with plain ``pytest`` and needs no conda environment::

   pip install . --group test
   pytest -n auto

Comparison tests
----------------

The tests in ``tests/comparison_tests/`` compare the resins models with other
implementations of the same resolution functions: Mantid's bundled PyChop, and
Abins (PyChop-based, PANTHER, TOSCA and Lagrange). They do not run these
implementations. Instead, they read reference values committed in
``tests/reference_data/``, which define both the test cases and the expected
result of each:

- a reference value, compared within a stated tolerance;
- no neutron transmission, which resins must report with a
  :py:class:`~resins.models.pychop.NoTransmissionError`;
- an incident energy or chopper frequency outside the allowed range, which resins
  must reject with an :py:class:`~resins.models.model_base.InvalidInputError`.

Points on both sides of each transmission edge and of each input limit are
included, as is every allowed chopper frequency of every configuration. A
coverage test fails if an instrument, model version or configuration is added to
resins without reference data.

Known disagreements with the reference are listed in
``tests/comparison_tests/reference.py`` and are strict expected failures, so
fixing one makes the suite fail until its entry is removed.

Regenerating the reference data
-------------------------------

Reference data is generated with Mantid, in locked `pixi <https://pixi.sh>`_
environments under ``dev/reference/``, and only by hand: never in routine CI.
The ``dev/reference/README.md`` file explains the environments and each
maintenance workflow: the non-destructive sanity check (``pixi run validate``), a
deliberate refresh (``pixi run update``), adding an instrument or model version,
and adopting a new Mantid release.
