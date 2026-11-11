Lagrange
########

LAGRANGE is an indirect-geometry spectrometer at the Institut Laue Langevin. [#]_

.. rubric:: Models

``AbINS_v2`` is the default Lagrange model (also selected by the unversioned name ``AbINS``). It
matches the Abins implementation in Mantid 6.15 and later. For each monochromator the width is
a polynomial of the energy transfer, with a minimum width for Cu(220) (``floored_polynomial``).

.. deprecated:: Unreleased

   ``AbINS_v1`` is deprecated in favour of ``AbINS_v2``. It never matched any released
   Mantid/Abins implementation exactly. It applies the intended 25 meV low-energy cutoff for
   Cu(220), which differs from both Abins ≤6.14 (the cutoff was compared with energies in cm\ :sup:`-1`,
   so the flat segment switched at about 3.1 meV) and Abins ≥6.15 (the cutoff was replaced by a
   floor). It is kept unchanged for reproducibility, and also raises an error for Cu(220) below
   about 1.6 meV, where its polynomial is negative.

.. list-table:: History of the Cu(220) low-energy width
   :header-rows: 1

   * - Implementation
     - Behaviour
   * - Abins ≤6.14
     - Flat 0.8 meV (2σ) segment up to about 3.1 meV (cutoff of 25 meV compared in cm\ :sup:`-1`)
   * - resins ``AbINS_v1``
     - Flat 0.8 meV (2σ) below 25 meV, the polynomial above it (step at 25 meV)
   * - Abins ≥6.15, resins ``AbINS_v2``
     - ``max(polynomial, 0.8 meV)`` (2σ), continuous, valid from 0 meV

.. rubric:: References

.. [#] https://www.ill.eu/users/instruments/instruments-list/in1-lagrange/description/instrument-layout

.. include:: ../../include/auto-instruments/lagrange-auto.rst
