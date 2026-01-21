ResINS |version|
****************

Python library for working with resolution functions of inelastic neutron
scattering (INS) instruments. This package exists to centralise all things related to resolution of
INS instruments and make it easier to work with. It pools related code from existing projects,
namely `AbINS <https://github.com/mantidproject/mantid/tree/main/scripts/abins>`_ and
`PyChop <https://github.com/mducle/pychop/tree/main>`_, as well as implementing code
published in literature. The main purposes are:

1. Provide one, central place implementing various models for INS instruments (resolution functions)
2. Provide a simple way to obtain the broadening at a given frequency, for a given instrument and settings
3. (in progress) Provide a way to apply broadening to a spectrum


Using ResINS
============

.. toctree::
    :maxdepth: 2

    installation
    quickstart
    glossary
    theory
    howto
    tutorials
    api
    instruments
    dev
