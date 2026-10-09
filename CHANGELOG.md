## [Unreleased](https://github.com/pace-neutrons/resins/compare/0.1.0...HEAD)

### Changed

- The comparison tests now read reference values committed in
  `tests/reference_data/` and no longer need conda, Mantid or PyChop; they run
  with `pip install . --group test` and `pytest`. tox, tox-conda and the git
  dependency on mducle/pychop are removed, and CI runs on a Python version
  matrix without conda.
- Reference data is generated reproducibly with locked pixi environments in
  `dev/reference/` (see its README).

### Fixed

- Test defects: the TOSCA comparison never asserted, its forward and backward
  fixtures were identical, the meV/cm⁻¹ constants differed from Abins, and the
  PyChop chopper-width tolerance (1e-8 s², against values of order 1e-10 s²)
  could not detect any difference.

## [0.1.0](https://github.com/pace-neutrons/resins/releases/tag/0.1.0)

Initial PyPI release. API should be considered unstable between minor
versions until 1.0.

Resolution functions from AbINS (including Pychop) are implemented and
validated. Currently broaden() method operates on "sparse" points; on
dense pre-binned data this is not computationally optimal but gives an
"exact" reference for approximate optimised schemes.
