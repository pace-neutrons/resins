## [Unreleased](https://github.com/pace-neutrons/resins/compare/0.1.0...HEAD)

### Removed

- **Breaking:** removed the `INSTRUMENT_DATA_PATH` variable from `resins.instrument`. Instrument
  data files are now located with `importlib.resources`.
  ([#65](https://github.com/pace-neutrons/resins/pull/65))

### Documentation

- Added a developer page on testing and on regenerating the reference data.
- Fixed ambiguous cross-references to `data_class` and `polynomial` in the PyChop API
  pages, which failed the docs build when warnings are errors. `PyChopModelCNCS` and
  `PyChopModelLET` now declare `data_class` themselves, and `PyChopModelNonFermi` no
  longer lists inherited members automatically.

### Maintenance

- Documentation requirements
  - Moved the documentation requirements from `docs/requirements.txt` to a
    `doc` dependency group in `pyproject.toml`, installed with `pip install
    --group doc`. This is possible now that `sphinx_parsed_codeblock` is on
    PyPI. ([#66](https://github.com/pace-neutrons/resins/pull/66))
  - Pinned `pygments < 2.21` in the `doc` group, because `sphinx_parsed_codeblock`
    imports `escape_html`, which Pygments 2.21 removed.
- Unit testing
  - Patched `importlib.resources.files()` in the `Instrument` unit tests, in
    place of the removed `INSTRUMENT_DATA_PATH` variable.
    ([#65](https://github.com/pace-neutrons/resins/pull/65))
  - Reworked the comparison tests to read reference values committed in
    `tests/reference_data/`, instead of running Mantid, Abins and PyChop. They
    now run with `pip install . --group test` and `pytest`, without conda.
  - Generated the reference data with locked pixi environments in
    `dev/reference/` (see its README). Regeneration is manual, never part of
    routine CI.
  - Removed `tox`, `tox-conda` and the git dependency on mducle/pychop.
  - Set pytest `testpaths` to `tests/`, so `pytest` skips the developer
    scripts in `dev/`.
  - Made the Abins TOSCA test _assert_ agreement with Abins; it previously
    could not fail.
  - Replaced the TOSCA "forward" and "backward" fixtures, which were
    identical, with a single resolution function.
  - Converted energies with Abins' `MILLI_EV_TO_WAVENUMBER` in both
    directions, instead of a slightly different inverse constant.
  - Compared PyChop chopper widths with a relative tolerance of 1e-7; the
    previous absolute 1e-8 s² exceeded the values themselves (order 1e-10 s²).
- GitHub Actions
  - Replaced the Miniconda and `tox` test workflow with `setup-python` on
    Python 3.11 and 3.14 (Ubuntu) and 3.14 (Windows).
  - Added a manually triggered `generate_reference_data.yml` workflow.

## [0.1.0](https://github.com/pace-neutrons/resins/releases/tag/0.1.0) - 2026-01-21

Initial PyPI release. API should be considered unstable between minor
versions until 1.0.

Resolution functions from AbINS (including Pychop) are implemented and
validated. Currently broaden() method operates on "sparse" points; on
dense pre-binned data this is not computationally optimal but gives an
"exact" reference for approximate optimised schemes.
