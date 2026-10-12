## [Unreleased](https://github.com/pace-neutrons/resins/compare/0.1.0...HEAD)

### Removed

- **Breaking:** removed the `INSTRUMENT_DATA_PATH` variable from `resins.instrument`. Instrument
  data files are now located with `importlib.resources`.
  ([#65](https://github.com/pace-neutrons/resins/pull/65))

### Documentation

- Added a developer docs page on testing and on regenerating the reference data.
- Fixed ambiguous cross-references to `data_class` and `polynomial` in the PyChop API docs.
  (`PyChopModelCNCS` and `PyChopModelLET` now declare `data_class` themselves;
  `PyChopModelNonFermi` no longer lists inherited members.)

### Maintenance

- Comparison tests
  - Reworked the tests to read reference values committed in `tests/reference_data/`, instead of
    running Mantid, Abins and PyChop. (They now run with plain `pytest`, without conda.)
  - Generated the reference data with locked pixi environments in `dev/reference/`. (Manual only,
    never part of routine CI; see its README.)
  - Made the Abins TOSCA test assert agreement with Abins. (It could not fail before.)
  - Replaced the identical TOSCA "forward" and "backward" fixtures with one resolution function.
  - Converted energies with Abins' `MILLI_EV_TO_WAVENUMBER` in both directions, not a slightly
    different inverse constant.
- Developer tools
  - Removed obsolete developer scratch scripts (`dev/design.md`, `dev/developer_utils.py`,
    `dev/plotting/plot_pychop.py`, `dev/convert_units.py` and `dev/test_convert_units.py`).
- Unit testing
  - Patched `importlib.resources.files()` in the `Instrument` tests, in place of the removed
    `INSTRUMENT_DATA_PATH`. ([#65](https://github.com/pace-neutrons/resins/pull/65))
  - Set pytest `testpaths` to `tests/`, so `pytest` skips the developer scripts in `dev/`.
- Dependencies
  - Moved the documentation requirements from `docs/requirements.txt` to a `doc` dependency group
    in `pyproject.toml`. (Possible now that `sphinx_parsed_codeblock` is on PyPI.)
    ([#66](https://github.com/pace-neutrons/resins/pull/66))
  - Pinned `pygments < 2.21` in the `doc` group. (`sphinx_parsed_codeblock` imports `escape_html`,
    which Pygments 2.21 removed.)
  - Added a `test` dependency group, installed with `pip install --group test`.
  - Removed `tox`, `tox-conda` and the git dependency on mducle/pychop.
  - Removed `Pint` from the `dev` extra now that `dev/convert_units.py` is removed.
- GitHub Actions
  - Replaced the Miniconda and `tox` test workflow with `setup-python` on Python 3.11 and 3.14
    (Ubuntu) and 3.14 (Windows).
  - Added a manually triggered `generate_reference_data.yml` workflow.
  - Updated GitHub Actions to current releases (`actions/checkout@v7`, `actions/setup-python@v7`,
    `prefix-dev/setup-pixi@v0.11.0`, `actions/upload-artifact@v7`).

## [0.1.0](https://github.com/pace-neutrons/resins/releases/tag/0.1.0) - 2026-01-21

Initial PyPI release. API should be considered unstable between minor
versions until 1.0.

Resolution functions from AbINS (including Pychop) are implemented and
validated. Currently broaden() method operates on "sparse" points; on
dense pre-binned data this is not computationally optimal but gives an
"exact" reference for approximate optimised schemes.
