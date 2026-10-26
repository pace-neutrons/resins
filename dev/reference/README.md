# Reference data for the resins comparison tests

## 1. What this is

resins resolution models are validated against other implementations: Mantid's
bundled `pychop` (PyChop), and Abins (PyChop-based, PANTHER, TOSCA and Lagrange
resolution). Running those implementations needs [Mantid](https://mantidproject.org), which is available
available through conda with strict dependency pins.
The reference values are generated on a supported platform using [pixi](https://pixi.prefix.dev)
environments and committed to `tests/reference_data/`.
This doesn't require resins to be imported, so avoids e.g. Numpy version conflicts.

The resins validation test suite (`tests/comparison_tests/`) reads
those files and never needs to directly import Mantid, Abins or
PyChop.

This directory holds everything needed to regenerate the files:

| Path | Content |
|---|---|
| `pixi.toml`, `pixi.lock` | Locked Mantid environments and maintenance "tasks" (`pixi run ...`) |
| `config.yaml` | Which (instrument, model version) pairs are compared, in which environment, and the sampling parameters |
| `scripts/` | Generators and tools. They never import resins; configurations and limits are read from `src/resins/instrument_data/*.yaml` as plain data |
| `build/` | Scratch output (git-ignored): `build/reference_data/` |

Reference data is not regenerated in routine CI; the workflows are run as needed on a developer workstation or by actively dispatching the `generate_reference_data` GitHub workflow.

### Which Mantid version produces which file

Multiple Mantid versions are required to reproduce different model
versions:  each is pinned by a different pixi environment.
`pixi.lock` fixes the exact builds, and each reference file records
the Mantid version that produced it in its metadata (see `pixi run
summary`).

| Environment | Python | Mantid | Reference files |
|---|---|---|---|
| `mantid611` | 3.10 | 6.11 | `pychop/ARCS_PyChop_fit_v1`, `pychop/SEQUOIA_PyChop_fit_v1` |
| `mantid615` | 3.11 | 6.15 | `pychop/ARCS_PyChop_fit_v2`, `pychop/SEQUOIA_PyChop_fit_v2`, `pychop/MAPS_PyChop_fit_v1`, `pychop/MARI_PyChop_fit_v1`, `pychop/MERLIN_PyChop_fit_v1`, `pychop/HYSPEC_PyChop_fit_v1`, `pychop/CNCS_PyChop_fit_v1`, `pychop/LET_PyChop_fit_v1`, `abins/pychop`, `abins/panther`, `abins/tosca`, `abins/lagrange` |

### What a reference file contains

Each `.npz` file (loaded with `allow_pickle=False`) holds:

- `cases`: a structured table of model inputs and their expected outcome (`value`, `no_transmission`, or rejected input);
- `resolution` (PyChop files) or `sigma` (Abins files): 2D float64 reference values in meV aligned with `cases` (`NaN` where no transmission exists);
- `metadata`: JSON provenance recording the Mantid version, generator settings, and unit conventions.

Cases are declared in `cases.yaml` to sample regular frequency/energy grids, zero-transmission slit edges, and parameter limits.

## 2. Setup

Install [pixi](https://pixi.sh) (version 0.50 or later), then, from this
directory:

```bash
cd dev/reference
pixi install --all
```

This creates both Mantid environments under `.pixi/` (a few GB; the first
install downloads the Mantid conda packages). Check that they work:

```bash
pixi run -e mantid611 python -c "import mantid, pychop; print(mantid.__version__)"   # 6.11.0
pixi run -e mantid615 python -c "import mantid, pychop; print(mantid.__version__)"   # 6.15.0
```

Mantid 6.11 is not available for ARM macOS (`osx-arm64`), so the two files
requiring it (ARCS and SEQUOIA `PyChop_fit_v1`) cannot be regenerated there.

List the available commands with `pixi task list`, or `pixi task list -e mantid615`
for the environment-specific ones.

## 3. Routine sanity check

```bash
pixi run validate
```

This does not modify any committed file. It runs, in order:

1. `generate`: regenerates all data into `build/reference_data/` (about two
   minutes);
2. `compare`: compares `build/reference_data/` with `tests/reference_data/`.

`compare` prints, for every file, the cases added or removed, the outcome flips,
and the maximum relative and absolute deviation of every value array, and ends
with `PASS` or `FAIL`. It fails on any of: a case added or removed, an outcome
flip, a deviation above 1e-7 relative, a file added or missing, or a total size
above the 5 MB budget.

An *outcome flip* means that a case changed between a value, no transmission and
a rejected input. It is not numerical noise: near a transmission edge it means
PyChop's edge moved. Edge points are placed 0.1% either side of the edge, far
more than the cross-version noise (below 1e-8), so a flip points to a real change in
PyChop or to a change of the resins instrument YAML that the generator reads.

The same command runs in the manual `generate_reference_data` GitHub workflow,
which uploads `build/` as an artifact.

## 4. Changing sampling or instrument data

Edit the sampling parameters in `config.yaml`, or the resins instrument YAML
(`src/resins/instrument_data/*.yaml`), then:

```bash
pixi run update
```

`update` runs `generate`, then `install` (replaces `tests/reference_data/` with the
regenerated files, removing stale ones), then `summary` (case counts per file and
outcome, sizes, and the Mantid version of each file).

Then:

1. check `git diff --stat tests/reference_data` and the summary;
2. run the resins test suite from the repository root: `pytest -n auto`;
3. commit the data together with the YAML or `config.yaml` change.

## 5. Adding an instrument, model version or configuration

**A new configuration (e.g. a chopper package) of an instrument that is
already compared**: add it to the resins YAML, make sure PyChop knows it (or give
the PyChop name in `configuration_map` of the instrument's `config.yaml` entry),
and run `pixi run update`.

**A new PyChop instrument or model version**: add an entry to `pychop:` in
`config.yaml`:

```yaml
  - instrument: NEWINSTRUMENT         # resins instrument name
    model_version: PyChop_fit_v1      # resins model name
    environment: mantid615            # which Mantid environment produces it
    pychop_name: NEWINSTRUMENT        # name in Mantid's bundled pychop
    frequency_settings: [chopper_frequency]   # resins setting names, in PyChop order
    configuration_map: {}             # resins -> PyChop names, where they differ
```

Choose the environment by checking which Mantid version holds the PyChop data
that the resins model was built from. A model version that needs a Mantid
version no environment has needs a new environment first (section 6). Then run
`pixi run update`.

**A new Abins comparison**: extend `scripts/generate_abins.py` and
`tests/comparison_tests/test_abins_models.py`; the `abins:` section of
`config.yaml` holds its parameters.

## 6. Adopting a new Mantid release

Environments are pinned to explicit Mantid minor versions; there is no floating
"latest" one. To see what a new release changes before touching committed data
(for example Mantid 6.16):

1. Add an environment to `pixi.toml`, next to the existing ones:

   ```toml
   [feature.mantid616.dependencies]
   python = "3.12.*"
   mantid = "6.16.*"

   [environments]
   mantid616 = { features = ["mantid616"], no-default-feature = false }
   ```

   and `pixi install -e mantid616`. The `mantid` conda package fixes the Python
   and numpy versions; check them with `pixi search mantid`, and set `python` to
   match. If entries of `config.yaml` will use the new environment, it also needs
   a `generate-mantid616` task (in `[feature.mantid616.tasks]`, like
   `generate-mantid615`) and a line for it in the `generate` task's `depends-on`.

2. To see what the new release changes, temporarily point the entries of
   interest in `config.yaml` at the new environment, then run `pixi run generate`
   and `pixi run compare`. The report names every case that differs.

3. Decide for each differing instrument:

   - *the older resins model is simply out of date*: repoint its entry in
     `config.yaml` to the new environment and regenerate. The resins YAML
     parameters must then be updated to match, which is a separate change to the
     library;
   - *both the old and the new behaviour deserve a model version*: add a new resins
     model version to the YAML (like `PyChop_fit_v2` of ARCS), keep the old entry
     on its environment, and add a new entry for the new version (section 5).

4. Run `pixi run update` and follow section 4.

5. When no entry refers to the old environment any more, remove it from
   `pixi.toml` and run `pixi lock`.

## 7. Known discrepancies

Disagreements between resins and the reference that are understood but not yet
resolved are listed in `KNOWN_DISCREPANCIES` in
`tests/comparison_tests/reference.py`. The affected cases are strict expected
failures: if one starts passing, the suite fails until the entry is removed.

| Reference | Affected cases | Reason | Follow-up |
|---|---|---|---|
| `pychop/ARCS_PyChop_fit_v2` | `ARCS-700-1.5-SMI`, value cases, and 12 no-transmission cases just beyond Mantid's transmission edge | Mantid has `pslit: 1.` mm, resins 1.5 mm | Raise with Mantid/SNS; update the resins YAML or the reference, then drop the entry |
| `abins/lagrange` | Cu(220), 25 meV <= E < 34.5 meV | `AbINS_v1` has a cutoff where Abins has a 0.8 meV floor | Removed by change `resins-lagrange-abins-v2` |

If regenerating with different sampling moves the 12 no-transmission points,
re-derive them: run `pytest --runxfail -n auto
tests/comparison_tests/test_pychop_resolution.py -k "ARCS-700-1.5-SMI"` and list
the failing `test_no_transmission` cases.

Two other differences are deliberate, not discrepancies:

- PyChop's `setEi` limits (`ei_limits`) are not applied. `getResolution`, which is
  what resins is compared with, does not use them, and they are narrower than the
  resins ranges for several instruments (for example CNCS, up to 30 meV against
  80 meV in resins). `generate` prints the number of such cases per file. The
  incident energy limits of each case come from the resins restrictions.
- Abins supports only some choppers (no MAPS `B`, no MARI `B` or `C`), so these are
  not in `abins/pychop`.

## 8. Troubleshooting

- *`pixi install` fails to solve `mantid611` on ARM macOS*: Mantid 6.11 was not
  published for `osx-arm64`. On Apple Silicon, regenerate those files using
  the manual GitHub workflow or on Linux (or run
  `pixi run -e mantid615 generate-mantid615` for the Mantid 6.15 models).
- *Solver or lock errors for old Mantid versions*: do not run `pixi update` or edit
  `pixi.lock` by hand. `pixi.lock` pins exact builds; use `pixi install --locked` to
  fail instead of re-solving. If the `mantid` channel drops a package, the committed
  data still works for testing; only regeneration is blocked.
- *Mantid 6.11 gives a wrong resolution when several PyChop instruments are
  alive*: up to Mantid 6.11, `pychop.Instrument` reads the chopper frequency of the
  most recently created instance. The generators work around this
  (`PyChopEvaluator` in `scripts/generate_pychop.py`); do not keep several
  instruments and use them in turns in new scripts.
- *`compare` reports added or removed cases without any change to the config*: the
  case table depends on the resins instrument YAML. Check `git diff
  src/resins/instrument_data`.
