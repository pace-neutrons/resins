"""Compare resins resolution models with committed Abins reference values.

Abins values were generated with the Abins package bundled with Mantid (see ``dev/reference``)
and stored in meV, converted with Abins' own constants in both directions. Every comparison
asserts agreement.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from resins.instrument import Instrument
from resins.models.pychop import NoTransmissionError

from .reference import (
    EVALUATED,
    NO_TRANSMISSION,
    VALUE,
    case_params,
    resolve,
)


def sigma_at(model, energies: np.ndarray) -> np.ndarray:
    return model.get_characteristics(energies[:, None])["sigma"]


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files="abins/pychop",
        scope="abins",
        outcomes=EVALUATED,
    ),
)
def test_pychop_against_abins(param):
    ref, i = resolve(param)
    case = ref.cases[i]
    instrument = Instrument.from_default(str(case["instrument"]))
    model_version = ref.metadata["model_versions"][str(case["instrument"])]
    kwargs = {
        "chopper_package": str(case["configuration"]),
        "e_init": float(case["e_init"]),
        "chopper_frequency": float(case["f0"]),
    }

    if case["outcome"] == NO_TRANSMISSION:
        # Abins' PyChop reports no transmission here; resins must agree.
        with pytest.raises(NoTransmissionError):
            instrument.get_resolution_function(model_version, **kwargs)
        return

    model = instrument.get_resolution_function(model_version, **kwargs)
    energies = np.linspace(0, case["e_init"], ref.metadata["n_points"])

    assert_allclose(sigma_at(model, energies), ref["sigma"][i], rtol=1e-5)


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files="abins/panther",
        scope="abins",
        outcomes=(VALUE,),
    ),
)
def test_panther_against_abins(param):
    ref, i = resolve(param)
    e_init = float(ref.cases[i]["e_init"])
    model = Instrument.from_default("PANTHER").get_resolution_function(
        "AbINS", e_init=e_init
    )
    energies = np.linspace(0, e_init, ref.metadata["n_points"])

    assert_allclose(sigma_at(model, energies), ref["sigma"][i])


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files="abins/tosca",
        scope="abins",
        outcomes=(VALUE,),
    ),
)
def test_tosca_against_abins(param):
    ref, i = resolve(param)
    model = Instrument.from_default("TOSCA").get_resolution_function("AbINS")
    energy = np.array([ref.cases[i]["energy_transfer"]])

    assert_allclose(sigma_at(model, energy), ref["sigma"][i])


@pytest.mark.parametrize(
    "param",
    case_params(
        reference_files="abins/lagrange",
        scope="abins",
        outcomes=(VALUE,),
    ),
)
def test_lagrange_against_abins(param):
    ref, i = resolve(param)
    case = ref.cases[i]
    model = Instrument.from_default("Lagrange").get_resolution_function(
        "AbINS", monochromator=str(case["configuration"])
    )
    energy = np.array([case["energy_transfer"]])

    assert_allclose(sigma_at(model, energy), ref["sigma"][i], rtol=1e-5)
