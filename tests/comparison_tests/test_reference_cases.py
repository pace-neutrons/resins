"""Checks of the case parametrization helpers in ``reference.py``."""

from __future__ import annotations

import pytest

from .reference import (
    EVALUATED,
    KNOWN_DISCREPANCIES,
    NO_TRANSMISSION,
    REJECTED_EI,
    REJECTED_FREQUENCY,
    VALUE,
    case_id,
    case_params,
    known_discrepancy_marks,
    load,
    pychop_reference_files,
    stems,
)

ALL_STEMS = pychop_reference_files() + stems("abins")


def test_reference_files_are_present():
    assert len(pychop_reference_files()) == 10
    assert stems("abins") == [
        "abins/lagrange",
        "abins/panther",
        "abins/pychop",
        "abins/tosca",
    ]


@pytest.mark.parametrize("stem", ALL_STEMS)
def test_case_ids_are_unique(stem):
    ref = load(stem)
    ids = [case_id(ref, case_index) for case_index, _ in enumerate(ref.cases)]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("stem", ALL_STEMS)
def test_cases_are_unique(stem):
    ref = load(stem)
    names = [name for name in ref.cases.dtype.names if name not in ("outcome", "reason")]
    keys = [tuple(case[name].item() for name in names) for case in ref.cases]
    assert len(set(keys)) == len(keys)


def test_case_id_is_readable():
    ref = load("pychop/ARCS_PyChop_fit_v2")
    case_index = next(
        idx
        for idx, case in enumerate(ref.cases)
        if case["configuration"] == "ARCS-100-1.5-AST" and case["e_init"] == 200.0
    )
    assert case_id(ref, case_index).startswith("ARCS-PyChop_fit_v2-ARCS-100-1.5-AST-ei=200-f=")


@pytest.mark.parametrize("discrepancy", KNOWN_DISCREPANCIES, ids=lambda d: d.file)
def test_marks_attach_only_to_matching_cases(discrepancy):
    ref = load(discrepancy.file)
    (scope,) = discrepancy.scopes
    marked = {
        case_index
        for case_index, _ in enumerate(ref.cases)
        if known_discrepancy_marks(ref, case_index, scope)
    }
    expected = {
        case_index
        for case_index, _ in enumerate(ref.cases)
        if discrepancy.applies(ref, case_index)
    }
    assert marked == expected
    assert marked  # the discrepancy affects at least one case

    # Never attached to other files
    for stem in ALL_STEMS:
        if stem != discrepancy.file:
            other = load(stem)
            assert not any(
                known_discrepancy_marks(other, case_index, scope)
                for case_index, _ in enumerate(other.cases)
            )


def test_arcs_discrepancy_covers_only_that_configuration():
    ref = load("pychop/ARCS_PyChop_fit_v2")
    for case_index, case in enumerate(ref.cases):
        marked = bool(known_discrepancy_marks(ref, case_index, "resolution"))
        if marked:
            assert case["configuration"] == "ARCS-700-1.5-SMI"
            assert case["outcome"] in EVALUATED
        if case["configuration"] == "ARCS-700-1.5-SMI" and case["outcome"] == VALUE:
            assert marked


def test_marks_depend_on_scope():
    ref = load("pychop/ARCS_PyChop_fit_v2")
    case_index = next(
        idx
        for idx, case in enumerate(ref.cases)
        if case["configuration"] == "ARCS-700-1.5-SMI" and case["outcome"] == VALUE
    )
    assert known_discrepancy_marks(ref, case_index, "resolution")
    assert not known_discrepancy_marks(ref, case_index, "abins")


def test_lagrange_discrepancy_window():
    ref = load("abins/lagrange")
    for case_index, case in enumerate(ref.cases):
        marked = bool(known_discrepancy_marks(ref, case_index, "abins"))
        expected = (
            case["configuration"] == "Cu(220)" and 25 <= case["energy_transfer"] < 34.5
        )
        assert marked == expected


def test_case_params_filter_by_outcome():
    stem = "pychop/MARI_PyChop_fit_v1"
    ref = load(stem)
    for outcome in (VALUE, NO_TRANSMISSION, REJECTED_EI, REJECTED_FREQUENCY):
        params = case_params(
            reference_files=stem, scope="resolution", outcomes=(outcome,)
        )
        assert len(params) == int((ref.cases["outcome"] == outcome).sum())
