"""Unit tests for scripts/sampling.py using synthetic transmission predicates."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sampling  # noqa: E402

DELTA = 1e-3
RANGE = (1.0, 1000.0)


def edges(predicate, ei_range=RANGE):
    return sampling.find_transmission_edges(predicate, ei_range, n_scan=200, rtol=1e-7)


def test_no_edge():
    assert edges(lambda e: True) == []
    assert sampling.edge_points(lambda e: False, RANGE) == []


def test_single_low_edge():
    (edge,) = edges(lambda e: e > 37.3)
    assert edge == pytest.approx(37.3, rel=1e-6)


def test_single_high_edge():
    (edge,) = edges(lambda e: e < 455.0)
    assert edge == pytest.approx(455.0, rel=1e-6)


def test_two_edges():
    low, high = edges(lambda e: 20.0 < e < 300.0)
    assert low == pytest.approx(20.0, rel=1e-6)
    assert high == pytest.approx(300.0, rel=1e-6)


@pytest.mark.parametrize(
    "predicate, expected_edges",
    [
        (lambda e: e > 37.3, [37.3]),
        (lambda e: e < 455.0, [455.0]),
        (lambda e: 20.0 < e < 300.0, [20.0, 300.0]),
    ],
)
def test_edge_points_straddle_each_edge(predicate, expected_edges):
    points = sampling.edge_points(predicate, RANGE, delta=DELTA)
    assert len(points) == 2 * len(expected_edges)
    for edge, (below, above) in zip(expected_edges, zip(points[::2], points[1::2])):
        assert below == pytest.approx(edge * (1 - DELTA), rel=1e-5)
        assert above == pytest.approx(edge * (1 + DELTA), rel=1e-5)
        assert predicate(below) != predicate(above)


def test_edge_points_outside_range_are_dropped():
    # Edge within delta of the lower limit: the outer point falls outside the range.
    points = sampling.edge_points(lambda e: e > 1.0005, RANGE, delta=DELTA)
    assert all(RANGE[0] <= p <= RANGE[1] for p in points)


@pytest.mark.parametrize("lists", [[[60, 120, 180, 240, 300]], [[60, 120, 180], [10, 20, 30, 40, 50]]])
def test_covering_design_includes_every_value(lists):
    combinations = sampling.frequency_combinations(lists)
    for k, values in enumerate(lists):
        assert {c[k] for c in combinations} == {float(v) for v in values}
    if len(lists) > 1:
        assert len(combinations) < np.prod([len(v) for v in lists])
    else:
        assert len(combinations) == len(lists[0])


def test_covering_design_let_size():
    values = list(range(10, 301, 10))
    assert len(sampling.frequency_combinations([values, values])) == 30


def test_ei_limit_points_sides():
    restrictions = {"e_init": [20, 1500]}
    points = sampling.ei_limit_points(restrictions["e_init"], DELTA)
    inside = [p for p in points if 20 <= p <= 1500]
    outside = [p for p in points if not 20 <= p <= 1500]
    assert max(p for p in outside if p < 20) < 20 <= min(inside)
    assert max(inside) <= 1500 < max(outside)
    assert len(outside) == 2


def test_ei_limit_points_zero_lower_limit():
    points = sampling.ei_limit_points([0, 2000], DELTA)
    assert any(p < 0 for p in points)
    assert 0 not in points  # zero energy is undefined for PyChop
    assert 2000 in points and any(p > 2000 for p in points)


def test_frequency_limit_points_are_outside():
    restriction = [50, 601, 50]
    for frequency in sampling.frequency_limit_points(restriction, DELTA):
        assert frequency not in range(*restriction)
    assert max(sampling.frequency_limit_points(restriction, DELTA)) > 600


def test_check_restrictions():
    restrictions = {"e_init": [20, 1500], "f": [60, 601, 60]}
    check = lambda e, f: sampling.check_restrictions(restrictions, e, ["f"], [f])  # noqa: E731
    assert check(100, 120) is None
    assert check(19.9, 120) == ("rejected_ei", "e_init")
    assert check(1500.1, 120) == ("rejected_ei", "e_init")
    assert check(100, 600.6) == ("rejected_frequency", "f")
    assert check(100, 90) == ("rejected_frequency", "f")
    # Ei is checked first
    assert check(10, 90) == ("rejected_ei", "e_init")


def synthetic_cases(transmits):
    restrictions = {"e_init": [2, 500], "f": [60, 361, 60]}
    defaults = {"e_init": 100, "f": 180}
    return sampling.sample_configuration(
        "cfg", restrictions, defaults, ["f"], transmits, sampling.SamplingParameters()
    )


def test_sample_configuration_regular_and_limits():
    cases = synthetic_cases(lambda e, f: True)
    frequencies = {c.frequencies[0] for c in cases if c.origin == "regular"}
    assert frequencies == {60.0, 120.0, 180.0, 240.0, 300.0, 360.0}
    assert len({(c.e_init, c.frequencies) for c in cases}) == len(cases)
    assert not [c for c in cases if c.origin == "edge"]
    restrictions = {"e_init": [2, 500], "f": [60, 361, 60]}
    rejected = [
        c
        for c in cases
        if sampling.check_restrictions(restrictions, c.e_init, ["f"], c.frequencies)
    ]
    assert {r for c in rejected if (r := sampling.check_restrictions(restrictions, c.e_init, ["f"], c.frequencies))} == {
        ("rejected_ei", "e_init"),
        ("rejected_frequency", "f"),
    }


def test_sample_configuration_edges_depend_on_frequency():
    # Transmission needs e above 10 * f/60 (a low edge that rises with frequency).
    cases = synthetic_cases(lambda e, f: e > 10.0 * f[0] / 60)
    edge_cases = [c for c in cases if c.origin == "edge"]
    assert len(edge_cases) >= 8
    for case in edge_cases:
        edge = 10.0 * case.frequencies[0] / 60
        assert abs(case.e_init / edge - 1) == pytest.approx(1e-3, rel=0.1)


def test_resins_yaml_is_readable():
    model = sampling.load_model("MARI", "PyChop_fit_v1")
    options = sampling.configuration_options(model)
    assert set(options) == {"A", "B", "C", "G", "R", "S"}
    assert sampling.effective_restrictions(model, options["G"])["e_init"] == [0, 181]
    assert sampling.effective_restrictions(model, options["A"])["e_init"] == [0, 2000]


def test_nominal_energy_is_inside_the_range():
    # Default Ei (400) is outside this configuration's range [0, 181], as for MERLIN G.
    restrictions = {"e_init": [0, 181], "f": [50, 601, 50]}
    cases = sampling.sample_configuration(
        "G", restrictions, {"e_init": 400, "f": 400}, ["f"], None, sampling.SamplingParameters()
    )
    limit_frequency_cases = [
        c for c in cases if c.origin == "limit" and c.frequencies[0] not in range(50, 601, 50)
    ]
    assert len(limit_frequency_cases) == 2
    for case in limit_frequency_cases:
        assert 0 <= case.e_init <= 181
