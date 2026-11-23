"""Common scaling must not change a finite-data normalized error metric."""

from types import SimpleNamespace

import numpy as np
import pytest

from ambrs.analysis import nmae


def output(values):
    return SimpleNamespace(compute_variable=lambda *_: np.asarray(values, dtype=float))


@pytest.mark.parametrize("scale", [1.0, 1.0e-308, 1.0e308])
def test_reference_sum_cannot_overflow_to_a_false_zero_error(scale):
    with np.errstate(all="raise"):
        error = nmae([output([scale, scale])], [output([scale, 0.0])], "x")
    assert error == pytest.approx(0.5)


def test_opposite_signed_finite_values_do_not_overflow_during_subtraction():
    largest = np.finfo(float).max
    with np.errstate(all="raise"):
        error = nmae([output([largest, -largest])],
                     [output([-largest, largest])], "x")
    assert error == pytest.approx(2.0)


def test_identical_extreme_values_still_have_zero_error():
    with np.errstate(all="raise"):
        error = nmae([output([1.0e308, 1.0e308])],
                     [output([1.0e308, 1.0e308])], "x")
    assert error == 0.0


def test_scaling_keeps_reference_weighting_across_outputs():
    assert nmae([output([1.0]), output([9.0])],
                [output([0.0]), output([9.0])], "x") == pytest.approx(0.1)


def test_unrepresentably_large_error_returns_infinity():
    with np.errstate(all="raise"):
        assert np.isinf(nmae([output([1.0e-308])], [output([1.0e308])], "x"))


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 0.0]])
def test_zero_reference_keeps_undefined_error_behavior(values):
    assert np.isnan(nmae([output(values)], [output(values)], "x"))


def test_nonfinite_data_keeps_its_previous_behavior():
    with np.errstate(all="ignore"):
        assert np.isnan(nmae([output([np.nan])], [output([1.0])], "x"))
        assert np.isnan(nmae([output([np.inf])], [output([1.0])], "x"))
        assert np.isinf(nmae([output([1.0])], [output([np.inf])], "x"))


@pytest.mark.parametrize("value", [1.0, 2.0, 1.0e100])
def test_a_larger_element_does_not_erase_a_small_representable_difference(value):
    reference = np.array([value, 1.0e308])
    candidate = np.array([np.nextafter(value, np.inf), 1.0e308])
    expected = (candidate[0] - reference[0]) / 1.0e308
    with np.errstate(all="raise"):
        result = nmae([output(reference)], [output(candidate)], "x")
    assert result == expected


@pytest.mark.parametrize("dtype, value", [
    (np.uint64, 2**60),
    (np.uint64, 2**64 - 2),
    (np.int64, 2**60),
    (np.int64, -(2**63)),
])
def test_integer_differences_are_preserved_beyond_float64_precision(dtype, value):
    reference = np.array([value], dtype=dtype)
    candidate = np.array([value + 1], dtype=dtype)
    first = SimpleNamespace(compute_variable=lambda *_: reference)
    second = SimpleNamespace(compute_variable=lambda *_: candidate)
    assert nmae([first], [second], "x") == 1 / abs(value)


def test_unsigned_subtraction_cannot_wrap():
    first = SimpleNamespace(compute_variable=lambda *_: np.array([3], dtype=np.uint64))
    second = SimpleNamespace(compute_variable=lambda *_: np.array([0], dtype=np.uint64))
    assert nmae([first], [second], "x") == 1.0


def test_zero_integer_reference_keeps_undefined_behavior():
    first = SimpleNamespace(compute_variable=lambda *_: np.array([0], dtype=np.int64))
    second = SimpleNamespace(compute_variable=lambda *_: np.array([1], dtype=np.int64))
    assert np.isnan(nmae([first], [second], "x"))


def test_complex_component_magnitudes_need_not_be_representable():
    largest = np.finfo(float).max
    reference = np.array([complex(largest, largest)])
    candidate = -reference
    first = SimpleNamespace(compute_variable=lambda *_: reference)
    second = SimpleNamespace(compute_variable=lambda *_: candidate)
    with np.errstate(all="raise"):
        assert nmae([first], [second], "x") == pytest.approx(2.0)


def test_small_differences_accumulate_before_final_underflow_rounding():
    reference = np.array([1.0] * 20 + [1.0e308])
    candidate = np.array([np.nextafter(1.0, np.inf)] * 20 + [1.0e308])
    expected = ((candidate[0] - reference[0]) * 20) / 1.0e308
    with np.errstate(all="raise"):
        result = nmae([output(reference)], [output(candidate)], "x")
    assert result == expected
