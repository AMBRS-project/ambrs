"""Preserve the original NMAE calculation on ordinary model outputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from ambrs.analysis import nmae


def output(values):
    return SimpleNamespace(compute_variable=lambda *_: np.asarray(values, dtype=float))


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_reference_weighting_is_unchanged(scale):
    assert nmae([output([scale, scale])], [output([scale, 0.0])], "x") == 0.5


def test_weighting_across_outputs():
    assert nmae([output([1.0]), output([9.0])],
                [output([0.0]), output([9.0])], "x") == pytest.approx(0.1)


def test_matching_multidimensional_arrays():
    reference = output([[1.0, 2.0], [3.0, 4.0]])
    candidate = output([[0.0, 2.0], [3.0, 3.0]])
    assert nmae([reference], [candidate], "x") == pytest.approx(0.2)


def test_equal_size_different_shape_is_rejected():
    with pytest.raises(ValueError, match="matching shapes"):
        nmae([output([[1.0, 2.0]])], [output([1.0, 2.0])], "x")


def test_empty_output_lists():
    assert np.isnan(nmae([], [], "x"))


@pytest.mark.parametrize("values", [[], [0.0], [0.0, 0.0]])
def test_zero_reference_keeps_undefined_error_behavior(values):
    assert np.isnan(nmae([output(values)], [output(values)], "x"))


def test_nonfinite_data_keeps_its_previous_behavior():
    with np.errstate(all="ignore"):
        assert np.isnan(nmae([output([np.nan])], [output([1.0])], "x"))
        assert np.isnan(nmae([output([np.inf])], [output([1.0])], "x"))
        assert np.isinf(nmae([output([1.0])], [output([np.inf])], "x"))
