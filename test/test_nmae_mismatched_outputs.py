"""NMAE rejects incomplete or incompatible model comparisons."""
from types import SimpleNamespace
import numpy as np
import pytest
from ambrs.analysis import nmae


def _output(values):
    return SimpleNamespace(compute_variable=lambda *_: np.asarray(values))


def test_nmae_rejects_different_numbers_of_scenarios():
    with pytest.raises(ValueError, match="same length"):
        nmae([_output([1.0]), _output([2.0])], [_output([1.0])], "Nccn")


def test_nmae_rejects_mismatched_variable_shapes():
    with pytest.raises(ValueError, match="matching shapes"):
        nmae([_output([1.0, 2.0])], [_output([1.0])], "Nccn")


def test_nmae_handles_matching_arrays():
    assert nmae([_output([2.0, 2.0])], [_output([1.0, 1.0])], "Nccn") == 0.5
