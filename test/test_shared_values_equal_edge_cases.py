"""Exercise shared forcing equality for supported and incomparable values."""
from dataclasses import dataclass

import numpy as np
import pytest

from ambrs.ppe import _shared_values_equal


@pytest.mark.parametrize("left,right,expected", [
    (np.array(["SO4"]), np.array(["SO4"]), True),
    (np.array(["SO4"]), np.array(["BC"]), False),
    (np.array([1.0, np.nan]), np.array([1.0, np.nan]), True),
    (np.array([1.0, np.nan]), np.array([1.0, 2.0]), False),
    ({"rate": 1.0}, {"time": 1.0}, False),
    ([1.0], [1.0, 2.0], False),
])
def test_shared_values_equality(left, right, expected):
    assert _shared_values_equal(left, right) is expected


def test_distinct_dataclass_types_are_not_equal():
    @dataclass
    class Emission:
        rate: float

    @dataclass
    class Background:
        rate: float

    assert not _shared_values_equal(Emission(1.0), Background(1.0))


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_incomparable_shared_objects_are_not_equal(error):
    class Incomparable:
        def __eq__(self, other):
            raise error("undefined equality")

    assert not _shared_values_equal(Incomparable(), Incomparable())
