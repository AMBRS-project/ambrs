"""Regression test for build_gas_mixture species/value alignment."""

import numpy as np

import ambrs.gas as gas


def test_build_gas_mixture_includes_soag_species_metadata():
    mixture = gas.build_gas_mixture(
        {"units": "ppb", "SO2": 1.0, "H2SO4": 2.0, "SOAG": 3.0}
    )

    assert tuple(spec.name for spec in mixture.species) == ("SO2", "H2SO4", "SOAG")
    assert len(mixture.species) == len(mixture.mole_ratio)
    np.testing.assert_allclose(mixture.mole_ratio, [1.0e-9, 2.0e-9, 3.0e-9])
