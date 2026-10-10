"""Regression tests for GasMixture._add_gas."""

import numpy as np

from ambrs.gas import GasMixture, GasSpecies


def test_add_gas_replaces_existing_species_without_changing_shape():
    so2 = GasSpecies(name="SO2", molar_mass=64.07)
    mixture = GasMixture(species=(so2,), mole_ratio=np.array([1.0e-9]))

    mixture._add_gas(GasSpecies(name="SO2", molar_mass=64.07), 2.0e-9)

    assert len(mixture.species) == 1
    np.testing.assert_allclose(mixture.mole_ratio, [2.0e-9])


def test_add_gas_appends_new_species_and_keeps_species_and_values_aligned():
    so2 = GasSpecies(name="SO2", molar_mass=64.07)
    h2so4 = GasSpecies(name="H2SO4", molar_mass=98.079)
    mixture = GasMixture(species=(so2,), mole_ratio=np.array([1.0e-9]))

    mixture._add_gas(h2so4, 3.0e-12)

    assert tuple(spec.name for spec in mixture.species) == ("SO2", "H2SO4")
    np.testing.assert_allclose(mixture.mole_ratio, [1.0e-9, 3.0e-12])
