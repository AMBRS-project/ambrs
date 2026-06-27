import numpy as np
import pytest

import ambrs.gas as gas


def test_gas_species_validates_positive_molar_mass():
    species = gas.GasSpecies(name="SO2", molar_mass=64.07, aliases=("sulfur_dioxide",))

    assert species.name == "SO2"
    assert species.molar_mass == 64.07
    assert species.aliases == ("sulfur_dioxide",)


def test_gas_species_rejects_non_positive_molar_mass():
    with pytest.raises(ValueError, match="Non-positive molar mass"):
        gas.GasSpecies(name="SO2", molar_mass=0.0)


def test_find_matches_name_alias_and_missing_species():
    species = [
        gas.GasSpecies(name="SO2", molar_mass=64.07, aliases=("sulfur_dioxide",)),
        gas.GasSpecies(name="H2SO4", molar_mass=98.079),
    ]

    assert gas.GasSpecies.find(species, "SO2") == 0
    assert gas.GasSpecies.find(species, "sulfur_dioxide") == 0
    assert gas.GasSpecies.find(species, "O3") == -1


def test_build_gas_mixture_converts_ppb_to_mole_ratio():
    mixture = gas.build_gas_mixture(
        {"units": "ppb", "SO2": 1000.0, "H2SO4": 2.0}
    )

    assert [species.name for species in mixture.species] == ["SO2", "H2SO4"]
    np.testing.assert_allclose(mixture.mole_ratio, [1.0e-6, 2.0e-9])


@pytest.mark.parametrize("unit", ["ratio", "mole_ratio", "mol_ratio"])
def test_build_gas_mixture_accepts_mole_ratio_aliases(unit):
    mixture = gas.build_gas_mixture(
        {"units": unit, "SO2": 1.0e-9, "H2SO4": 2.0e-12}
    )

    np.testing.assert_allclose(mixture.mole_ratio, [1.0e-9, 2.0e-12, 0.0])


def test_build_gas_mixture_converts_kg_per_kg_to_mole_ratio():
    mixture = gas.build_gas_mixture(
        {"units": "kg_per_kg", "SO2": 1.0e-9, "H2SO4": 2.0e-9, "SOAG": 3.0e-9}
    )

    expected = [
        1.0e-9 * 28.97 / 64.07,
        2.0e-9 * 28.97 / 98.079,
        3.0e-9 * 28.97 / 250.0,
    ]
    np.testing.assert_allclose(mixture.mole_ratio, expected)


def test_build_gas_mixture_rejects_unsupported_units():
    with pytest.raises(ValueError, match="Unsupported gas units"):
        gas.build_gas_mixture({"units": "ppm"})
