"""The tuple fix must not alter exact-name lookup semantics."""
import numpy as np

from ambrs.gas import GasMixture, GasSpecies


def test_alias_does_not_replace_existing_species():
    original = GasSpecies(name="SO2", molar_mass=64.07, aliases=("H2SO4",))
    added = GasSpecies(name="H2SO4", molar_mass=98.079)
    mixture = GasMixture(species=(original,), mole_ratio=np.array([2e-9]))
    mixture._add_gas(added, 3e-9)
    assert mixture.species == (original, added)
    np.testing.assert_array_equal(mixture.mole_ratio, [2e-9, 3e-9])


def test_exact_name_wins_over_earlier_alias():
    alias = GasSpecies(name="SO2", molar_mass=64.07, aliases=("H2SO4",))
    exact = GasSpecies(name="H2SO4", molar_mass=98.079)
    mixture = GasMixture(species=(alias, exact), mole_ratio=np.array([2e-9, 3e-9]))
    mixture._add_gas(exact, 5e-9)
    assert mixture.species == (alias, exact)
    np.testing.assert_array_equal(mixture.mole_ratio, [2e-9, 5e-9])


def test_case_differences_are_not_matches():
    upper = GasSpecies(name="SO2", molar_mass=64.07)
    lower = GasSpecies(name="so2", molar_mass=64.07)
    mixture = GasMixture(species=(upper,), mole_ratio=np.array([2e-9]))
    mixture._add_gas(lower, 3e-9)
    assert mixture.species == (upper, lower)
    np.testing.assert_array_equal(mixture.mole_ratio, [2e-9, 3e-9])


def test_existing_duplicate_names_keep_original_update_behavior():
    so2 = GasSpecies(name="SO2", molar_mass=64.07)
    mixture = GasMixture(species=(so2, so2), mole_ratio=np.array([2e-9, 3e-9]))
    mixture._add_gas(so2, 5e-9)
    assert mixture.species == (so2, so2)
    np.testing.assert_array_equal(mixture.mole_ratio, [5e-9, 5e-9])


def test_empty_mixture_can_receive_first_species():
    so2 = GasSpecies(name="SO2", molar_mass=64.07)
    mixture = GasMixture(species=(), mole_ratio=np.array([]))
    mixture._add_gas(so2, 5e-9)
    assert mixture.species == (so2,)
    np.testing.assert_array_equal(mixture.mole_ratio, [5e-9])
