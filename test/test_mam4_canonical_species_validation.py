"""Canonical MAM4 names must not collide or silently accept unknown names."""
from types import SimpleNamespace
import pytest

from ambrs.aerosol import AerosolModeState, AerosolSpecies
from ambrs.mam4 import AerosolMassFractions


def _mode(species):
    return AerosolModeState(name="test-mode", species=tuple(species), number=1.0,
                           geom_mean_diam=1e-7, log10_geom_std_dev=0.2,
                           mass_fractions=tuple(1.0 / len(species) for _ in species))


def test_duplicate_aliases_cannot_map_to_the_same_mam4_field():
    pom = AerosolSpecies(name="POM", molar_mass=12.0, density=1000.0)
    oc = AerosolSpecies(name="OC", aliases=("POM",), molar_mass=12.0, density=1000.0)
    mode = _mode((pom, oc))
    scenario = SimpleNamespace(size=SimpleNamespace(modes=(mode,) * 4))
    with pytest.raises(ValueError, match="Duplicate canonical MAM4 species"):
        AerosolMassFractions(scenario)


def test_unsupported_species_are_rejected_before_dynamic_dataclass():
    nitrate = AerosolSpecies(name="NO3", molar_mass=62.0, density=1800.0)
    mode = _mode((nitrate,))
    scenario = SimpleNamespace(size=SimpleNamespace(modes=(mode,) * 4))
    with pytest.raises(ValueError, match="Unsupported MAM4 aerosol species"):
        AerosolMassFractions(scenario)
