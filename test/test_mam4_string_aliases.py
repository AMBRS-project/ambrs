"""Regression for bare-string MAM4 aliases used by the repository demo."""

from types import SimpleNamespace

import pytest

from ambrs.aerosol import AerosolModeState, AerosolSpecies
from ambrs.mam4 import AerosolMassFractions, _mam4_species_name


@pytest.mark.parametrize(
    "original, alias",
    [("OC", "POM"), ("MSA", "SOA"), ("OIN", "DST"), ("Na", "NCL")],
)
def test_demo_single_string_alias_resolves_to_one_name(original, alias):
    species = AerosolSpecies(
        original, molar_mass=12.0, density=1000.0, aliases=alias
    )
    assert _mam4_species_name(species) == alias


def test_primary_name_takes_precedence_over_bare_string_alias():
    species = AerosolSpecies(
        "SO4", molar_mass=96.0, density=1800.0, aliases="POM"
    )
    assert _mam4_species_name(species) == "SO4"


def test_unsupported_bare_string_alias_still_raises():
    species = AerosolSpecies(
        "OC", molar_mass=12.0, density=1000.0, aliases="UNKNOWN"
    )
    with pytest.raises(ValueError, match="Unsupported MAM4 aerosol species"):
        _mam4_species_name(species)


def test_demo_aliases_are_usable_in_all_four_mass_fraction_modes():
    species = {
        name: AerosolSpecies(name, molar_mass=12.0, density=1000.0, aliases=alias)
        for name, alias in (
            ("OC", "POM"), ("MSA", "SOA"), ("OIN", "DST"), ("Na", "NCL")
        )
    }
    for name in ("SO4", "BC"):
        species[name] = AerosolSpecies(name, molar_mass=12.0, density=1000.0)

    def mode(name, members):
        entries = tuple(species[member] for member in members)
        fractions = tuple((i + 1) / 100.0 for i in range(len(entries)))
        return AerosolModeState(
            name=name,
            species=entries,
            number=1.0,
            geom_mean_diam=1e-7,
            log10_geom_std_dev=0.2,
            mass_fractions=fractions,
        )

    modes = (
        mode("accumulation", ("SO4", "OC", "MSA", "BC", "OIN", "Na")),
        mode("aitken", ("SO4", "MSA", "Na")),
        mode("coarse", ("OIN", "Na", "SO4", "BC", "OC", "MSA")),
        mode("primary carbon", ("OC", "BC")),
    )
    fractions = AerosolMassFractions(SimpleNamespace(size=SimpleNamespace(modes=modes)))
    assert fractions.accum.POM == pytest.approx(0.02)
    assert fractions.accum.SOA == pytest.approx(0.03)
    assert fractions.accum.DST == pytest.approx(0.05)
    assert fractions.accum.NCL == pytest.approx(0.06)
    assert fractions.aitken.SOA == pytest.approx(0.02)
    assert fractions.aitken.NCL == pytest.approx(0.03)
    assert fractions.coarse.DST == pytest.approx(0.01)
    assert fractions.coarse.NCL == pytest.approx(0.02)
    assert fractions.coarse.POM == pytest.approx(0.05)
    assert fractions.coarse.SOA == pytest.approx(0.06)
    assert fractions.pcarbon.POM == pytest.approx(0.01)
