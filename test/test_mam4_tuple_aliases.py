"""MAM4 uses string aliases when constructing aerosol fraction fields."""
from types import SimpleNamespace
from ambrs.aerosol import AerosolModeState, AerosolSpecies
from ambrs.mam4 import AerosolMassFractions


def test_mam4_mass_fraction_alias_tuple_creates_string_field():
    so4 = AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0,
                         aliases=("sulfate", "sulphate"))
    organic = AerosolSpecies(name="OC", molar_mass=12.0, density=1000.0,
                             aliases=("POM", "organic"))
    mode = AerosolModeState(name="mode", species=(so4, organic), number=1.0,
                            geom_mean_diam=1e-7, log10_geom_std_dev=0.2,
                            mass_fractions=(0.6, 0.4))
    scenario = SimpleNamespace(size=SimpleNamespace(modes=(mode, mode, mode, mode)))
    fractions = AerosolMassFractions(scenario)
    for m in (fractions.accum, fractions.aitken, fractions.coarse, fractions.pcarbon):
        assert m.SO4 == 0.6
        assert m.POM == 0.4
