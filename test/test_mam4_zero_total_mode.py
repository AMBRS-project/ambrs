"""MAM4 rejects mass-fraction totals that cannot be normalized."""
import pytest

from ambrs import aerosol, mam4
from ambrs.gas import GasSpecies
from ambrs.scenario import Scenario


@pytest.mark.parametrize("invalid_mode", [1, 2, 3, 4])
def test_zero_total_modal_mass_fractions_raise_meaningful_error(invalid_mode):
    fractions = [1.0] * 4
    fractions[invalid_mode - 1] = 0.0
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    pom = aerosol.AerosolSpecies(name="POM", molar_mass=12.0, density=1000.0)
    modes = tuple(aerosol.AerosolModeState(
        name=name, species=(species,), number=1e8, geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(fraction,),
    ) for name, species, fraction in zip(
        ("accumulation", "aitken", "coarse", "primary carbon"),
        (so4, so4, so4, pom), fractions))
    scenario = Scenario(aerosols=(so4, pom), gases=(
        GasSpecies(name="SO2", molar_mass=64.07),
        GasSpecies(name="H2SO4", molar_mass=98.079),
    ), size=aerosol.AerosolModalSizeState(modes=modes), gas_concs=(1e-9, 1e-9),
        flux=0.0, relative_humidity=0.5, temperature=298.0,
        pressure=101325.0, height=500.0)
    with pytest.raises(ValueError, match=rf"mode {invalid_mode}.*positive, finite"):
        mam4.AerosolModel(aerosol.AerosolProcesses()).create_input(scenario, 60.0, 2)
