"""MAM4 supplies zero for absent optional species in each aerosol mode."""
from ambrs import aerosol, mam4
from ambrs.gas import GasSpecies
from ambrs.scenario import Scenario


def test_missing_modal_species_are_zero():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    pom = aerosol.AerosolSpecies(name="POM", molar_mass=12.0, density=1000.0)
    modes = tuple(aerosol.AerosolModeState(
        name=name, species=(sp,), number=1e8, geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(1.0,),
    ) for name, sp in zip(("accumulation", "aitken", "coarse", "primary carbon"),
                          (so4, so4, so4, pom)))
    gases = (GasSpecies(name="SO2", molar_mass=64.07),
             GasSpecies(name="H2SO4", molar_mass=98.079))
    scenario = Scenario(aerosols=(so4, pom), gases=gases,
        size=aerosol.AerosolModalSizeState(modes=modes),
        gas_concs=(1e-9, 1e-9), flux=0.0, relative_humidity=0.5,
        temperature=298.0, pressure=101325.0, height=500.0)
    values = mam4.AerosolModel(aerosol.AerosolProcesses()).create_input(scenario, 60.0, 2)
    assert values.mfso41 == 1.0
    assert values.mfpom1 == values.mfbc1 == values.mfsoa1 == 0.0
    assert values.mfso42 == 1.0
    assert values.mfsoa2 == values.mfncl2 == 0.0
    assert values.mfpom4 == 1.0
    assert values.mfbc4 == 0.0
