"""Regression tests for non-mutating ensemble member extraction."""

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe


def _size_population():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=97.071, density=1770)
    return aerosol.AerosolModalSizePopulation(
        modes=(
            aerosol.AerosolModePopulation(
                name="aitken",
                species=(so4,),
                number=np.array([1.0e8, 2.0e8]),
                geom_mean_diam=1.0e-7,
                log10_geom_std_dev=np.array([0.2, 0.3]),
                mass_fractions=(1.0,),
            ),
        )
    )


def test_mode_member_does_not_broadcast_scalars_into_the_population():
    population = _size_population()

    member = population.member(1)

    assert population.modes[0].geom_mean_diam == 1.0e-7
    assert population.modes[0].mass_fractions == (1.0,)
    assert member.geom_mean_diam == 1.0e-7
    assert member.mass_fractions == (1.0,)


def test_ensemble_member_does_not_mutate_scalar_fields_or_gas_concentrations():
    so2 = gas.GasSpecies(name="SO2", molar_mass=64.07)
    ensemble = ppe.Ensemble(
        aerosols=_size_population().modes[0].species,
        gases=(so2,),
        size=_size_population(),
        gas_concs=(7.0,),
        flux=0.0,
        relative_humidity=np.array([0.4, 0.5]),
        temperature=298.0,
        pressure=101325.0,
        height=500.0,
    )

    member = ensemble.member(1)

    assert ensemble.gas_concs == (7.0,)
    assert ensemble.flux == 0.0
    assert ensemble.temperature == 298.0
    assert ensemble.pressure == 101325.0
    assert member.gas_concs == (7.0,)
    assert member.relative_humidity == 0.5
    assert member.temperature == 298.0
