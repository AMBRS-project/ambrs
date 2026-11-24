"""Regression tests for aerosol forcing metadata in LHS ensembles."""

import scipy.stats

import ambrs.aerosol as aerosol
import ambrs.ppe as ppe


def test_lhs_preserves_aerosol_emissions_and_background_on_members():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=97.071, density=1770)
    emissions = ["emission-sentinel"]
    background = ["background-sentinel"]

    dist = scipy.stats.uniform(0.2, 0.1)
    specification = ppe.EnsembleSpecification(
        name="forcing-metadata",
        aerosols=(so4,),
        gases=(),
        size=aerosol.AerosolModalSizeDistribution(
            modes=(
                aerosol.AerosolModeDistribution(
                    name="aitken",
                    species=(so4,),
                    number=scipy.stats.uniform(1.0e8, 1.0e7),
                    geom_mean_diam=scipy.stats.uniform(1.0e-7, 1.0e-8),
                    log10_geom_std_dev=dist,
                    mass_fractions=(scipy.stats.uniform(0.9, 0.1),),
                ),
            )
        ),
        gas_concs=(),
        flux=scipy.stats.uniform(0.0, 1.0),
        relative_humidity=scipy.stats.uniform(0.3, 0.2),
        temperature=scipy.stats.uniform(280.0, 10.0),
        pressure=scipy.stats.uniform(90000.0, 10000.0),
        height=500.0,
        aerosol_emissions=emissions,
        aerosol_background=background,
    )

    ensemble = ppe.lhs(specification, 3, seed=7)

    assert ensemble.aerosol_emissions is emissions
    assert ensemble.aerosol_background is background
    for member in ensemble:
        assert member.aerosol_emissions is emissions
        assert member.aerosol_background is background
