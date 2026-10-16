"""LHS keeps real shared aerosol events on sampled members."""
import scipy.stats
from ambrs import aerosol, ppe
from ambrs.emissions import AerosolEmissions


def test_lhs_preserves_real_aerosol_events():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    state = aerosol.AerosolModalSizeState(modes=(aerosol.AerosolModeState(
        name="accumulation", species=(so4,), number=1e8, geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(1.0,)),))
    event = AerosolEmissions(time=0.0, rate=0.01, size=state)
    spec = ppe.EnsembleSpecification(name="test", aerosols=(so4,), gases=(),
        size=aerosol.AerosolModalSizeDistribution(modes=(aerosol.AerosolModeDistribution(
            name="accumulation", species=(so4,),
            number=scipy.stats.uniform(1e8, 1e7),
            geom_mean_diam=scipy.stats.uniform(1e-7, 1e-8),
            log10_geom_std_dev=scipy.stats.uniform(0.2, 0.01),
            mass_fractions=(scipy.stats.uniform(0.9, 0.01),)),)),
        gas_concs=(), flux=0.0, relative_humidity=0.5, temperature=298.0,
        pressure=101325.0, height=500.0,
        aerosol_emissions=[event], aerosol_background=[event])
    ensemble = ppe.lhs(spec, 2, seed=11)
    assert len(ensemble) == 2
    assert ensemble.member(1).aerosol_emissions[0] is event
    assert ensemble.member(0).aerosol_background[0] is event
