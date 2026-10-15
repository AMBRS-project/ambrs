"""Semantically equal NumPy-valued forcing is accepted across members."""
from dataclasses import replace
import numpy as np
import pytest

from ambrs import aerosol, ppe
from ambrs.emissions import AerosolEmissions
from ambrs.scenario import Scenario


def _scenario(number):
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    mode = aerosol.AerosolModeState(name="accumulation", species=(so4,),
        number=np.array(number), geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(np.array(1.0),))
    state = aerosol.AerosolModalSizeState(modes=(mode,))
    return Scenario(aerosols=(so4,), gases=(), size=state, gas_concs=(),
        flux=0.0, relative_humidity=0.5, temperature=298.0,
        pressure=101325.0, height=500.0,
        aerosol_emissions=[AerosolEmissions(time=0.0, rate=0.1, size=state)])


def test_equivalent_distinct_numpy_backed_forcing_events_are_accepted():
    a, b = _scenario(1e8), _scenario(1e8)
    ensemble = ppe.ensemble_from_scenarios([a, b])
    assert len(ensemble) == 2


def test_different_numpy_backed_forcing_events_are_rejected():
    a, b = _scenario(1e8), _scenario(2e8)
    with pytest.raises(ValueError, match="aerosol_emissions"):
        ppe.ensemble_from_scenarios([a, b])
