"""MAM4 gas input and output use the same dry-air molar mass."""

import numpy as np

from ambrs.aerosol import AerosolModalSizeState
from ambrs.gas import GasSpecies, build_gas_mixture
from ambrs.mam4 import GasMixingRatios
from ambrs.scenario import Scenario


def test_mam4_gas_mixing_ratios_round_trip_through_mass_basis():
    scenario = Scenario(
        aerosols=(),
        gases=(GasSpecies("SO2", 64.07), GasSpecies("H2SO4", 98.079)),
        size=AerosolModalSizeState(()),
        gas_concs=(2.0e-9, 3.0e-9),
        flux=0.0,
        relative_humidity=0.5,
        temperature=298.0,
        pressure=101325.0,
        height=100.0,
    )

    mass_ratios = GasMixingRatios(scenario)
    np.testing.assert_allclose(
        [mass_ratios.SO2, mass_ratios.H2SO4],
        [2.0e-9 * 64.07 / 28.97, 3.0e-9 * 98.079 / 28.97],
        rtol=1e-12, atol=0.0,
    )

    restored = build_gas_mixture({
        "units": "kg_per_kg",
        "SO2": mass_ratios.SO2,
        "H2SO4": mass_ratios.H2SO4,
    })
    indices = [GasSpecies.find(restored.species, species.name) for species in scenario.gases]
    assert -1 not in indices
    np.testing.assert_allclose(
        restored.mole_ratio[indices], scenario.gas_concs, rtol=1e-12, atol=0.0,
    )
