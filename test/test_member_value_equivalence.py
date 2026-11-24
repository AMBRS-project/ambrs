"""Member extraction retains the values of the former broadcast implementation."""
from dataclasses import replace

import numpy as np
import pytest

from ambrs import aerosol, gas, ppe


def ensemble():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    bc = aerosol.AerosolSpecies(name="BC", molar_mass=12.0, density=1000.0)
    mode = aerosol.AerosolModePopulation(
        name="accumulation", species=(so4, bc),
        number=np.array([2e8, 3e8]), geom_mean_diam=1.5e-7,
        log10_geom_std_dev=np.array([0.2, 0.35]),
        mass_fractions=(0.25, np.array([0.75, 0.75])),
    )
    return ppe.Ensemble(
        aerosols=(so4, bc),
        gases=(gas.GasSpecies(name="SO2", molar_mass=64.07),),
        size=aerosol.AerosolModalSizePopulation(modes=(mode,)),
        gas_concs=(np.array([2e-9, 3e-9]),), flux=2.5,
        relative_humidity=np.array([0.4, 0.6]), temperature=295.0,
        pressure=np.array([100000.0, 101325.0]), height=500.0,
    )


def test_repeated_extraction_matches_previous_broadcast_values():
    source = ensemble()
    mode = source.size.modes[0]
    sources = [(source, name) for name in (
        "flux", "relative_humidity", "temperature", "pressure", "gas_concs",
    )] + [(mode, name) for name in (
        "number", "geom_mean_diam", "log10_geom_std_dev", "mass_fractions",
    )]
    originals = [(owner, name, getattr(owner, name)) for owner, name in sources]
    snapshots = [(owner, name, np.asarray(value).copy())
                 for owner, name, value in originals if not isinstance(value, tuple)]
    for i in (1, 0, 1):
        member = source.member(i)
        for owner, name, value in originals:
            extracted = member if owner is source else member.size.modes[0]
            if isinstance(value, tuple):
                expected = tuple((v * np.ones(2))[i] for v in value)
            else:
                expected = (value * np.ones(2))[i]
            np.testing.assert_array_equal(getattr(extracted, name), expected)
            assert getattr(owner, name) is value
        for owner, name, snapshot in snapshots:
            np.testing.assert_array_equal(getattr(owner, name), snapshot)
    np.testing.assert_array_equal(mode.mass_fractions[1], [0.75, 0.75])
    np.testing.assert_array_equal(source.gas_concs[0], [2e-9, 3e-9])


@pytest.mark.parametrize("value", [np.array([0.2]), np.array([[0.2, 0.3]])])
def test_invalid_mode_field_shape_is_rejected(value):
    mode = replace(ensemble().size.modes[0], geom_mean_diam=value)
    with pytest.raises(ValueError, match="geom_mean_diam must be scalar"):
        mode.member(0)


@pytest.mark.parametrize("value", [np.array([295.0]), np.array([[295.0, 296.0]])])
def test_invalid_ensemble_field_shape_is_rejected(value):
    source = replace(ensemble(), temperature=value)
    with pytest.raises(ValueError, match="temperature must be scalar"):
        source.member(0)
