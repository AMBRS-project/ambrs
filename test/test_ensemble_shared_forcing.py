"""Scenario aggregation cannot silently replace per-member forcing."""
from dataclasses import replace
import pytest
from ambrs import aerosol, ppe
from ambrs.scenario import Scenario


def _scenario(**overrides):
    bc = aerosol.AerosolSpecies(name="BC", molar_mass=12.0, density=1000.0)
    state = aerosol.AerosolModalSizeState(modes=(aerosol.AerosolModeState(
        name="mode", species=(bc,), number=1e6, geom_mean_diam=1e-7,
        log10_geom_std_dev=0.2, mass_fractions=(1.0,),
    ),))
    base = Scenario(aerosols=(bc,), gases=(), size=state, gas_concs=(),
                    flux=0.0, relative_humidity=0.4, temperature=298.0,
                    pressure=101325.0, height=500.0)
    return replace(base, **overrides)


@pytest.mark.parametrize("field,first,second", [
    ("height", 500.0, 600.0),
    ("gas_emissions", [(0.0, {"rate": 1.0})], [(0.0, {"rate": 2.0})]),
    ("gas_background", [(0.0, {"rate": 1.0})], [(0.0, {"rate": 2.0})]),
    ("aerosol_emissions", ["A"], ["B"]),
    ("aerosol_background", ["A"], ["B"]),
])
def test_rejects_mismatched_shared_values(field, first, second):
    with pytest.raises(ValueError, match=field):
        ppe.ensemble_from_scenarios([_scenario(**{field: first}), _scenario(**{field: second})])


def test_accepts_matching_forcing():
    scenes = [_scenario(gas_background=[(0.0, {"rate": 1.0})]) for _ in range(2)]
    ensemble = ppe.ensemble_from_scenarios(scenes)
    assert len(ensemble) == 2
    assert ensemble.gas_background == scenes[0].gas_background
