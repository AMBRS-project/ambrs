"""Aggregation must not relabel or discard a scenario's species and modes."""

from dataclasses import replace

import numpy as np
import pytest

from ambrs import aerosol, ppe
from ambrs.gas import GasSpecies
from ambrs.scenario import Scenario


def _scenario():
    sulfate = aerosol.AerosolSpecies(name="SO4", molar_mass=96.0, density=1800.0)
    carbon = aerosol.AerosolSpecies(name="BC", molar_mass=12.0, density=1000.0)
    modes = tuple(
        aerosol.AerosolModeState(
            name=name, species=(sulfate, carbon), number=number,
            geom_mean_diam=1e-7, log10_geom_std_dev=0.2,
            mass_fractions=(0.75, 0.25),
        )
        for name, number in (("accumulation", 1e8), ("coarse", 2e8))
    )
    return Scenario(
        aerosols=(sulfate, carbon),
        gases=(GasSpecies(name="SO2", molar_mass=64.07),
               GasSpecies(name="H2SO4", molar_mass=98.079)),
        size=aerosol.AerosolModalSizeState(modes=modes),
        gas_concs=(1e-9, 2e-9), flux=0.0, relative_humidity=0.5,
        temperature=298.0, pressure=101325.0, height=500.0,
    )


def _with_first_mode(scenario, mode):
    return replace(scenario, size=replace(scenario.size, modes=(mode, scenario.size.modes[1])))


@pytest.mark.parametrize("change, message", [
    (lambda s: replace(s, gases=s.gases[::-1], gas_concs=s.gas_concs[::-1]), "gases"),
    (lambda s: replace(s, gases=(replace(s.gases[0], molar_mass=65.0), s.gases[1])), "gases"),
    (lambda s: replace(s, aerosols=s.aerosols[::-1]), "aerosols"),
    (lambda s: replace(s, size=replace(s.size, modes=s.size.modes + (s.size.modes[0],))), "mode count"),
    (lambda s: replace(s, size=replace(s.size, modes=s.size.modes[::-1])), "mode 1.*name"),
    (lambda s: _with_first_mode(s, replace(s.size.modes[0],
         species=s.size.modes[0].species[::-1], mass_fractions=(0.25, 0.75))), "mode 1.*species"),
    (lambda s: _with_first_mode(s, replace(s.size.modes[0], species=(
         replace(s.aerosols[0], density=1900.0), s.aerosols[1]))), "mode 1.*species"),
    (lambda s: replace(s, gas_concs=s.gas_concs + (3e-9,)), "gas_concs"),
    (lambda s: _with_first_mode(s, replace(s.size.modes[0], mass_fractions=(1.0,))), "mass_fractions"),
])
def test_rejects_metadata_that_cannot_be_stored_per_member(change, message):
    first = _scenario()
    second = change(_scenario())
    with pytest.raises(ValueError, match=message):
        ppe.ensemble_from_scenarios([first, second])


def test_rejects_a_different_size_representation_in_later_scenarios():
    with pytest.raises(TypeError, match="Scenario 2.*size"):
        ppe.ensemble_from_scenarios([_scenario(), replace(_scenario(), size=object())])


def test_accepts_equal_metadata_and_preserves_member_values():
    first = _scenario()
    second = replace(_scenario(), gas_concs=(3e-9, 4e-9), temperature=285.0)
    ensemble = ppe.ensemble_from_scenarios([first, second])
    result = ensemble.member(1)
    assert result.gases == second.gases
    assert result.aerosols == second.aerosols
    np.testing.assert_allclose(result.gas_concs, second.gas_concs, rtol=1e-12, atol=0)
    assert result.temperature == 285.0
    assert result.size == second.size


@pytest.mark.parametrize("change, message", [
    (lambda s: replace(s, gas_concs=s.gas_concs + (3e-9,)), "gas_concs"),
    (lambda s: replace(s, gas_concs=s.gas_concs[:1]), "gas_concs"),
    (lambda s: _with_first_mode(s, replace(s.size.modes[0],
         mass_fractions=(0.5, 0.25, 0.25))), "mass_fractions"),
    (lambda s: _with_first_mode(s, replace(s.size.modes[0],
         mass_fractions=(1.0,))), "mass_fractions"),
])
@pytest.mark.parametrize("members", [1, 2])
def test_reference_and_equal_members_must_match_their_species_columns(change, message, members):
    # Cross-member equality does not establish a valid reference schema.
    scenarios = [change(_scenario()) for _ in range(members)]
    with pytest.raises(ValueError, match=message):
        ppe.ensemble_from_scenarios(scenarios)


def test_reference_without_gas_species_accepts_empty_gas_concentrations():
    scenario = replace(_scenario(), gases=(), gas_concs=())
    result = ppe.ensemble_from_scenarios([scenario]).member(0)
    assert result.gases == () and result.gas_concs == ()
