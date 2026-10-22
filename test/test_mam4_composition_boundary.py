"""Native modal fields must faithfully represent the supplied composition."""

from dataclasses import replace

import numpy as np
import pytest

from ambrs import aerosol, mam4
from ambrs.gas import GasSpecies
from ambrs.scenario import Scenario


def make_scenario():
    sulfate = aerosol.AerosolSpecies("SO4", 96.0, 1800.0)
    organic = aerosol.AerosolSpecies("POM", 12.0, 1000.0)
    modes = tuple(
        aerosol.AerosolModeState(
            name=name,
            species=(species,),
            number=1.0e8,
            geom_mean_diam=1.0e-7,
            log10_geom_std_dev=0.2,
            mass_fractions=(1.0,),
        )
        for name, species in zip(
            ("accumulation", "aitken", "coarse", "primary carbon"),
            (sulfate, sulfate, sulfate, organic),
        )
    )
    return Scenario(
        aerosols=(sulfate, organic),
        gases=(GasSpecies("SO2", 64.07), GasSpecies("H2SO4", 98.079)),
        size=aerosol.AerosolModalSizeState(modes),
        gas_concs=(1e-9, 1e-9),
        flux=0.0,
        relative_humidity=0.5,
        temperature=298.0,
        pressure=101325.0,
        height=500.0,
    )


def change_mode(scenario, index, **changes):
    modes = list(scenario.size.modes)
    modes[index] = replace(modes[index], **changes)
    return replace(scenario, size=replace(scenario.size, modes=tuple(modes)))


@pytest.mark.parametrize("index, name", [(1, "BC"), (1, "POM"), (3, "SO4")])
@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_rejects_species_without_a_native_field(index, name, fraction):
    scenario = make_scenario()
    extra = aerosol.AerosolSpecies(name, 12.0, 1000.0)
    original = scenario.size.modes[index]
    scenario = change_mode(
        scenario,
        index,
        species=original.species + (extra,),
        mass_fractions=(1.0 - fraction, fraction),
    )
    model = mam4.AerosolModel(aerosol.AerosolProcesses())
    with pytest.raises(ValueError, match=rf"mode {index + 1}.*{name}"):
        model.create_input(scenario, 60.0, 2)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf, -np.inf])
def test_rejects_invalid_components_even_with_a_positive_total(index, invalid):
    scenario = make_scenario()
    original = scenario.size.modes[index]
    extra_name = "SOA" if index in (1, 2) else "BC"
    extra = aerosol.AerosolSpecies(extra_name, 12.0, 1000.0)
    scenario = change_mode(
        scenario,
        index,
        species=original.species + (extra,),
        mass_fractions=(invalid, 2.0),
    )
    with pytest.raises(ValueError, match=rf"mode {index + 1}.*nonnegative, finite"):
        mam4.AerosolModel(aerosol.AerosolProcesses()).create_input(scenario, 60.0, 2)


def test_sparse_valid_composition_is_written_without_redistribution(tmp_path):
    scenario = make_scenario()
    black_carbon = aerosol.AerosolSpecies("BC", 12.0, 1000.0)
    scenario = change_mode(
        scenario,
        0,
        species=(scenario.aerosols[0], black_carbon),
        mass_fractions=(2.0, 3.0),
    )
    model = mam4.AerosolModel(aerosol.AerosolProcesses())
    inputs = model.create_input(scenario, 60.0, 2)
    model.write_input_files(inputs, tmp_path, "case")
    namelist = tmp_path / "namelist"
    assert mam4.get_mam_input("mfso41", namelist) == pytest.approx(0.4)
    assert mam4.get_mam_input("mfbc1", namelist) == pytest.approx(0.6)
    assert mam4.get_mam_input("mfso42", namelist) == 1.0
    assert mam4.get_mam_input("mfpom4", namelist) == 1.0
    assert scenario.size.modes[0].mass_fractions == (2.0, 3.0)


def test_alias_is_checked_using_its_resolved_native_field():
    scenario = make_scenario()
    organic = aerosol.AerosolSpecies("OC", 12.0, 1000.0, aliases="POM")
    scenario = change_mode(scenario, 3, species=(organic,), mass_fractions=(1.0,))
    model = mam4.AerosolModel(aerosol.AerosolProcesses())
    assert model.create_input(scenario, 60.0, 2).mfpom4 == 1.0

    scenario = change_mode(scenario, 1, species=(organic,), mass_fractions=(1.0,))
    with pytest.raises(ValueError, match="mode 2.*POM"):
        model.create_input(scenario, 60.0, 2)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("fractions", [(), (0.5, 0.5)])
def test_source_mass_fraction_columns_must_match_species(index, fractions):
    scenario = change_mode(make_scenario(), index, mass_fractions=fractions)
    with pytest.raises(ValueError, match=rf"mode {index + 1}.*count must match species"):
        mam4.AerosolModel(aerosol.AerosolProcesses()).create_input(scenario, 60.0, 2)
