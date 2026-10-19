"""MOSAIC requires a declared aerosol-water slot, including for dry particles."""

from copy import deepcopy
from dataclasses import replace

import pytest

from ambrs.aerosol import (
    AerosolModalSizeState,
    AerosolModeState,
    AerosolProcesses,
    AerosolSpecies,
)
from ambrs.gas import GasSpecies
from ambrs.partmc import AerosolModel
from ambrs.scenario import Scenario


def make_scenario(*, aerosol_water=False, water_fraction=0.0, gas_water=False):
    carbon = AerosolSpecies("BC", molar_mass=12.01, density=1800.0)
    water = AerosolSpecies("H2O", molar_mass=18.015, density=1000.0)
    species = (carbon, water) if aerosol_water else (carbon,)
    fractions = (1.0 - water_fraction, water_fraction) if aerosol_water else (1.0,)
    return Scenario(
        aerosols=species,
        gases=(GasSpecies("H2O", molar_mass=18.015),) if gas_water else (),
        size=AerosolModalSizeState(
            modes=(
                AerosolModeState(
                    name="initial",
                    species=species,
                    number=1.0e8,
                    geom_mean_diam=1.0e-7,
                    log10_geom_std_dev=0.2,
                    mass_fractions=fractions,
                ),
            )
        ),
        gas_concs=(0.01,) if gas_water else (),
        flux=0.0,
        relative_humidity=0.5,
        temperature=298.0,
        pressure=101325.0,
        height=500.0,
    )


@pytest.mark.parametrize("gas_water", [False, True])
def test_mosaic_rejects_missing_aerosol_water(gas_water):
    model = AerosolModel(AerosolProcesses(condensation=True), n_part=100)
    scenario = make_scenario(gas_water=gas_water)

    with pytest.raises(ValueError, match="MOSAIC requires H2O as an aerosol species"):
        model.create_input(scenario, dt=1.0, nstep=1)


@pytest.mark.parametrize("water_fraction", [0.0, 0.25])
def test_mosaic_accepts_and_preserves_initial_water(water_fraction, tmp_path):
    model = AerosolModel(AerosolProcesses(condensation=True), n_part=100)
    scenario = make_scenario(aerosol_water=True, water_fraction=water_fraction)
    original = deepcopy(scenario)

    model_input = model.create_input(scenario, dt=1.0, nstep=1)
    model.write_input_files(model_input, str(tmp_path), "scenario")

    assert model_input.do_mosaic
    assert model_input.aerosol_init[0].mass_frac["H2O"] == water_fraction
    assert scenario == original
    assert any(
        line.startswith("H2O\t")
        for line in (tmp_path / "aero_data.dat").read_text().splitlines()
    )
    composition = dict(
        line.split()
        for line in (tmp_path / "aero_init_comp_1.dat").read_text().splitlines()
        if not line.startswith("#")
    )
    assert float(composition["H2O"]) == water_fraction


@pytest.mark.parametrize(
    "processes",
    [
        AerosolProcesses(coagulation=True),
        AerosolProcesses(condensation=True, do_camp_chem=True),
    ],
)
def test_non_mosaic_input_does_not_require_aerosol_water(processes, tmp_path):
    model = AerosolModel(processes, n_part=100)
    model_input = model.create_input(make_scenario(), dt=1.0, nstep=1)

    model.write_input_files(model_input, str(tmp_path), "scenario")

    assert not model_input.do_mosaic
    assert (tmp_path / "scenario.spec").is_file()


def test_direct_input_rejected_before_any_files_are_written(tmp_path):
    model = AerosolModel(AerosolProcesses(), n_part=100)
    model_input = model.create_input(make_scenario(), dt=1.0, nstep=1)
    model_input = replace(model_input, do_mosaic=True)

    with pytest.raises(ValueError, match="MOSAIC requires H2O as an aerosol species"):
        model.write_input_files(model_input, str(tmp_path), "scenario")

    assert list(tmp_path.iterdir()) == []


def test_water_alias_does_not_replace_the_required_native_species_name():
    model = AerosolModel(AerosolProcesses(condensation=True), n_part=100)
    scenario = make_scenario(aerosol_water=True)
    water = replace(scenario.aerosols[1], name="water", aliases=("H2O",))
    species = (scenario.aerosols[0], water)
    scenario = replace(
        scenario,
        aerosols=species,
        size=AerosolModalSizeState(
            modes=(replace(scenario.size.modes[0], species=species),),
        ),
    )

    with pytest.raises(ValueError, match="MOSAIC requires H2O as an aerosol species"):
        model.create_input(scenario, dt=1.0, nstep=1)
