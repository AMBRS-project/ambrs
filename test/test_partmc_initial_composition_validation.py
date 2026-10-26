"""Reject invalid hand-built initial states before native PartMC input exists."""
import numpy as np
import pytest

from ambrs.aerosol import (
    AerosolModeState, AerosolModalSizeState, AerosolProcesses, AerosolSpecies,
)
from ambrs.partmc import AerosolModel
from ambrs.scenario import Scenario


def scenario(fractions, names=("SO4", "BC"), number=1e6):
    species = tuple(AerosolSpecies(name, molar_mass=96.0, density=1500.0) for name in names)
    mode = AerosolModeState("manual mode", species, number, 1e-7, 0.2, fractions)
    return Scenario(species, (), AerosolModalSizeState((mode,)), (),
                    0.0, 0.5, 298.0, 101325.0, 100.0)


def model(camp=False):
    return AerosolModel(AerosolProcesses(do_camp_chem=camp), n_part=100)


@pytest.mark.parametrize("camp", [False, True])
@pytest.mark.parametrize("bad", [-0.5, np.nan, np.inf, -np.inf, 0.2 + 1j, [0.5], "invalid"])
def test_invalid_initial_component_is_rejected(camp, bad):
    with pytest.raises(ValueError, match="manual mode.*SO4.*nonnegative, finite real scalar"):
        model(camp).create_input(scenario((bad, 1.5)), dt=1.0, nstep=1)


@pytest.mark.parametrize("fractions", [(1.0,), (0.2, 0.3, 0.5)])
def test_missing_or_extra_initial_fraction_is_rejected(fractions):
    with pytest.raises(ValueError, match="one mass fraction per species"):
        model().create_input(scenario(fractions), dt=1.0, nstep=1)


def test_camp_filter_does_not_hide_a_negative_source_component():
    with pytest.raises(ValueError, match="OC.*nonnegative"):
        model(True).create_input(scenario((-0.5, 1.5), names=("OC", "BC")), dt=1.0, nstep=1)


def test_zero_number_does_not_exempt_invalid_composition():
    with pytest.raises(ValueError, match="nonnegative"):
        model().create_input(scenario((-0.5, 1.5), number=0.0), dt=1.0, nstep=1)


def test_valid_proportions_and_sulfate_split_are_preserved():
    source = scenario((0.25, 0.75))
    result = model().create_input(source, dt=1.0, nstep=1)
    fractions = result.aerosol_init[0].mass_frac
    assert fractions["BC"] == 0.75
    assert fractions["SO4"] + fractions["NH4"] == pytest.approx(0.25)
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert source.size.modes[0].mass_fractions == (0.25, 0.75)


@pytest.mark.parametrize("camp", [False, True])
def test_tiny_and_zero_nonnegative_proportions_are_not_clipped(camp):
    result = model(camp).create_input(scenario((0.0, np.float64(1e-200))), dt=1.0, nstep=1)
    key = "core.mixed.BC" if camp else "BC"
    assert result.aerosol_init[0].mass_frac[key] == 1e-200
