"""Regression test for PartMC gas-background propagation."""

import os
import tempfile

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.partmc as partmc
from ambrs.scenario import Scenario


def test_partmc_create_input_preserves_and_writes_gas_background():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=97.071, density=1770)
    so2 = gas.GasSpecies(name="SO2", molar_mass=64.07)
    background = [
        (0.0, {"rate": 0.01, "SO2": 10.0}),
        (60.0, {"rate": 0.02, "SO2": 20.0}),
    ]
    scenario = Scenario(
        aerosols=(so4,),
        gases=(so2,),
        size=aerosol.AerosolModalSizeState(
            modes=(
                aerosol.AerosolModeState(
                    name="aitken",
                    species=(so4,),
                    number=1.0e8,
                    geom_mean_diam=1.0e-7,
                    log10_geom_std_dev=0.2,
                    mass_fractions=(1.0,),
                ),
            )
        ),
        gas_concs=(1.0e-9,),
        flux=0.0,
        relative_humidity=0.5,
        temperature=298.0,
        pressure=101325.0,
        height=500.0,
        gas_background=background,
    )
    model = partmc.AerosolModel(aerosol.AerosolProcesses(), n_part=10)

    model_input = model.create_input(scenario, dt=60.0, nstep=2)

    assert model_input.gas_background is background

    with tempfile.TemporaryDirectory() as tmp:
        model.write_input_files(model_input, tmp, "case")
        with open(os.path.join(tmp, "gas_back.dat"), encoding="utf-8") as handle:
            content = handle.read()

    assert "time\t0.0\t60.0" in content
    assert "rate\t0.01\t0.02" in content
    assert "SO2\t10.0\t20.0" in content
