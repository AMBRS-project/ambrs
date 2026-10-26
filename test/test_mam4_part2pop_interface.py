"""Exercise retrieval with real files and the real part2pop reconstruction."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset
from part2pop.constants import MOLAR_MASS_DRY_AIR, R

from ambrs import mam4


@pytest.fixture
def mam4_run(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    modes = tuple(
        SimpleNamespace(
            species=(SimpleNamespace(name=name),),
            number=1.0e6 if index == 0 else 0.0,
            geom_mean_diam=100.0e-9,
            log10_geom_std_dev=np.log10(1.6),
            mass_fractions=(1.0,),
        )
        for index, name in enumerate(("SO4", "SO4", "SO4", "OC"))
    )
    scenario = SimpleNamespace(
        size=SimpleNamespace(modes=modes),
        temperature=298.15,
        pressure=101325.0,
        relative_humidity=0.5,
    )
    lines = ["&chem_input"]
    for mode in range(1, 5):
        lines.extend((
            f"numc{mode} = {1.0e6 if mode == 1 else 0.0},",
            f"mfso4{mode} = 1.0,",
            f"mfpom{mode} = 0.0,",
        ))
    lines.extend(("qh2so4 = 1.0e-12,", "qso2 = 2.0e-12,", "qsoag = 0.0,", "/"))
    (directory / "namelist").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with Dataset(directory / "mam_output.nc", "w") as dataset:
        dataset.createDimension("mode", 4)
        dataset.createDimension("time", 3)
        for name in ("num_aer", "so4_aer", "soa_aer", "dgn_a", "dgn_awet"):
            values = dataset.createVariable(name, "f8", ("mode", "time"))
            values[:] = 0.0
            if name == "num_aer":
                values[0, :] = [10.0, 20.0, 30.0]
            elif name == "so4_aer":
                values[0, :] = [1.0e-10, 3.0e-10, 6.0e-10]
            elif name == "soa_aer":
                values[0, :] = [3.0e-10, 3.0e-10, 2.0e-10]
            elif name == "dgn_a":
                values[:] = 100.0e-9
            else:
                values[:] = 120.0e-9
        gases = dataset.createVariable("h2so4_gas", "f8", ("time",))
        gases[:] = [1.0e-12, 2.0e-12, 3.0e-12]
    return scenario, str(tmp_path)


@pytest.mark.parametrize("timestep, record", [(2, 0), (3, 1), (4, 2)])
def test_gas_and_population_use_the_same_first_interior_and_final_record(mam4_run, timestep, record):
    scenario, root = mam4_run
    output = mam4.retrieve_model_state(
        "case", scenario, timestep, ensemble_output_dir=root, N_bins=8
    )
    population = output.particle_population
    rho_dry_air = MOLAR_MASS_DRY_AIR * scenario.pressure / (R * scenario.temperature)
    assert output.timestep == timestep
    assert population.get_Ntot() == pytest.approx(10.0 * (record + 1) * rho_dry_air)
    sulfate = population.get_mass_conc("SO4")
    organic = population.get_mass_conc("OC")
    assert sulfate / (sulfate + organic) == pytest.approx((0.25, 0.5, 0.75)[record])
    gas_index = [gas.name for gas in output.gas_mixture.species].index("H2SO4")
    assert output.gas_mixture.mole_ratio[gas_index] == pytest.approx(
        (record + 1) * 1.0e-12 * 28.97 / 98.079, rel=1.0e-12, abs=0.0
    )


@pytest.mark.parametrize("timestep", [1, 2, 4])
def test_material_and_diagnostic_overrides_reach_initial_and_output_populations(mam4_run, timestep):
    scenario, root = mam4_run
    modifications = {
        "SO4": {"density": 1400.0, "kappa": 0.25,
                "n_550": 1.8, "m_log10_Jhet": 0.0, "b_log10_Jhet": 12.0}
    }
    original = deepcopy(modifications)
    output = mam4.retrieve_model_state(
        "case", scenario, timestep, species_modifications=modifications,
        ensemble_output_dir=root, N_bins=8,
    )
    population = output.particle_population
    sulfate = next(species for species in population.species if species.name == "SO4")
    assert sulfate.density == 1400.0
    assert sulfate.kappa == 0.25
    assert population.species_modifications == original
    assert modifications == original
