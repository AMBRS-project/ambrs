"""Retrieve measured gas values and reject invalid public record indices."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from netCDF4 import Dataset

from ambrs import mam4


def make_scenario():
    return SimpleNamespace(
        size=SimpleNamespace(modes=()),
        temperature=298.0,
        pressure=101325.0,
        relative_humidity=0.5,
    )


def write_output(root, *, soag=True):
    directory = root / "case"
    directory.mkdir()
    with Dataset(directory / "mam_output.nc", "w") as dataset:
        dataset.createDimension("time", 3)
        values = dataset.createVariable("h2so4_gas", "f8", ("time",))
        values[:] = [1e-9, 2e-9, 3e-9]
        if soag:
            values = dataset.createVariable("soag_gas", "f8", ("time",))
            values[:] = [4e-9, 5e-9, 6e-9]


@pytest.mark.parametrize("timestep", [-5, -1, 0])
def test_invalid_integer_timestep_fails_before_accessing_scenario_or_files(timestep):
    with patch.object(mam4, "Dataset") as dataset:
        with pytest.raises(ValueError, match="timestep"):
            mam4.retrieve_model_state("case", None, timestep)
    dataset.assert_not_called()


@pytest.mark.parametrize("timestep", [1.0, 2.5, "2", True, np.bool_(True), None])
def test_noninteger_timestep_is_not_coerced(timestep):
    with pytest.raises(TypeError, match="integer >= 1"):
        mam4.retrieve_model_state("case", None, timestep)


@pytest.mark.parametrize("timestep, index", [(2, 0), (np.int64(4), 2)])
def test_netcdf_gases_use_the_requested_record(tmp_path, timestep, index):
    write_output(tmp_path)
    with patch.object(mam4, "build_population", return_value=object()) as population:
        result = mam4.retrieve_model_state(
            "case", make_scenario(), timestep, ensemble_output_dir=str(tmp_path)
        )
    # build_gas_mixture uses 28.97 g/mol dry air. The existing constant
    # discrepancy with MAM4 input conversion is a separate policy decision.
    np.testing.assert_allclose(
        result.gas_mixture.mole_ratio[1:],
        [(index + 1) * 1e-9 * 28.97 / 98.079, (index + 4) * 1e-9 * 28.97 / 250.0],
    )
    assert result.timestep == timestep
    assert population.call_args.args[0]["timestep"] == index


def test_out_of_range_record_is_rejected_before_population_reconstruction(tmp_path):
    write_output(tmp_path)
    with patch.object(mam4, "build_population") as population:
        with pytest.raises(ValueError, match="timestep 5.*outside.*records"):
            mam4.retrieve_model_state(
                "case", make_scenario(), 5, ensemble_output_dir=str(tmp_path)
            )
    population.assert_not_called()
    # The failed read must also have released the dataset handle.
    with Dataset(tmp_path / "case" / "mam_output.nc", "a") as dataset:
        dataset.setncattr("opened_after_error", 1)


def test_legacy_output_without_soag_retains_its_previous_default(tmp_path):
    write_output(tmp_path, soag=False)
    with patch.object(mam4, "build_population", return_value=object()):
        result = mam4.retrieve_model_state(
            "case", make_scenario(), 2, ensemble_output_dir=str(tmp_path)
        )
    assert result.gas_mixture.mole_ratio[2] == 0.0


def test_initial_soag_is_read_from_the_written_namelist(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "namelist").write_text(
        "&chem_input\nqso2 = 1e-9,\nqh2so4 = 2e-9,\nqsoag = 3e-9,\n/\n",
        encoding="utf-8",
    )
    with patch.object(mam4, "build_population", return_value=object()):
        result = mam4.retrieve_model_state(
            "case", make_scenario(), np.int64(1), ensemble_output_dir=str(tmp_path)
        )
    np.testing.assert_allclose(
        result.gas_mixture.mole_ratio,
        [1e-9 * 28.97 / 64.07, 2e-9 * 28.97 / 98.079, 3e-9 * 28.97 / 250.0],
    )


def test_initial_namelist_without_soag_keeps_the_legacy_default(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "namelist").write_text(
        "&chem_input\nqso2 = 1e-9,\nqh2so4 = 2e-9,\n/\n",
        encoding="utf-8",
    )
    with patch.object(mam4, "build_population", return_value=object()):
        result = mam4.retrieve_model_state(
            "case", make_scenario(), 1, ensemble_output_dir=str(tmp_path)
        )
    assert result.gas_mixture.mole_ratio[2] == 0.0
