"""MAM4 timestep numbering: 1 is initial, 2 is NetCDF index zero."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from ambrs import mam4


def test_post_initial_timestep_two_reads_netcdf_zero():
    state = SimpleNamespace(size=SimpleNamespace(modes=()), temperature=298.0,
        pressure=101325.0, relative_humidity=0.5)
    ds = SimpleNamespace(variables={"h2so4_gas": np.array([42.0])})
    with patch.object(mam4, "Dataset", return_value=ds), \
         patch.object(mam4, "build_population", return_value=object()) as particle, \
         patch.object(mam4, "build_gas_mixture", return_value=object()) as gas:
        output = mam4.retrieve_model_state("case", state, timestep=2)
    assert output.timestep == 2
    assert particle.call_args.args[0]["timestep"] == 0
    assert gas.call_args.args[0]["H2SO4"] == 42.0


def test_zero_timestep_is_invalid():
    state = SimpleNamespace(size=SimpleNamespace(modes=()), temperature=298.0,
        pressure=101325.0, relative_humidity=0.5)
    with pytest.raises(ValueError, match="timestep=0 is invalid"):
        mam4.retrieve_model_state("case", state, timestep=0)
