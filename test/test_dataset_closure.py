"""Dataset context managers release model output resources."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from ambrs import mam4, partmc


def _netcdf(variables):
    ds = MagicMock()
    ds.__enter__.return_value.variables = variables
    return ds


def test_mam4_closes_dataset_after_retrieval():
    state = SimpleNamespace(size=SimpleNamespace(modes=()),
        temperature=298.0, pressure=101325.0, relative_humidity=0.5)
    ds = _netcdf({"h2so4_gas": np.array([2.0])})
    with patch.object(mam4, "Dataset", return_value=ds), \
         patch.object(mam4, "build_population", return_value=object()), \
         patch.object(mam4, "build_gas_mixture", return_value=object()):
        mam4.retrieve_model_state("case", state, timestep=2)
    ds.__exit__.assert_called_once()


def test_partmc_closes_dataset_after_retrieval():
    ds = _netcdf({
        "gas_species": SimpleNamespace(names="SO2"),
        "gas_mixing_ratio": np.array([10.0]),
        "temperature": np.array([298.0]),
        "pressure": np.array([101325.0]),
        "relative_humidity": np.array([0.5]),
    })
    with patch.object(partmc, "get_ncfile", return_value="out.nc"), \
         patch.object(partmc, "Dataset", return_value=ds), \
         patch.object(partmc, "build_population", return_value=object()), \
         patch.object(partmc, "build_gas_mixture", return_value=object()):
        partmc.retrieve_model_state("case", object(), timestep=1)
    ds.__exit__.assert_called_once()
