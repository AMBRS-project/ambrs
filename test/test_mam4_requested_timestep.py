"""The externally requested timestep is distinct from the NetCDF index."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from ambrs import mam4


def test_retrieved_mam4_output_keeps_requested_timestep():
    scenario = SimpleNamespace(
        size=SimpleNamespace(modes=()), temperature=298.0,
        pressure=101325.0, relative_humidity=0.5,
    )
    dataset = MagicMock()
    dataset.__enter__.return_value.variables = {"h2so4_gas": np.array([1.0, 2.0, 3.0])}
    with patch.object(mam4, "Dataset", return_value=dataset), \
         patch.object(mam4, "build_population", return_value=object()) as population, \
         patch.object(mam4, "build_gas_mixture", return_value=object()) as gases:
        result = mam4.retrieve_model_state("case", scenario, timestep=4)
    assert result.timestep == 4
    assert population.call_args.args[0]["timestep"] == 4
    assert gases.call_args.args[0]["H2SO4"] == 3.0
