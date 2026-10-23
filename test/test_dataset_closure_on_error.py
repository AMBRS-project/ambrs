"""NetCDF readers close their datasets even when decoding fails."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from ambrs import mam4


def test_mam4_dataset_closes_on_missing_gas_variable():
    state = SimpleNamespace(size=SimpleNamespace(modes=()), temperature=298.0,
        pressure=101325.0, relative_humidity=0.5)
    dataset = MagicMock()
    dataset.__enter__.return_value.variables = {}
    with patch.object(mam4, "Dataset", return_value=dataset), \
         patch.object(mam4, "build_population", return_value=object()):
        with pytest.raises(KeyError):
            mam4.retrieve_model_state("case", state, timestep=2)
    dataset.__exit__.assert_called_once()
