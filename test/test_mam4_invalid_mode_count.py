"""MAM4 should report the supplied mode count, not raise AttributeError."""
from types import SimpleNamespace
import pytest

from ambrs.aerosol import AerosolModalSizeState, AerosolProcesses
from ambrs.mam4 import AerosolModel


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_rejects_incorrect_modal_mode_count(count):
    scenario = SimpleNamespace(size=AerosolModalSizeState(modes=(None,) * count))
    with pytest.raises(TypeError, match=rf"{count}-mode aerosol particle size state"):
        AerosolModel(AerosolProcesses()).create_input(scenario, 60.0, 2)
