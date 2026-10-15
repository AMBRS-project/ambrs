"""An empty default ensemble produces a clear error, not math.log10(0)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pytest

from ambrs import viz


class EmptyEnsemble:
    def __len__(self):
        return 0


def test_empty_default_ensemble_reports_input_error():
    fig = plt.figure()
    try:
        with pytest.raises(ValueError, match="empty ensemble"):
            viz.render_partmc_and_mam4_variable_grid(
                GridSpec(1, 1, figure=fig), varname="dNdlnD", var_cfg={},
                ensemble=EmptyEnsemble(), timesteps=(1,),
                partmc_dir="unused", mam4_dir=None,
            )
    finally:
        plt.close(fig)
