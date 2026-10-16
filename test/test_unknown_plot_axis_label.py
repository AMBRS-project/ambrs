"""Unknown plotted variables retain a defined x-axis label."""
from unittest.mock import patch
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ambrs import viz


def test_unknown_variable_does_not_reference_unbound_axis_label():
    class Ensemble:
        def __len__(self):
            return 1
        def member(self, i):
            return object()
    fig = plt.figure()
    try:
        with patch.object(viz.partmc, "retrieve_model_state", return_value=SimpleNamespace(particle_population=object())), \
             patch.object(viz, "build_plotter") as plotter:
            viz.render_partmc_and_mam4_variable_grid(
                GridSpec(1, 1, figure=fig), varname="custom_unknown_variable",
                var_cfg={}, ensemble=Ensemble(), timesteps=(1,),
                partmc_dir="unused", mam4_dir=None,
            )
        assert plotter.called
        assert any("x-axis" in axis.get_xlabel() for axis in fig.axes)
    finally:
        plt.close(fig)
