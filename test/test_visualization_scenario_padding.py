"""Regression test for matching PoolRunner scenario directory names."""
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ambrs import viz


class _Ensemble:
    def __len__(self):
        return 12

    def member(self, index):
        assert 0 <= index < 12
        return index


def test_default_scenario_names_are_zero_padded():
    fig = plt.figure()
    gs = GridSpec(12, 1, figure=fig)
    try:
        with patch.object(viz, "_base_styles", return_value={"partmc": {}, "mam4": {}}), \
             patch.object(viz, "build_plotter") as plotter, \
             patch.object(viz.partmc, "retrieve_model_state", return_value=SimpleNamespace(particle_population=object())) as retrieve:
            viz.render_partmc_and_mam4_variable_grid(
                gs, varname="dNdlnD", var_cfg={}, ensemble=_Ensemble(),
                timesteps=(1,), partmc_dir="unused", mam4_dir=None,
            )
        names = [call.kwargs["scenario_name"] for call in retrieve.call_args_list]
        assert names == [f"{i:02d}" for i in range(1, 13)]
        assert plotter.call_count == 12
    finally:
        plt.close(fig)
