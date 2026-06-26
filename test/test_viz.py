from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import ambrs.viz as viz


def test_make_variable_configs_are_array_backed():
    dndln_cfg = viz.make_dNdlnD_cfg(D_range=(1.0e-9, 1.0e-6), N_bins=4)
    ccn_cfg = viz.make_frac_ccn_cfg(s_grid=[0.1, 1.0])
    bscat_cfg = viz.make_bscat_cfg(wvl_grid=[0.4e-6, 0.5e-6], rh_grid=[0.0, 0.9])

    assert dndln_cfg["D"].shape == (4,)
    assert dndln_cfg["normalize"] is True
    assert dndln_cfg["method"] == "kde"
    np.testing.assert_allclose(ccn_cfg["s_grid"], [0.1, 1.0])
    np.testing.assert_allclose(bscat_cfg["wvl_grid"], [0.4e-6, 0.5e-6])
    np.testing.assert_allclose(bscat_cfg["rh_grid"], [0.0, 0.9])


def test_plot_range_bars_returns_figure_with_expected_axes():
    df = pd.DataFrame(
        {
            "variable": ["temperature", "temperature", "flux", "flux"],
            "value": [280.0, 300.0, 1.0e-9, 1.0e-8],
            "sample": [1, 2, 1, 2],
        }
    )

    fig = viz.plot_range_bars(
        df,
        ["temperature", "flux"],
        scale_info={"flux": "log"},
        highlight_idx=[2],
        highlight_colors=["tab:red"],
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "Temperature"
    assert fig.axes[1].get_ylabel() == "Flux"
    assert fig.axes[1].get_xscale() == "log"
    plt.close(fig)


def test_plot_range_bars_handles_missing_variable():
    fig = viz.plot_range_bars(
        pd.DataFrame({"variable": [], "value": [], "sample": []}),
        ["temperature"],
    )

    assert len(fig.axes) == 1
    assert fig.axes[0].axison is False
    plt.close(fig)


def test_build_input_ranges_dataframe_extracts_attributes_and_gases():
    ensemble = SimpleNamespace(
        temperature=np.array([280.0, 290.0]),
        relative_humidity=np.array([0.4, 0.5]),
        pressure=101325.0,
        gas_concs=(np.array([1.0e-9, 2.0e-9]), np.array([3.0e-9, 4.0e-9])),
    )

    df = viz.build_input_ranges_dataframe(
        ensemble,
        variables={"Temperature": "temperature", "Pressure": "pressure"},
        gas_names=["SO2", "H2SO4"],
        sample_ids=[10, 11],
    )

    assert set(df["variable"]) == {
        "Temperature",
        "Pressure",
        "SO2 (mixing ratio)",
        "H2SO4 (mixing ratio)",
    }
    assert df[df["variable"] == "Pressure"]["value"].tolist() == [101325.0, 101325.0]
    assert df["sample"].min() == 10
    assert df["sample"].max() == 11


def test_build_input_ranges_dataframe_rejects_uninferable_ensemble_size():
    with pytest.raises(ValueError, match="Could not infer ensemble size"):
        viz.build_input_ranges_dataframe(SimpleNamespace())


def test_render_grid_uses_model_output_and_plotter(monkeypatch):
    calls = []

    class FakePlotter:
        def __init__(self, cfg):
            self.cfg = cfg

        def plot(self, population, ax, add_xlabel, add_ylabel, label):
            calls.append(
                {
                    "varname": self.cfg["varname"],
                    "population": population,
                    "label": label,
                    "add_xlabel": add_xlabel,
                    "add_ylabel": add_ylabel,
                }
            )
            ax.plot([1.0e-9, 1.0e-6], [1.0, 2.0], label=label)

    def fake_build_plotter(kind, cfg):
        assert kind == "state_line"
        return FakePlotter(cfg)

    def fake_retrieve_model_state(**kwargs):
        return SimpleNamespace(particle_population={"scenario": kwargs["scenario_name"]})

    ensemble = SimpleNamespace(
        __len__=lambda self: 1,
        member=lambda index: SimpleNamespace(index=index),
    )
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)

    monkeypatch.setattr(viz, "build_plotter", fake_build_plotter)
    monkeypatch.setattr(viz.partmc, "retrieve_model_state", fake_retrieve_model_state)

    returned_fig, axes = viz.render_partmc_and_mam4_variable_grid(
        gs,
        varname="dNdlnD",
        var_cfg=viz.make_dNdlnD_cfg(N_bins=3),
        ensemble=ensemble,
        scenario_names=["1"],
        timesteps=[0],
        partmc_dir="partmc-output",
        legend_loc="upper left",
        xscale="log",
    )

    assert returned_fig is fig
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_xscale() == "log"
    assert axes[0, 0].get_xlabel() == r"diameter [$\mu$m]"
    assert axes[0, 0].get_ylabel() == "normalized number density"
    assert calls == [
        {
            "varname": "dNdlnD",
            "population": {"scenario": "1"},
            "label": "PartMC",
            "add_xlabel": False,
            "add_ylabel": False,
        }
    ]
    plt.close(fig)
