from math import log10

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
import ambrs.viz as viz


so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=97.071, density=1770)
soa = aerosol.AerosolSpecies(name="OC", molar_mass=12.01, density=1000)
so2 = gas.GasSpecies(name="SO2", molar_mass=64.07)
h2so4 = gas.GasSpecies(name="H2SO4", molar_mass=98.079)


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
    n = 2
    ensemble = ppe.Ensemble(
        aerosols=(so4, soa),
        gases=(so2, h2so4),
        size=aerosol.AerosolModalSizePopulation(
            modes=(
                aerosol.AerosolModePopulation(
                    name="aitken",
                    species=(so4, soa),
                    number=np.full(n, 5.0e8),
                    geom_mean_diam=np.full(n, 1.0e-7),
                    log10_geom_std_dev=np.full(n, log10(1.6)),
                    mass_fractions=(np.full(n, 0.7), np.full(n, 0.3)),
                ),
            ),
        ),
        gas_concs=(np.array([1.0e-9, 2.0e-9]), np.array([3.0e-9, 4.0e-9])),
        flux=np.full(n, 0.0),
        temperature=np.array([280.0, 290.0]),
        relative_humidity=np.array([0.4, 0.5]),
        pressure=np.full(n, 101325.0),
        height=500.0,
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
    class EmptyEnsemble:
        pass

    try:
        viz.build_input_ranges_dataframe(EmptyEnsemble())
    except ValueError as err:
        assert "Could not infer ensemble size" in str(err)
    else:
        raise AssertionError("Expected ValueError")
