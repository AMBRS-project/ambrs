from math import log10
import unittest

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


def make_ensemble(gas_concs=None):
    n = 2
    if gas_concs is None:
        gas_concs = (np.array([1.0e-9, 2.0e-9]), np.array([3.0e-9, 4.0e-9]))
    return ppe.Ensemble(
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
        gas_concs=gas_concs,
        flux=np.full(n, 0.0),
        temperature=np.array([280.0, 290.0]),
        relative_humidity=np.array([0.4, 0.5]),
        pressure=np.full(n, 101325.0),
        height=500.0,
    )


class TestVizHelpers(unittest.TestCase):
    def test_make_variable_configs_are_array_backed(self):
        dndln_cfg = viz.make_dNdlnD_cfg(D_range=(1.0e-9, 1.0e-6), N_bins=4)
        ccn_cfg = viz.make_frac_ccn_cfg(s_grid=[0.1, 1.0])
        bscat_cfg = viz.make_bscat_cfg(
            wvl_grid=[0.4e-6, 0.5e-6], rh_grid=[0.0, 0.9]
        )

        self.assertEqual(dndln_cfg["D"].shape, (4,))
        self.assertTrue(dndln_cfg["normalize"])
        self.assertEqual(dndln_cfg["method"], "kde")
        np.testing.assert_allclose(ccn_cfg["s_grid"], [0.1, 1.0])
        np.testing.assert_allclose(bscat_cfg["wvl_grid"], [0.4e-6, 0.5e-6])
        np.testing.assert_allclose(bscat_cfg["rh_grid"], [0.0, 0.9])

    def test_make_dndln_cfg_accepts_custom_options(self):
        cfg = viz.make_dNdlnD_cfg(
            D_range=(1.0e-8, 1.0e-7), N_bins=3, normalize=False, method="hist"
        )

        np.testing.assert_allclose(cfg["D"], np.logspace(-8, -7, 3))
        self.assertFalse(cfg["normalize"])
        self.assertEqual(cfg["method"], "hist")

    def test_scenario_colors_uses_palette_or_tab10(self):
        self.assertEqual(viz._scenario_colors(2, palette=["red", "blue", "green"]), ["red", "blue"])
        self.assertEqual(len(viz._scenario_colors(12)), 12)

    def test_base_and_row_styles_keep_model_keys_and_row_color(self):
        base_styles = viz._base_styles()
        row_styles = viz._row_styles(base_styles, "tab:orange")

        self.assertEqual(set(row_styles), {"partmc", "mam4"})
        self.assertEqual(row_styles["partmc"]["color"], "tab:orange")
        self.assertEqual(row_styles["mam4"]["color"], "tab:orange")
        self.assertEqual(row_styles["partmc"]["linestyle"], "-")
        self.assertEqual(row_styles["mam4"]["linestyle"], "--")
        self.assertEqual(row_styles["partmc"]["linewidth"], 2.0)
        self.assertEqual(row_styles["mam4"]["linewidth"], 3.0)

    def test_as_array_expands_scalars_and_rejects_wrong_length(self):
        np.testing.assert_allclose(viz._as_array(3.0, 2), [3.0, 3.0])
        np.testing.assert_allclose(viz._as_array([1.0, 2.0], 2), [1.0, 2.0])
        self.assertIsNone(viz._as_array([1.0], 2))

    def test_format_panel_sets_scales_and_spines(self):
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(1.0, 10.0)
            ax.set_ylim(1.0, 10.0)
            viz._format_panel(ax, xscale="log", yscale="log")

            self.assertEqual(ax.get_xscale(), "log")
            self.assertEqual(ax.get_yscale(), "log")
            self.assertFalse(ax.spines["top"].get_visible())
            self.assertFalse(ax.spines["right"].get_visible())
        finally:
            plt.close(fig)

    def test_format_panel_can_keep_spines(self):
        fig, ax = plt.subplots()
        try:
            viz._format_panel(ax, minimal_spines=False)

            self.assertTrue(ax.spines["top"].get_visible())
            self.assertTrue(ax.spines["right"].get_visible())
        finally:
            plt.close(fig)

    def test_add_row_label_adds_text(self):
        fig, ax = plt.subplots()
        try:
            ax.set_xlim(0.0, 10.0)
            ax.set_ylim(0.0, 2.0)

            viz._add_row_label(ax, "scenario 1", color="tab:blue")

            self.assertEqual(len(ax.texts), 1)
            self.assertEqual(ax.texts[0].get_text(), "scenario 1")
            self.assertEqual(ax.texts[0].get_color(), "tab:blue")
        finally:
            plt.close(fig)


class TestPlotRangeBars(unittest.TestCase):
    def test_plot_range_bars_returns_figure_with_expected_axes(self):
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
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(fig.axes[0].get_ylabel(), "Temperature")
            self.assertEqual(fig.axes[1].get_ylabel(), "Flux")
            self.assertEqual(fig.axes[1].get_xscale(), "log")
        finally:
            plt.close(fig)

    def test_plot_range_bars_handles_missing_variable(self):
        fig = viz.plot_range_bars(
            pd.DataFrame({"variable": [], "value": [], "sample": []}),
            ["temperature"],
        )
        try:
            self.assertEqual(len(fig.axes), 1)
            self.assertFalse(fig.axes[0].axison)
            self.assertEqual(fig.axes[0].texts[0].get_text(), "No samples")
        finally:
            plt.close(fig)

    def test_plot_range_bars_handles_equal_value_range(self):
        fig = viz.plot_range_bars(
            pd.DataFrame(
                {
                    "variable": ["temperature", "temperature"],
                    "value": [280.0, 280.0],
                    "sample": [1, 2],
                }
            ),
            ["temperature"],
        )
        try:
            xmin, xmax = fig.axes[0].get_xlim()
            self.assertLess(xmin, 280.0)
            self.assertGreater(xmax, 280.0)
        finally:
            plt.close(fig)

    def test_plot_range_bars_uses_list_highlight_colors(self):
        fig = viz.plot_range_bars(
            pd.DataFrame(
                {
                    "variable": ["temperature", "temperature"],
                    "value": [280.0, 290.0],
                    "sample": [1, 2],
                }
            ),
            ["temperature"],
            highlight_idx=[2],
            highlight_colors=["tab:red"],
        )
        try:
            self.assertGreaterEqual(len(fig.axes[0].collections), 3)
        finally:
            plt.close(fig)

    def test_plot_range_bars_uses_dict_highlight_colors(self):
        fig = viz.plot_range_bars(
            pd.DataFrame(
                {
                    "variable": ["temperature", "temperature", "temperature"],
                    "value": [280.0, 290.0, 300.0],
                    "sample": [1, 2, 3],
                }
            ),
            ["temperature"],
            highlight_idx=[2, 3],
            highlight_colors={2: "tab:red", 3: "tab:green"},
        )
        try:
            self.assertGreaterEqual(len(fig.axes[0].collections), 4)
        finally:
            plt.close(fig)


class TestBuildInputRangesDataFrame(unittest.TestCase):
    def test_extracts_attributes_and_iterable_gases(self):
        df = viz.build_input_ranges_dataframe(
            make_ensemble(),
            variables={"Temperature": "temperature", "Pressure": "pressure"},
            gas_names=["SO2", "H2SO4"],
            sample_ids=[10, 11],
        )

        self.assertEqual(
            set(df["variable"]),
            {
                "Temperature",
                "Pressure",
                "SO2 (mixing ratio)",
                "H2SO4 (mixing ratio)",
            },
        )
        self.assertEqual(
            df[df["variable"] == "Pressure"]["value"].tolist(),
            [101325.0, 101325.0],
        )
        self.assertEqual(df["sample"].min(), 10)
        self.assertEqual(df["sample"].max(), 11)

    def test_default_variables_include_available_standard_attributes(self):
        df = viz.build_input_ranges_dataframe(make_ensemble())

        self.assertEqual(
            set(df["variable"]),
            {"Temperature (K)", "Relative humidity", "Pressure (Pa)", "Flux"},
        )
        self.assertEqual(len(df), 8)

    def test_scalar_expansion_preserves_sample_ids(self):
        df = viz.build_input_ranges_dataframe(
            make_ensemble(),
            variables={"Pressure": "pressure"},
            sample_ids=[3, 4],
        )

        self.assertEqual(df["value"].tolist(), [101325.0, 101325.0])
        self.assertEqual(df["sample"].tolist(), [3, 4])

    def test_extracts_matrix_shaped_gas_concentrations(self):
        df = viz.build_input_ranges_dataframe(
            make_ensemble(
                gas_concs=np.array(
                    [
                        [1.0e-9, 3.0e-9],
                        [2.0e-9, 4.0e-9],
                    ]
                )
            ),
            variables={},
            gas_names=["SO2", "H2SO4"],
        )

        self.assertEqual(
            df[df["variable"] == "SO2 (mixing ratio)"]["value"].tolist(),
            [1.0e-9, 2.0e-9],
        )
        self.assertEqual(
            df[df["variable"] == "H2SO4 (mixing ratio)"]["value"].tolist(),
            [3.0e-9, 4.0e-9],
        )

    def test_ignores_malformed_gas_data(self):
        df = viz.build_input_ranges_dataframe(
            make_ensemble(gas_concs={"bad": [1.0, 2.0]}),
            variables={"Temperature": "temperature"},
            gas_names=["SO2", "H2SO4"],
        )

        self.assertEqual(set(df["variable"]), {"Temperature"})

    def test_infers_size_from_later_candidate_after_scalar_candidate(self):
        base = make_ensemble()
        ensemble = ppe.Ensemble(
            aerosols=base.aerosols,
            gases=base.gases,
            size=base.size,
            gas_concs=base.gas_concs,
            flux=base.flux,
            temperature=base.temperature,
            relative_humidity=0.5,
            pressure=base.pressure,
            height=base.height,
        )

        df = viz.build_input_ranges_dataframe(
            ensemble,
            variables={"Temperature": "temperature"},
        )

        self.assertEqual(df["sample"].tolist(), [1, 2])
        self.assertEqual(df["value"].tolist(), [280.0, 290.0])

    def test_omits_requested_variable_with_wrong_length_values(self):
        ensemble = make_ensemble(gas_concs=np.array([1.0]))

        df = viz.build_input_ranges_dataframe(
            ensemble,
            variables={
                "Temperature": "temperature",
                "Gas concentrations": "gas_concs",
            },
        )

        self.assertEqual(set(df["variable"]), {"Temperature"})

    def test_rejects_uninferable_ensemble_size(self):
        with self.assertRaisesRegex(ValueError, "Could not infer ensemble size"):
            viz.build_input_ranges_dataframe(object())


if __name__ == "__main__":
    unittest.main()
