import numpy as np
import scipy.stats
import unittest
from part2pop import build_population

import ambrs.analysis as analysis


def make_output(particle_population):
    return analysis.Output(
        model_name="model",
        scenario_name="scenario",
        scenario=None,
        timestep=0,
        particle_population=particle_population,
        gas_mixture=None,
        thermodynamics={},
    )


def make_particle_population(number):
    return build_population(
        {
            "type": "binned_lognormals",
            "N_sigmas": 10,
            "N_bins": 50,
            "N": [number],
            "GMD": [1.0e-7],
            "GSD": [1.6],
            "aero_spec_names": [["SO4", "NH4"]],
            "aero_spec_fracs": [[0.73, 0.27]],
        }
    )


def output_pair():
    return (
        make_output(make_particle_population(5.0e8)),
        make_output(make_particle_population(1.0e9)),
    )


class TestAnalysis(unittest.TestCase):
    def test_output_compute_variable_uses_real_part2pop_variable(self):
        output = make_output(make_particle_population(5.0e8))

        result = output.compute_variable("Nccn", {"s_grid": [0.001, 0.01, 0.1]})

        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_nmae_uses_real_outputs(self):
        output1, output2 = output_pair()
        var_cfg = {"s_grid": [0.001, 0.01, 0.1]}
        values1 = output1.compute_variable("Nccn", var_cfg)
        values2 = output2.compute_variable("Nccn", var_cfg)
        expected = np.sum(np.abs(values2 - values1)) / np.sum(np.abs(values1))

        np.testing.assert_allclose(
            analysis.nmae([output1], [output2], "Nccn", var_cfg),
            expected,
        )

    def test_nmae_returns_nan_for_empty_inputs_and_zero_denominator(self):
        zero_output = make_output(make_particle_population(0.0))
        nonzero_output = make_output(make_particle_population(5.0e8))

        self.assertTrue(np.isnan(analysis.nmae([], [], "Nccn")))
        self.assertTrue(
            np.isnan(
                analysis.nmae(
                    [zero_output],
                    [nonzero_output],
                    "Nccn",
                    {"s_grid": [0.001, 0.01, 0.1]},
                )
            )
        )

    def test_kl_divergence_forward_and_backward(self):
        output1, output2 = output_pair()
        var_cfg = {"D_grid": np.logspace(-9, -6, 8)}
        values1 = output1.compute_variable("dNdlnD", var_cfg)
        values2 = output2.compute_variable("dNdlnD", var_cfg)
        p1 = values1 / np.sum(values1)
        p2 = values2 / np.sum(values2)

        expected_forward = scipy.stats.entropy(p1, p2)
        expected_backward = scipy.stats.entropy(p2, p1)

        np.testing.assert_allclose(
            analysis.kl_divergence(output1, output2, var_cfg),
            expected_forward,
        )
        np.testing.assert_allclose(
            analysis.kl_divergence(output1, output2, {**var_cfg, "backward": True}),
            expected_backward,
        )

    def test_kl_divergence_returns_nan_for_zero_sum_distribution(self):
        output1 = make_output(make_particle_population(0.0))
        output2 = make_output(make_particle_population(5.0e8))

        self.assertTrue(
            np.isnan(
                analysis.kl_divergence(
                    output1, output2, {"D_grid": np.logspace(-9, -6, 8)}
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
