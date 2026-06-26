from types import SimpleNamespace

import numpy as np
import pytest
import scipy.stats

import ambrs.analysis as analysis


class FakeOutput:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def compute_variable(self, varname, var_cfg=None):
        self.calls.append((varname, var_cfg))
        return self.values


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


def test_output_compute_variable_uses_build_variable_fallback(monkeypatch):
    population = object()
    calls = []

    class FakeVariable:
        def compute(self, population):
            calls.append({"population": population})
            return {"Nccn": np.array([1.0])}

    def fake_build_variable(name, scope, var_cfg):
        calls.append({"name": name, "scope": scope, "var_cfg": var_cfg})
        return FakeVariable()

    monkeypatch.delattr(analysis.ppa, "compute_variable", raising=False)
    monkeypatch.setattr(analysis.ppa, "build_variable", fake_build_variable)

    output = make_output(population)

    result = output.compute_variable("Nccn")
    np.testing.assert_allclose(result["Nccn"], [1.0])
    assert calls == [
        {"name": "Nccn", "scope": "population", "var_cfg": {}},
        {"population": population},
    ]


def test_output_compute_variable_prefers_direct_dispatcher(monkeypatch):
    population = object()
    calls = []

    def fake_compute_variable(*, population, varname, var_cfg):
        calls.append(
            {"population": population, "varname": varname, "var_cfg": var_cfg}
        )
        return {"Nccn": np.array([2.0])}

    monkeypatch.setattr(analysis.ppa, "compute_variable", fake_compute_variable, raising=False)

    output = make_output(population)

    result = output.compute_variable("Nccn", {"s_grid": [0.1]})
    np.testing.assert_allclose(result["Nccn"], [2.0])
    assert calls == [
        {"population": population, "varname": "Nccn", "var_cfg": {"s_grid": [0.1]}}
    ]


def test_nmae_collapses_dict_returned_arrays():
    output1 = FakeOutput({"Nccn": np.array([1.0, 3.0])})
    output2 = FakeOutput({"Nccn": np.array([2.0, 1.0])})

    assert analysis.nmae([output1], [output2], "Nccn", {"s": 0.1}) == pytest.approx(
        3.0 / 4.0
    )
    assert output1.calls == [("Nccn", {"s": 0.1})]
    assert output2.calls == [("Nccn", {"s": 0.1})]


def test_nmae_accepts_direct_array_return_values():
    output1 = FakeOutput(np.array([[1.0, 2.0]]))
    output2 = FakeOutput(np.array([[2.0, 4.0]]))

    assert analysis.nmae([output1], [output2], "b_ext") == pytest.approx(1.0)


def test_nmae_returns_nan_for_empty_inputs_and_zero_denominator():
    assert np.isnan(analysis.nmae([], [], "Nccn"))
    assert np.isnan(
        analysis.nmae(
            [FakeOutput({"Nccn": np.array([0.0, 0.0])})],
            [FakeOutput({"Nccn": np.array([1.0, 2.0])})],
            "Nccn",
        )
    )


def test_kl_divergence_forward_and_backward():
    output1 = FakeOutput({"dNdlnD": np.array([1.0, 3.0])})
    output2 = FakeOutput({"dNdlnD": np.array([2.0, 2.0])})

    expected_forward = scipy.stats.entropy([0.25, 0.75], [0.5, 0.5])
    expected_backward = scipy.stats.entropy([0.5, 0.5], [0.25, 0.75])

    assert analysis.kl_divergence(output1, output2) == pytest.approx(expected_forward)
    assert analysis.kl_divergence(
        output1, output2, {"backward": True}
    ) == pytest.approx(expected_backward)


def test_kl_divergence_returns_nan_for_zero_sum_distribution():
    output1 = FakeOutput({"dNdlnD": np.array([0.0, 0.0])})
    output2 = FakeOutput({"dNdlnD": np.array([1.0, 1.0])})

    assert np.isnan(analysis.kl_divergence(output1, output2))
