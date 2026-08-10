import tempfile
from math import log10
from pathlib import Path

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.emissions import AerosolEmissions
from ambrs.scenario import Scenario
from ambrs.workflow import PartMCAerosolModel, lhs


class IdentityDistribution:
    def ppf(self, q):
        return q

    def rvs(self, n):
        return np.linspace(0.1, 0.9, n)


def _build_specification():
    a1 = aerosol.AerosolSpecies("A1", 50.0, 1000.0, hygroscopicity=0.1)
    a2 = aerosol.AerosolSpecies("A2", 60.0, 1200.0, hygroscopicity=0.2)
    g1 = gas.GasSpecies("G1", 30.0)
    d = IdentityDistribution()
    specification = ppe.EnsembleSpecification(
        name="workflow_index_test",
        aerosols=(a1, a2),
        gases=(g1,),
        size=aerosol.AerosolModalSizeDistribution(
            modes=(
                aerosol.AerosolModeDistribution("m1", (a1, a2), d, d, d, (d, d)),
                aerosol.AerosolModeDistribution("m2", (a1,), d, d, d, (d,)),
            )
        ),
        gas_concs=(d,),
        flux=d,
        relative_humidity=d,
        temperature=d,
        pressure=d,
        height=100.0,
    )
    return specification, a1, g1


def test_lhs_uses_distinct_cumulative_factor_columns(monkeypatch):
    specification, _, _ = _build_specification()

    def fake_lhs(n_factors, n, criterion=None, iterations=None):
        values = np.arange(1, n_factors + 1, dtype=float) / (n_factors + 1.0)
        return np.tile(values, (n, 1))

    monkeypatch.setattr("ambrs.workflow.pyDOE.lhs", fake_lhs)
    ensemble = lhs(specification, 2, seed=42)
    member = ensemble.member(0)

    # Factor order:
    # m1 N,GMD,GSD,mf1,mf2 = 1..5
    # m2 N,GMD,GSD,mf1 = 6..9
    # gas = 10; flux,RH,T,P = 11..14
    denominator = 15.0
    assert np.isclose(member.size.modes[0].number, 1 / denominator)
    assert np.isclose(member.size.modes[0].geom_mean_diam, 2 / denominator)
    assert np.isclose(member.size.modes[0].log10_geom_std_dev, 3 / denominator)
    assert np.isclose(
        member.size.modes[0].mass_fractions[0]
        / member.size.modes[0].mass_fractions[1],
        4 / 5,
    )
    assert np.isclose(member.size.modes[1].number, 6 / denominator)
    assert np.isclose(member.gas_concs[0], 10 / denominator)
    assert np.isclose(member.flux, 11 / denominator)
    assert np.isclose(member.relative_humidity, 12 / denominator)
    assert np.isclose(member.temperature, 13 / denominator)
    assert np.isclose(member.pressure, 14 / denominator)


def test_lhs_seed_is_reproducible_and_preserves_rng_state():
    specification, _, _ = _build_specification()

    np.random.seed(0)
    reference = np.random.random(10)
    np.random.seed(0)
    ensemble1 = lhs(specification, 5, seed=7)
    after = np.random.random(10)
    np.testing.assert_array_equal(reference, after)

    ensemble2 = lhs(specification, 5, seed=7)
    for index in range(5):
        member1 = ensemble1.member(index)
        member2 = ensemble2.member(index)
        assert member1.temperature == member2.temperature
        assert member1.relative_humidity == member2.relative_humidity
        assert member1.flux == member2.flux
        assert member1.pressure == member2.pressure
        for mode1, mode2 in zip(member1.size.modes, member2.size.modes):
            assert mode1.number == mode2.number
            assert mode1.geom_mean_diam == mode2.geom_mean_diam
            np.testing.assert_array_equal(mode1.mass_fractions, mode2.mass_fractions)


def test_partmc_adapter_forwards_background_and_writes_aerosol_sources():
    _, a1, g1 = _build_specification()
    state = aerosol.AerosolModalSizeState(
        modes=(
            aerosol.AerosolModeState(
                "source",
                (a1,),
                1.0e6,
                1.0e-7,
                log10(1.6),
                (1.0,),
            ),
        )
    )
    scenario = Scenario(
        aerosols=(a1,),
        gases=(g1,),
        size=state,
        gas_concs=(1.0e-9,),
        flux=0.0,
        relative_humidity=0.5,
        temperature=290.0,
        pressure=100000.0,
        height=100.0,
        gas_background=[(0.0, {"rate": 0.0, "G1": 0.0})],
        aerosol_emissions=[AerosolEmissions(0.0, 1.0e-5, state)],
    )

    model = PartMCAerosolModel(
        aerosol.AerosolProcesses(condensation=True),
        n_part=10,
    )
    input_data = model.create_input(scenario, 60.0, 1)
    assert input_data.gas_background == scenario.gas_background

    with tempfile.TemporaryDirectory() as directory:
        model.write_input_files(input_data, directory, "member")
        assert Path(directory, "aero_emit.dat").exists()
        assert Path(directory, "aero_emit_dist_1_dist.dat").exists()
