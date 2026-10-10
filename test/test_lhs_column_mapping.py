"""Regression test for Latin-hypercube mode-column indexing."""

from unittest.mock import patch

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.ppe as ppe


class _RecordingDistribution:
    def __init__(self):
        self.calls = []

    def ppf(self, q):
        q = np.asarray(q)
        self.calls.append(q.copy())
        return np.ones_like(q, dtype=float)


def test_lhs_uses_distinct_cumulative_columns_for_modes_with_different_species_counts():
    so4 = aerosol.AerosolSpecies(name="SO4", molar_mass=97.071, density=1770)
    oc = aerosol.AerosolSpecies(name="OC", molar_mass=12.01, density=1000)

    m0_number, m0_gmd, m0_gsd, m0_f0 = [_RecordingDistribution() for _ in range(4)]
    m1_number, m1_gmd, m1_gsd, m1_f0, m1_f1 = [
        _RecordingDistribution() for _ in range(5)
    ]
    flux, rh, temperature, pressure = [_RecordingDistribution() for _ in range(4)]

    specification = ppe.EnsembleSpecification(
        name="column-map",
        aerosols=(so4, oc),
        gases=(),
        size=aerosol.AerosolModalSizeDistribution(
            modes=(
                aerosol.AerosolModeDistribution(
                    name="one-species",
                    species=(so4,),
                    number=m0_number,
                    geom_mean_diam=m0_gmd,
                    log10_geom_std_dev=m0_gsd,
                    mass_fractions=(m0_f0,),
                ),
                aerosol.AerosolModeDistribution(
                    name="two-species",
                    species=(so4, oc),
                    number=m1_number,
                    geom_mean_diam=m1_gmd,
                    log10_geom_std_dev=m1_gsd,
                    mass_fractions=(m1_f0, m1_f1),
                ),
            )
        ),
        gas_concs=(),
        flux=flux,
        relative_humidity=rh,
        temperature=temperature,
        pressure=pressure,
        height=500.0,
    )

    captured = {}

    def fake_lhs(n_factors, samples, criterion=None, iterations=None):
        design = np.tile(
            np.arange(1, n_factors + 1, dtype=float) / (n_factors + 1),
            (samples, 1),
        )
        captured["design"] = design
        return design

    with patch.object(ppe.pyDOE, "lhs", side_effect=fake_lhs):
        ppe.lhs(specification, 3)

    design = captured["design"]
    expected = [
        (m0_number, 0),
        (m0_gmd, 1),
        (m0_gsd, 2),
        (m0_f0, 3),
        (m1_number, 4),
        (m1_gmd, 5),
        (m1_gsd, 6),
        (m1_f0, 7),
        (m1_f1, 8),
    ]
    for distribution, column in expected:
        assert len(distribution.calls) == 1
        np.testing.assert_array_equal(distribution.calls[0], design[:, column])
