# unit tests for the ambrs.ppe package

import ambrs.ppe as ppe
import ambrs.scenario as scenario
import numpy as np
import unittest

class TestEnsemble(unittest.TestCase):
    """Unit tests for ambr.ppe.Ensemble"""

    def setUp(self):
        self.n = 100
        self.ref_scenario = Scenario(
            size = aerosol.AerosolModalSizeState(
                modes = (
                    aerosol.AerosolModeState(
                        name = "aitken",
                        species = [so4, soa, ncl],
                        number = 5e8,
                        geom_mean_diam = 1e-7,
                        mass_fractions = (0.4, 0.3, 0.3),
                    ),
                ),
            ),
            flux = 0.0,
            relative_humidity = 0.5,
            temperature = 298.0,
        )
        self.ensemble = Ensemble(
            size = aerosol.AerosolModalSizePopulation(
                modes = (
                    aerosol.AerosolModePopulation(
                        name = self.ref_scenario.size.modes[0].name,
                        species = self.ref_scenario.size.modes[0].species,
                        number = np.full(self.n, self.ref_scenario.size.modes[0].number),
                        geom_mean_diam = np.full(self.n, self.ref_scenario.size.modes[0].geom_mean_diam),
                        mass_fractions = (
                            np.full(self.n, self.ref_scenario.size.modes[0].mass_fractions[0]),
                            np.full(self.n, self.ref_scenario.size.modes[0].mass_fractions[1]),
                            np.full(self.n, self.ref_scenario.size.modes[0].mass_fractions[2]),
                        ),
                    ),
                ),
    `       ),
            flux = np.full(self.n, self.ref_scenario.flux),
            relative_humidity = np.full(self.n, self.ref_scenario.relative_humidity),
            temperature = np.full(self.n, self.ref_scenario.temperature),
        )

    def test_len(self):
        self.assertEqual(self.n, len(self.ensemble))

    def test_iteration(self):
        for scenario in self.ensemble:
            self.assertEqual(self.ref_scenario, scenario)

    def test_member(self):
        for i in range(self.n):
            self.assertEqual(self.ref_scenario, self.ensemble.member(i))

if __name__ == '__main__':
    unittest.main()
