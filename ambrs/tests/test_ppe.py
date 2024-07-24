# unit tests for the ambrs.ppe package

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.scenario import Scenario
import math
import numpy as np
import scipy.stats
import unittest

# relevant aerosol and gas species
so4 = aerosol.AerosolSpecies(name='so4', molar_mass = 97.071)
pom = aerosol.AerosolSpecies(name='pom', molar_mass = 12.01)
soa = aerosol.AerosolSpecies(name='soa', molar_mass = 12.01)
bc  = aerosol.AerosolSpecies(name='bc', molar_mass = 12.01)
dst = aerosol.AerosolSpecies(name='dst', molar_mass = 135.065)
ncl = aerosol.AerosolSpecies(name='ncl', molar_mass = 58.44)

so2   = gas.GasSpecies(name='so2', molar_mass = 64.07)
h2so4 = gas.GasSpecies(name='h2so4', molar_mass = 98.079)
soag  = gas.GasSpecies(name='soag', molar_mass = 12.01)

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
        self.ensemble = ppe.Ensemble(
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
            ),
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

    def test_ensemble_from_scenarios(self):
        ensemble = ppe.ensemble_from_scenarios([self.ref_scenario for i in range(self.n)])
        for scenario in ensemble:
            self.assertEqual(self.ref_scenario, scenario)

class TestSampling(unittest.TestCase):
    """Unit tests for ambr.ppe sampling functions"""

    def setUp(self):
        self.n = 100
        self.ensemble_spec = ppe.EnsembleSpecification(
            name = 'mam4_ensemble',
            size = aerosol.AerosolModalSizeDistribution(
                modes = [
                    aerosol.AerosolModeDistribution(
                        name = "accumulation",
                        species = [so4, pom, soa, bc, dst, ncl],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-7, 1.1e-7),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # soa
                            scipy.stats.uniform(0, 1), # bc
                            scipy.stats.uniform(0, 1), # dst
                            scipy.stats.uniform(0, 1), # ncl
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "aitken",
                        species = [so4, soa, ncl],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-8, 3e-8),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # soa
                            scipy.stats.uniform(0, 1), # ncl
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "coarse",
                        species = [dst, ncl, so4, bc, pom, soa],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-6, 2e-6),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # dst
                            scipy.stats.uniform(0, 1), # ncl
                            scipy.stats.uniform(0, 1), # so4
                            scipy.stats.uniform(0, 1), # bc
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # soa
                        ],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "primary carbon",
                        species = [pom, bc],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-8, 6e-8),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # bc
                        ],
                    ),
                ],
            ),
            flux = scipy.stats.loguniform(1e-2*1e-9, 1e1*1e-9),
            relative_humidity = scipy.stats.loguniform(1e-5, 0.99),
            temperature = scipy.stats.uniform(240, 310),
        )

    def test_sample(self):
        ensemble = ppe.sample(self.ensemble_spec, self.n)
        self.assertEqual(self.n, len(ensemble))
        self.assertIsNotNone(ensemble.specification)
        for member in ensemble:
            self.assertIsInstance(member.size, aerosol.AerosolModalSizeState)
            self.assertEqual(4, len(member.size.modes))
            for mode in member.size.modes:
                self.assertTrue(mode.number >= 3e7)
                self.assertTrue(mode.number <= 2e12)
                self.assertTrue(mode.geom_mean_diam >= 0.5e-8)
                self.assertTrue(mode.geom_mean_diam <= 2e-6)
                self.assertTrue(sum(mode.mass_fractions) - 1.0 < 1e-12)

    def test_lhs(self):
        ensemble = ppe.lhs(self.ensemble_spec, self.n)
        self.assertEqual(self.n, len(ensemble))
        self.assertIsNotNone(ensemble.specification)
        for member in ensemble:
            self.assertIsInstance(member.size, aerosol.AerosolModalSizeState)
            self.assertEqual(4, len(member.size.modes))
            for mode in member.size.modes:
                self.assertTrue(mode.number >= 3e7)
                self.assertTrue(mode.number <= 2e12)
                self.assertTrue(mode.geom_mean_diam >= 0.5e-8)
                self.assertTrue(mode.geom_mean_diam <= 2e-6)
                self.assertTrue(sum(mode.mass_fractions) - 1.0 < 1e-12)

    def test_temperature_sweep(self):
        ref_state = ppe.sample(self.ensemble_spec, 1).member(0)
        sweeps = ppe.AerosolParameterSweeps(
            temperature = ppe.LinearParameterSweep(273.0, 373.0, 100),
        )
        ensemble = ppe.sweep(ref_state, sweeps)
        self.assertEqual(100, len(ensemble))
        for i, member in enumerate(ensemble):
            Ti = 273.0 + 1.0*i
            self.assertTrue(abs(Ti - member.temperature) < 1e-12)

    def test_aitken_mode_number_conc_sweep(self):
        ref_state = ppe.sample(self.ensemble_spec, 1).member(0)
        sweeps = ppe.AerosolParameterSweeps(
            size = ppe.AerosolModalSizeParameterSweeps(
                modes = (None, # we sweep the number concentration in the aitken mode
                         ppe.AerosolModeParameterSweeps(
                            species = self.ensemble_spec.size.modes[1].species,
                            number = ppe.LogarithmicParameterSweep(3e7, 2e12, 100),
                         ),
                         None,
                         None),
            ),
        )
        ensemble = ppe.sweep(ref_state, sweeps)
        self.assertEqual(100, len(ensemble))
        for i, member in enumerate(ensemble):
            step = (math.log10(2e12) - math.log10(3e7)) / 100
            log_ni = math.log10(3e7) + step
            print(log_ni, math.log10(member.size.modes[1].number), abs(log_ni - math.log10(member.size.modes[1].number)))
            self.assertTrue(abs(log_ni - math.log10(member.size.modes[1].number)) < 1e-12)

if __name__ == '__main__':
    unittest.main()
