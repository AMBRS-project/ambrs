# unit tests for the ambrs.aerosol package

import ambrs.aerosol as aerosol
import numpy as np
import unittest

so4 = aerosol.AerosolSpecies(
    name = 'so4',
    molar_mass = 97.071,
)
soa = aerosol.AerosolSpecies(
    name = 'soa',
    molar_mass = 12.01,
)
ncl = aerosol.AerosolSpecies(
    name='ncl',
    molar_mass = 58.44,
)

class TestAerosolModeState(unittest.TestCase):
    """Unit tests for ambr.aerosol.AerosolModeState"""

    def setUp(self):
        self.mode_state = aerosol.AerosolModeState(
            name = "aitken",
            species = [so4, soa, ncl],
            number = 5e8,
            geom_mean_diam = 1e-7,
            mass_fractions = (0.4, 0.3, 0.3),
        )

    def test_mass_fraction(self):
        self.assertEqual(0.4, self.mode_state.mass_fraction('so4'))
        self.assertEqual(0.3, self.mode_state.mass_fraction('soa'))
        self.assertEqual(0.3, self.mode_state.mass_fraction('ncl'))

class TestAerosolModePopulation(unittest.TestCase):
    """Unit tests for ambr.aerosol.AerosolModePopulation"""

    def setUp(self):
        self.n = 100
        self.ref_state = aerosol.AerosolModeState(
            name = "aitken",
            species = [so4, soa, ncl],
            number = 5e8,
            geom_mean_diam = 1e-7,
            mass_fractions = (0.4, 0.3, 0.3),
        )
        self.mode_population = aerosol.AerosolModePopulation(
            name = self.ref_state.name,
            species = self.ref_state.species,
            number = np.full(self.n, self.ref_state.number),
            geom_mean_diam = np.full(self.n, self.ref_state.geom_mean_diam),
            mass_fractions = (
                np.full(self.n, self.ref_state.mass_fractions[0]),
                np.full(self.n, self.ref_state.mass_fractions[1]),
                np.full(self.n, self.ref_state.mass_fractions[2]),
            ),
        )

    def test_len(self):
        self.assertEqual(self.n, len(self.mode_population))

    def test_iteration(self):
        for state in self.mode_population:
            self.assertEqual(self.ref_state, state)

    def test_member(self):
        for i in range(self.n):
            self.assertEqual(self.ref_state, self.mode_population.member(i))

class TestAerosolModalSizePopulation(unittest.TestCase):
    """Unit tests for ambr.aerosol.AerosolModalSizePopulation"""

    def setUp(self):
        self.n = 100
        self.ref_state = aerosol.AerosolModalSizeState(
            modes = (
                aerosol.AerosolModeState(
                    name = "aitken",
                    species = [so4, soa, ncl],
                    number = 5e8,
                    geom_mean_diam = 1e-7,
                    mass_fractions = (0.4, 0.3, 0.3),
                ),
            ),
        )
        self.size_population = aerosol.AerosolModalSizePopulation(
            modes = (
                aerosol.AerosolModePopulation(
                    name = self.ref_state.modes[0].name,
                    species = self.ref_state.modes[0].species,
                    number = np.full(self.n, self.ref_state.modes[0].number),
                    geom_mean_diam = np.full(self.n, self.ref_state.modes[0].geom_mean_diam),
                    mass_fractions = (
                        np.full(self.n, self.ref_state.modes[0].mass_fractions[0]),
                        np.full(self.n, self.ref_state.modes[0].mass_fractions[1]),
                        np.full(self.n, self.ref_state.modes[0].mass_fractions[2]),
                    ),
                ),
            ),
        )

    def test_len(self):
        self.assertEqual(self.n, len(self.size_population))

    def test_iteration(self):
        for state in self.size_population:
            self.assertEqual(self.ref_state, state)

    def test_member(self):
        for i in range(self.n):
            self.assertEqual(self.ref_state, self.size_population.member(i))

if __name__ == '__main__':
    unittest.main()
