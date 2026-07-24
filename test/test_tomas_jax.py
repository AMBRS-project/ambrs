"""Tests for the TOMAS-JAX adapter (ambrs.tomas_jax).

TOMAS-JAX is an optional dependency, so every test here is skipped when it isn't
installed.
"""

import math
import unittest
import warnings

import numpy as np

import ambrs.tomas_jax as tomas_jax

HAVE_TOMAS = tomas_jax._TOMAS_AVAILABLE
BOXVOL = 1e6 # [cm^3]


def bin_boundaries():
    from tomas_jax.core.config import xk_boundaries
    return np.asarray(xk_boundaries(), dtype = float)


@unittest.skipUnless(HAVE_TOMAS, 'tomas_jax is not installed')
class TestSpeciesMapping(unittest.TestCase):
    """Unit tests for ambrs.tomas_jax.map_species_fractions"""

    def test_species_map_to_their_tomas_indices(self):
        from tomas_jax.core.config import SRTSO4, SRTNH4, SRTORG1
        fracs, remapped = tomas_jax.map_species_fractions(
            ('SO4', 'OC', 'NH4'), (0.5, 0.3, 0.2))
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(fracs[SRTSO4], 0.5)
        self.assertAlmostEqual(fracs[SRTORG1], 0.3) # OC -> organic slot
        self.assertAlmostEqual(fracs[SRTNH4], 0.2)

    def test_organics_aggregate_into_a_single_slot(self):
        from tomas_jax.core.config import SRTORG1
        fracs, remapped = tomas_jax.map_species_fractions(
            ('OC', 'ARO1', 'LIM2'), (0.2, 0.3, 0.1))
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(fracs[SRTORG1], 0.6)

    def test_species_without_a_tomas_slot_warn_but_conserve_mass(self):
        from tomas_jax.core.config import SRTSO4, SRTORG1
        with self.assertWarns(UserWarning):
            fracs, remapped = tomas_jax.map_species_fractions(
                ('SO4', 'BC'), (0.7, 0.3))
        self.assertIn('BC', remapped)
        self.assertAlmostEqual(fracs[SRTSO4], 0.7)
        self.assertAlmostEqual(fracs[SRTORG1], 0.3) # lumped, not dropped
        self.assertAlmostEqual(sum(fracs.values()), 1.0)


@unittest.skipUnless(HAVE_TOMAS, 'tomas_jax is not installed')
class TestLognormalBinning(unittest.TestCase):
    """Unit tests for ambrs.tomas_jax.lognormal_to_bins and distribute_mass"""

    def setUp(self):
        self.xk = bin_boundaries()

    def test_binning_conserves_number(self):
        n_total = 1e4 # [# cm^-3]
        Nk = tomas_jax.lognormal_to_bins(
            self.xk, n_total, 1e-7, math.log10(1.6), BOXVOL)
        self.assertEqual(Nk.shape, (len(self.xk) - 1,))
        self.assertTrue(np.all(Nk >= 0.0))
        # midpoint integration of a mode well inside the grid, to within a few %
        self.assertLess(abs(np.sum(Nk) / BOXVOL - n_total) / n_total, 0.05)

    def test_mass_is_split_across_the_requested_species(self):
        from tomas_jax.core.config import ICOMP, SRTSO4, SRTORG1
        Nk = tomas_jax.lognormal_to_bins(
            self.xk, 1e4, 1e-7, math.log10(1.6), BOXVOL)
        Mk = tomas_jax.distribute_mass(Nk, self.xk, {SRTSO4: 0.8, SRTORG1: 0.2})
        self.assertEqual(Mk.shape, (len(self.xk) - 1, ICOMP))
        expected = np.sum(Nk * np.sqrt(self.xk[:-1] * self.xk[1:]))
        self.assertAlmostEqual(np.sum(Mk) / expected, 1.0, places = 6)
        self.assertAlmostEqual(np.sum(Mk[:, SRTSO4]) / expected, 0.8, places = 6)
        self.assertAlmostEqual(np.sum(Mk[:, SRTORG1]) / expected, 0.2, places = 6)


if __name__ == '__main__':
    unittest.main()
