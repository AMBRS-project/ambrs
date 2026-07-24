"""Tests for the TOMAS-JAX adapter (ambrs.tomas_jax).

TOMAS-JAX is an optional dependency, so every test here is skipped when it isn't
installed.
"""

import math
import unittest
import warnings

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.tomas_jax as tomas_jax
from ambrs.scenario import Scenario

HAVE_TOMAS = tomas_jax._TOMAS_AVAILABLE
BOXVOL = 1e6 # [cm^3]

so4 = aerosol.AerosolSpecies(name = 'SO4', molar_mass = 97.071, density = 1770,
                             hygroscopicity = 0.507)
oc = aerosol.AerosolSpecies(name = 'OC', molar_mass = 12.01, density = 1000,
                            hygroscopicity = 0.1)
bc = aerosol.AerosolSpecies(name = 'BC', molar_mass = 12.01, density = 1800,
                            hygroscopicity = 0.0)
h2so4 = gas.GasSpecies(name = 'H2SO4', molar_mass = 98.079)
so2 = gas.GasSpecies(name = 'SO2', molar_mass = 64.07)


def bin_boundaries():
    from tomas_jax.core.config import xk_boundaries
    return np.asarray(xk_boundaries(), dtype = float)


def make_mode(name, species, number, geom_mean_diam, gsd = 1.6):
    """A single internally-mixed mode with mass split evenly across species."""
    return aerosol.AerosolModeState(
        name = name,
        species = tuple(species),
        number = number,
        geom_mean_diam = geom_mean_diam,
        log10_geom_std_dev = math.log10(gsd),
        mass_fractions = tuple([1.0 / len(species)] * len(species)),
    )


def make_scenario(modes = None, gases = (h2so4,), gas_concs = (1e7,),
                  temperature = 298.0, pressure = 101325.0):
    """A scenario with one sulfate Aitken mode unless modes are given."""
    if modes is None:
        modes = [make_mode('aitken', [so4], 1e9, 8e-8)]
    species = tuple({s for m in modes for s in m.species})
    return Scenario(
        aerosols = species,
        gases = gases,
        size = aerosol.AerosolModalSizeState(modes = tuple(modes)),
        gas_concs = gas_concs,
        flux = 0.0,
        relative_humidity = 0.5,
        temperature = temperature,
        pressure = pressure,
        height = 500.0,
    )


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


@unittest.skipUnless(HAVE_TOMAS, 'tomas_jax is not installed')
class TestCreateInput(unittest.TestCase):
    """Unit tests for ambrs.tomas_jax.AerosolModel.create_input"""

    def setUp(self):
        self.processes = aerosol.AerosolProcesses(
            coagulation = True, condensation = True, nucleation = True)
        self.model = tomas_jax.AerosolModel(self.processes)

    def test_input_has_the_expected_shapes_and_fields(self):
        from tomas_jax.core.config import ICOMP, N_GAS_SPECIES, SRTSO4
        input = self.model.create_input(make_scenario(), dt = 60.0, nstep = 10)
        nbins = len(input.xk) - 1
        self.assertEqual(input.Nk.shape, (nbins,))
        self.assertEqual(input.Mk.shape, (nbins, ICOMP))
        self.assertEqual(input.Gc.shape, (N_GAS_SPECIES,))
        self.assertEqual(input.dt, 60.0)
        self.assertEqual(input.nstep, 10)
        self.assertEqual(input.temp, 298.0)
        self.assertEqual(input.pres, 101325.0)
        self.assertTrue(np.all(input.Nk >= 0.0))
        self.assertGreater(input.Gc[SRTSO4], 0.0) # H2SO4 placed into Gc

    def test_processes_are_ordered_as_tomas_applies_them(self):
        input = self.model.create_input(make_scenario(), dt = 60.0, nstep = 10)
        self.assertEqual(input.processes,
                         ('nucleation', 'coagulation', 'condensation'))

    def test_disabled_processes_are_omitted(self):
        model = tomas_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))
        input = model.create_input(make_scenario(), dt = 60.0, nstep = 10)
        self.assertEqual(input.processes, ('coagulation',))

    def test_modes_are_summed_and_number_is_conserved(self):
        modes = [make_mode('aitken', [so4], 1e9, 3e-8),
                 make_mode('accumulation', [so4], 2e9, 1.2e-7)]
        input = self.model.create_input(
            make_scenario(modes = modes), dt = 60.0, nstep = 10)
        total = sum(m.number for m in modes)              # [# m^-3]
        recovered = np.sum(input.Nk) / self.model.boxvol * 1e6
        self.assertLess(abs(recovered - total) / total, 0.10)

    def test_species_without_a_tomas_slot_are_recorded(self):
        scenario = make_scenario(
            modes = [make_mode('accumulation', [so4, oc, bc], 1e9, 1e-7)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            input = self.model.create_input(scenario, dt = 60.0, nstep = 10)
        self.assertIn('BC', input.dropped_species)
        # mass is lumped rather than dropped
        expected = np.sum(input.Nk * np.sqrt(input.xk[:-1] * input.xk[1:]))
        self.assertAlmostEqual(np.sum(input.Mk) / expected, 1.0, places = 5)

    def test_scenario_nh3_overrides_the_nucleation_default(self):
        nh3 = gas.GasSpecies(name = 'NH3', molar_mass = 17.031)
        input = self.model.create_input(
            make_scenario(gases = (h2so4, nh3), gas_concs = (1e7, 5e9)),
            dt = 60.0, nstep = 10)
        self.assertEqual(input.nh3_conc, 5e9)

    def test_create_inputs_covers_the_whole_ensemble(self):
        scenarios = [make_scenario(), make_scenario(), make_scenario()]
        inputs = self.model.create_inputs(scenarios, dt = 60.0, nstep = 10)
        self.assertEqual(len(inputs), 3)
        self.assertTrue(all(isinstance(i, tomas_jax.Input) for i in inputs))

    def test_invalid_arguments_are_rejected(self):
        scenario = make_scenario()
        self.assertRaises(ValueError, self.model.create_input, scenario, 0.0, 10)
        self.assertRaises(ValueError, self.model.create_input, scenario, 60.0, 0)
        non_modal = make_scenario()
        non_modal.size = None
        self.assertRaises(TypeError, self.model.create_input, non_modal, 60.0, 10)


if __name__ == '__main__':
    unittest.main()
