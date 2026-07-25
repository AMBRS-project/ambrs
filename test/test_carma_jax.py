"""Tests for the CARMA-JAX adapter (ambrs.carma_jax).

CARMA-JAX is an optional dependency, so every test here is skipped when it isn't
installed.
"""

import math
import os
import tempfile
import unittest
import warnings

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.carma_jax as carma_jax
import ambrs.gas as gas
from ambrs.scenario import Scenario

HAVE_CARMA_JAX = carma_jax._CARMA_JAX_AVAILABLE

so4 = aerosol.AerosolSpecies(name = 'SO4', molar_mass = 97.071, density = 1770,
                             hygroscopicity = 0.507)
oc = aerosol.AerosolSpecies(name = 'OC', molar_mass = 12.01, density = 1000,
                            hygroscopicity = 0.1)
h2so4 = gas.GasSpecies(name = 'H2SO4', molar_mass = 98.079)


def make_mode(number, gmd, gsd = 1.6, species = (so4,), fractions = (1.0,)):
    return aerosol.AerosolModeState(
        name = 'mode', species = tuple(species), number = number,
        geom_mean_diam = gmd, log10_geom_std_dev = math.log10(gsd),
        mass_fractions = tuple(fractions))


def make_scenario(modes = None):
    modes = modes if modes is not None else (make_mode(1e11, 8e-8),)
    species = tuple({s for m in modes for s in m.species})
    return Scenario(
        aerosols = species, gases = (h2so4,),
        size = aerosol.AerosolModalSizeState(modes = tuple(modes)),
        gas_concs = (0.0,), flux = 0.0, relative_humidity = 0.5,
        temperature = 288.0, pressure = 101325.0, height = 500.0)


@unittest.skipUnless(HAVE_CARMA_JAX, 'carma_jax is not installed')
class TestBinning(unittest.TestCase):
    """Unit tests for ambrs.carma_jax.lognormal_to_bins"""

    def setUp(self):
        self.model = carma_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))

    def test_binning_is_exact_for_a_covered_mode(self):
        # the analytic bin integral preserves total number by construction
        binned = carma_jax.lognormal_to_bins(
            1e5, 8e-8, math.log10(1.6), self.model.rlow, self.model.rup)
        self.assertAlmostEqual(np.sum(binned) / 1e5, 1.0, places = 10)
        self.assertTrue(np.all(binned >= 0.0))

    def test_a_mode_off_the_grid_warns(self):
        with self.assertWarns(UserWarning):
            carma_jax.lognormal_to_bins(
                1e5, 1e-3, math.log10(1.6),   # 1 mm particles: off the grid
                self.model.rlow, self.model.rup)


@unittest.skipUnless(HAVE_CARMA_JAX, 'carma_jax is not installed')
class TestBlendCompositions(unittest.TestCase):
    """Unit tests for ambrs.carma_jax.blend_compositions"""

    def test_identical_modes_do_not_warn(self):
        modes = (make_mode(1e11, 3e-8), make_mode(1e10, 1e-7))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            names, fracs = carma_jax.blend_compositions(modes)
        self.assertEqual(names, ('SO4',))
        self.assertEqual(fracs, (1.0,))

    def test_differing_modes_blend_and_warn(self):
        modes = (make_mode(1e11, 3e-8),
                 make_mode(1e11, 1.2e-7, species = (so4, oc),
                           fractions = (0.5, 0.5)))
        with self.assertWarns(UserWarning):
            names, fracs = carma_jax.blend_compositions(modes)
        self.assertEqual(names, ('OC', 'SO4'))
        self.assertAlmostEqual(sum(fracs), 1.0)
        # the larger mode dominates by volume, so OC is well below 0.5
        self.assertLess(dict(zip(names, fracs))['OC'], 0.5)

    def test_water_is_dropped(self):
        h2o = aerosol.AerosolSpecies(name = 'H2O', molar_mass = 18.0,
                                     density = 1000.0)
        modes = (make_mode(1e11, 1e-7, species = (so4, h2o),
                           fractions = (0.5, 0.5)),)
        names, fracs = carma_jax.blend_compositions(modes)
        self.assertEqual(names, ('SO4',))
        self.assertEqual(fracs, (1.0,))


@unittest.skipUnless(HAVE_CARMA_JAX, 'carma_jax is not installed')
class TestCreateInput(unittest.TestCase):
    """Unit tests for ambrs.carma_jax.AerosolModel.create_input"""

    def setUp(self):
        self.model = carma_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))

    def test_unsupported_processes_raise_at_construction(self):
        for flag in ('condensation', 'nucleation', 'gas_phase_chemistry'):
            processes = aerosol.AerosolProcesses(
                coagulation = True, **{flag: True})
            self.assertRaises(ValueError, carma_jax.AerosolModel, processes)

    def test_modes_are_summed_and_number_is_conserved(self):
        modes = (make_mode(5e11, 3e-8), make_mode(1e11, 1.2e-7))
        input = self.model.create_input(make_scenario(modes), dt = 60.0,
                                        nstep = 10)
        self.assertEqual(input.pc.shape, (1, self.model.nbin, 1))
        total = np.sum(input.pc[0, :, 0]) * 1e6      # [# m^-3]
        self.assertAlmostEqual(total / 6e11, 1.0, places = 8)

    def test_invalid_arguments_are_rejected(self):
        scenario = make_scenario()
        self.assertRaises(ValueError, self.model.create_input, scenario, 0.0, 1)
        self.assertRaises(ValueError, self.model.create_input, scenario, 60.0, 0)
        non_modal = make_scenario()
        non_modal.size = None
        self.assertRaises(TypeError, self.model.create_input, non_modal, 60.0, 1)

    def test_create_inputs_covers_the_whole_ensemble(self):
        inputs = self.model.create_inputs(
            [make_scenario(), make_scenario()], dt = 60.0, nstep = 5)
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all(isinstance(i, carma_jax.Input) for i in inputs))


@unittest.skipUnless(HAVE_CARMA_JAX, 'carma_jax is not installed')
class TestRun(unittest.TestCase):
    """Unit tests for ambrs.carma_jax.AerosolModel.run"""

    def setUp(self):
        self.model = carma_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))

    def test_run_produces_a_populated_output(self):
        from part2pop import ParticlePopulation
        scenario = make_scenario()
        output = self.model.run(
            self.model.create_input(scenario, dt = 60.0, nstep = 5),
            scenario_name = 'scenario-1')
        self.assertEqual(output.model_name, 'carma-jax')
        self.assertEqual(output.scenario_name, 'scenario-1')
        self.assertIs(output.scenario, scenario)
        self.assertEqual(output.timestep, 5)
        self.assertEqual(output.thermodynamics,
                         {'T': 288.0, 'p': 101325.0, 'RH': 0.5})
        self.assertIsInstance(output.particle_population, ParticlePopulation)
        self.assertGreater(output.particle_population.get_Ntot(), 0.0)

    def test_coagulation_conserves_mass_and_reduces_number(self):
        scenario = make_scenario((make_mode(5e11, 3e-8),))
        early = self.model.run(
            self.model.create_input(scenario, dt = 60.0, nstep = 1))
        late = self.model.run(
            self.model.create_input(scenario, dt = 60.0, nstep = 120))
        early_pop = early.particle_population
        late_pop = late.particle_population
        self.assertLess(late_pop.get_Ntot(), early_pop.get_Ntot())
        mass = early_pop.get_tot_dry_mass()
        self.assertLess(abs(late_pop.get_tot_dry_mass() - mass) / mass, 1e-9)

    def test_the_output_works_with_the_framework_analysis(self):
        output = self.model.run(
            self.model.create_input(make_scenario(), dt = 60.0, nstep = 2))
        dNdlnD = np.asarray(output.compute_variable('dNdlnD'))
        self.assertGreater(float(np.sum(dNdlnD)), 0.0)


@unittest.skipUnless(HAVE_CARMA_JAX, 'carma_jax is not installed')
class TestEnsembleAndFiles(unittest.TestCase):
    """Unit tests for run_ensemble, the input files, and retrieve_model_state"""

    def setUp(self):
        self.model = carma_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))

    def test_run_ensemble_returns_one_output_per_input(self):
        inputs = self.model.create_inputs(
            [make_scenario() for _ in range(3)], dt = 60.0, nstep = 2)
        outputs = self.model.run_ensemble(inputs)
        self.assertEqual([o.scenario_name for o in outputs], ['1', '2', '3'])

    def test_run_ensemble_rejects_a_non_list(self):
        self.assertRaises(TypeError, self.model.run_ensemble, 'not-a-list')

    def test_input_files_round_trip(self):
        input = self.model.create_input(make_scenario(), dt = 60.0, nstep = 7)
        with tempfile.TemporaryDirectory() as dir:
            self.model.write_input_files(input, dir, 'scenario-1')
            self.assertTrue(os.path.exists(os.path.join(dir, 'scenario-1.npz')))
            self.assertTrue(os.path.exists(os.path.join(dir, 'scenario-1.txt')))
            restored = self.model.read_input(dir, 'scenario-1')
        np.testing.assert_allclose(restored.pc, input.pc)
        self.assertEqual(restored.nstep, input.nstep)
        self.assertEqual(restored.species_names, input.species_names)

    def test_read_input_rejects_a_mismatched_grid(self):
        input = self.model.create_input(make_scenario(), dt = 60.0, nstep = 1)
        other = carma_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True), nbin = 24)
        with tempfile.TemporaryDirectory() as dir:
            self.model.write_input_files(input, dir, 'scenario-1')
            self.assertRaises(ValueError, other.read_input, dir, 'scenario-1')

    def test_retrieve_model_state_reads_and_runs(self):
        from part2pop import ParticlePopulation
        scenario = make_scenario()
        input = self.model.create_input(scenario, dt = 60.0, nstep = 5)
        with tempfile.TemporaryDirectory() as root:
            dir = os.path.join(root, 'scenario-1')
            os.mkdir(dir)
            self.model.write_input_files(input, dir, 'scenario-1')
            output = carma_jax.retrieve_model_state(
                'scenario-1', scenario, timestep = 5,
                ensemble_output_dir = root)
        self.assertEqual(output.model_name, 'carma-jax')
        self.assertIs(output.scenario, scenario)
        self.assertIsInstance(output.particle_population, ParticlePopulation)

    def test_retrieve_model_state_rejects_a_nonpositive_timestep(self):
        self.assertRaises(ValueError, carma_jax.retrieve_model_state,
                          'scenario-1', make_scenario(), 0)

    def test_invocation_is_not_supported(self):
        from ambrs.aerosol_model import NotImplementedError as NotOverridden
        self.assertRaises(NotOverridden, self.model.invocation, 'exe', 'prefix')


if __name__ == '__main__':
    unittest.main()
