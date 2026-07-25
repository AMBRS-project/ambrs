"""Tests for the MAM4-JAX adapter (ambrs.mam4_jax).

MAM4-JAX is an optional dependency, so every test here is skipped when it isn't
installed.
"""

import math
import os
import tempfile
import unittest
import warnings

import numpy as np

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.mam4_jax as mam4_jax
from ambrs.scenario import Scenario

HAVE_MAM4_JAX = mam4_jax._MAM4_JAX_AVAILABLE

so4 = aerosol.AerosolSpecies(name = 'SO4', molar_mass = 97.071, density = 1770,
                             hygroscopicity = 0.507)
soa = aerosol.AerosolSpecies(name = 'MSA', molar_mass = 96.1, density = 1000,
                             hygroscopicity = 0.14)
pom = aerosol.AerosolSpecies(name = 'OC', molar_mass = 12.01, density = 1000,
                             hygroscopicity = 0.01)
bc = aerosol.AerosolSpecies(name = 'BC', molar_mass = 12.01, density = 1700,
                            hygroscopicity = 1e-10)
ncl = aerosol.AerosolSpecies(name = 'Na', molar_mass = 58.44, density = 1900,
                             hygroscopicity = 1.16)
dst = aerosol.AerosolSpecies(name = 'OIN', molar_mass = 135.0, density = 2600,
                             hygroscopicity = 0.068)
h2so4 = gas.GasSpecies(name = 'H2SO4', molar_mass = 98.079)
so2 = gas.GasSpecies(name = 'SO2', molar_mass = 64.07)

# the four MAM4 modes, in order: accumulation, aitken, coarse, primary carbon
T0 = 273.0    # [K]
P0 = 1.0e5    # [Pa]
RH0 = 0.9


def make_mode(name, species, fractions, number, gmd, gsd):
    return aerosol.AerosolModeState(
        name = name, species = tuple(species), number = number,
        geom_mean_diam = gmd, log10_geom_std_dev = math.log10(gsd),
        mass_fractions = tuple(fractions))


def make_modes():
    """The MAM4-JAX reference initial condition, expressed as AMBRS modes."""
    return (
        make_mode('accumulation', [so4, soa, ncl], [0.3, 0.3, 0.4],
                  1e8, 0.11e-6, 1.8),
        make_mode('aitken', [so4, soa, ncl], [0.3, 0.3, 0.4],
                  1e9, 0.026e-6, 1.6),
        make_mode('coarse', [so4, soa, ncl], [0.3, 0.3, 0.4],
                  1e5, 2.0e-6, 1.8),
        make_mode('primary carbon', [pom, bc], [0.5, 0.5],
                  2e8, 0.05e-6, 1.6),
    )


def make_scenario(modes = None, gases = (so2, h2so4), gas_concs = (1e-4, 1e-13)):
    modes = make_modes() if modes is None else modes
    species = tuple({s for m in modes for s in m.species})
    return Scenario(
        aerosols = species, gases = gases,
        size = aerosol.AerosolModalSizeState(modes = tuple(modes)),
        gas_concs = gas_concs, flux = 0.0, relative_humidity = RH0,
        temperature = T0, pressure = P0, height = 500.0)


@unittest.skipUnless(HAVE_MAM4_JAX, 'mam4_jax is not installed')
class TestSpeciesMapping(unittest.TestCase):
    """Unit tests for ambrs.mam4_jax.map_species_fractions"""

    def test_native_slots_are_used_where_they_exist(self):
        # accumulation (mode 0) carries all seven MAM4 species types
        fracs, remapped = mam4_jax.map_species_fractions(
            0, ('SO4', 'OC', 'BC', 'Na', 'OIN'), (0.2, 0.2, 0.2, 0.2, 0.2))
        self.assertEqual(remapped, ())
        self.assertEqual(len(fracs), 5)  # five distinct slots
        self.assertAlmostEqual(sum(fracs.values()), 1.0)

    def test_species_without_a_slot_lump_and_warn(self):
        # aitken (mode 1) has no black-carbon slot: BC lumps into sulfate
        with self.assertWarns(UserWarning):
            fracs, remapped = mam4_jax.map_species_fractions(
                1, ('SO4', 'BC'), (0.6, 0.4))
        self.assertIn('BC', remapped)
        sulfate_pcnst = mam4_jax.SLOT_OF_TYPE[1][mam4_jax._T_SULFATE]
        self.assertAlmostEqual(fracs[sulfate_pcnst], 1.0)  # 0.6 + lumped 0.4

    def test_water_is_dropped_not_lumped(self):
        h2o = aerosol.AerosolSpecies(name = 'H2O', molar_mass = 18.0,
                                     density = 1000.0)
        fracs, remapped = mam4_jax.map_species_fractions(
            0, ('SO4', 'H2O'), (0.5, 0.5))
        self.assertEqual(remapped, ())
        self.assertAlmostEqual(sum(fracs.values()), 1.0)  # renormalized dry


@unittest.skipUnless(HAVE_MAM4_JAX, 'mam4_jax is not installed')
class TestLognormalToQ(unittest.TestCase):
    """Unit tests for ambrs.mam4_jax.lognormal_to_q"""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.state, self.dropped = mam4_jax.lognormal_to_q(
                make_modes(), T0, P0, RH0)

    def test_number_round_trips_through_air_density(self):
        rho_air = mam4_jax._air_density(T0, P0)
        numbers = [self.state['q'][0, 0, p] * rho_air
                   for p in mam4_jax.NUMPTR_AMODE]
        np.testing.assert_allclose(numbers, [1e8, 1e9, 1e5, 2e8], rtol = 1e-12)

    def test_diameters_are_taken_verbatim(self):
        np.testing.assert_allclose(
            self.state['dgncur_a'][0, 0, :],
            [0.11e-6, 0.026e-6, 2.0e-6, 0.05e-6])

    def test_water_vapour_tracer_reflects_the_scenario_humidity(self):
        # wateruptake derives RH from q[0]; it must match rh * qsat(T, p)
        from mam4_jax.saturation import qsat_water
        expected = RH0 * float(qsat_water(T0, P0))
        self.assertAlmostEqual(self.state['q'][0, 0, 0], expected)
        self.assertGreater(self.state['q'][0, 0, 0], 0.0)

    def test_mode_mass_is_consistent_with_the_fixed_width_volume(self):
        # V = N * DUMFAC * D^3 must equal sum(M_i / rho_i)
        q = self.state['q'][0, 0]
        rho_air = mam4_jax._air_density(T0, P0)
        for m in range(mam4_jax.NMODES):
            volume = 0.0
            for slot in range(int(mam4_jax.NSPEC_AMODE[m])):
                pcnst = int(mam4_jax.LMASSPTR_AMODE[m, slot])
                t = mam4_jax.PCNST_TO_TYPE[pcnst]
                volume += q[pcnst] * rho_air / mam4_jax.SPECDENS_AMODE[t]
            number = q[int(mam4_jax.NUMPTR_AMODE[m])] * rho_air
            expected = (number * mam4_jax.DUMFAC_AMODE[m]
                        * self.state['dgncur_a'][0, 0, m] ** 3)
            self.assertAlmostEqual(volume / expected, 1.0, places = 10)

    def test_mismatched_gsd_warns(self):
        modes = list(make_modes())
        modes[0] = make_mode('accumulation', [so4], [1.0], 1e8, 0.11e-6, 1.2)
        with self.assertWarns(UserWarning):
            mam4_jax.lognormal_to_q(tuple(modes), T0, P0, RH0)


@unittest.skipUnless(HAVE_MAM4_JAX, 'mam4_jax is not installed')
class TestCreateInput(unittest.TestCase):
    """Unit tests for ambrs.mam4_jax.AerosolModel.create_input"""

    def setUp(self):
        self.model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(
            coagulation = True, condensation = True, nucleation = True))

    def test_input_has_the_expected_shapes_and_flags(self):
        input = self.model.create_input(make_scenario(), dt = 30.0, nstep = 10)
        self.assertEqual(input.q.shape, (1, 1, mam4_jax.PCNST))
        self.assertEqual(input.dgncur_a.shape, (1, 1, mam4_jax.NMODES))
        self.assertEqual(input.dt, 30.0)
        self.assertEqual(input.nstep, 10)
        self.assertEqual((input.mdo_gasaerexch, input.mdo_rename,
                          input.mdo_newnuc, input.mdo_coag), (1, 1, 1, 1))
        self.assertEqual(input.gaschem_rate, 0.0)  # gas_phase_chemistry off

    def test_gases_are_placed_into_q(self):
        input = self.model.create_input(make_scenario(), dt = 30.0, nstep = 1)
        self.assertEqual(input.q[0, 0, mam4_jax._SO2_IDX], 1e-4)
        self.assertEqual(input.q[0, 0, mam4_jax._H2SO4_IDX], 1e-13)
        self.assertEqual(input.so2_mmr, 1e-4)

    def test_soag_is_placed_when_present(self):
        soag = gas.GasSpecies(name = 'SOAG', molar_mass = 250.0)
        input = self.model.create_input(
            make_scenario(gases = (so2, h2so4, soag),
                          gas_concs = (1e-4, 1e-13, 5e-10)),
            dt = 30.0, nstep = 1)
        self.assertEqual(input.q[0, 0, mam4_jax._SOAG_IDX], 5e-10)

    def test_exactly_four_modes_are_required(self):
        scenario = make_scenario(modes = make_modes()[:3])
        self.assertRaises(TypeError, self.model.create_input, scenario, 30.0, 1)

    def test_invalid_arguments_are_rejected(self):
        scenario = make_scenario()
        self.assertRaises(ValueError, self.model.create_input, scenario, 0.0, 1)
        self.assertRaises(ValueError, self.model.create_input, scenario, 30.0, 0)
        non_modal = make_scenario()
        non_modal.size = None
        self.assertRaises(TypeError, self.model.create_input, non_modal, 30.0, 1)

    def test_create_inputs_covers_the_whole_ensemble(self):
        inputs = self.model.create_inputs(
            [make_scenario(), make_scenario()], dt = 30.0, nstep = 5)
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all(isinstance(i, mam4_jax.Input) for i in inputs))


@unittest.skipUnless(HAVE_MAM4_JAX, 'mam4_jax is not installed')
class TestRun(unittest.TestCase):
    """Unit tests for ambrs.mam4_jax.AerosolModel.run"""

    def test_run_produces_a_populated_output(self):
        from part2pop import ParticlePopulation
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(
            coagulation = True, condensation = True))
        scenario = make_scenario()
        output = model.run(model.create_input(scenario, dt = 30.0, nstep = 5),
                           scenario_name = 'scenario-1')
        self.assertEqual(output.model_name, 'mam4-jax')
        self.assertEqual(output.scenario_name, 'scenario-1')
        self.assertIs(output.scenario, scenario)
        self.assertEqual(output.timestep, 5)
        self.assertEqual(output.thermodynamics,
                         {'T': T0, 'p': P0, 'RH': RH0})
        self.assertIsInstance(output.particle_population, ParticlePopulation)
        self.assertGreater(output.particle_population.get_Ntot(), 0.0)

    def test_all_processes_off_preserves_the_aerosol(self):
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses())
        input = model.create_input(make_scenario(), dt = 30.0, nstep = 10)
        output = model.run(input)
        # with every mdo flag zero, amicphys is a passthrough; calcsize +
        # wateruptake must not change total number
        rho_air = mam4_jax._air_density(T0, P0)
        expected = sum(m.number for m in make_modes())
        self.assertAlmostEqual(
            output.particle_population.get_Ntot() / expected, 1.0, places = 6)

    def test_coagulation_reduces_number(self):
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation = True))
        early = model.run(model.create_input(make_scenario(), dt = 30.0, nstep = 1))
        late = model.run(model.create_input(make_scenario(), dt = 30.0, nstep = 60))
        self.assertLess(late.particle_population.get_Ntot(),
                        early.particle_population.get_Ntot())

    def test_the_output_works_with_the_framework_analysis(self):
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation = True))
        output = model.run(model.create_input(make_scenario(), dt = 30.0, nstep = 2))
        dNdlnD = np.asarray(output.compute_variable('dNdlnD'))
        self.assertGreater(float(np.sum(dNdlnD)), 0.0)

    def test_the_compiled_step_is_reused(self):
        model = mam4_jax.AerosolModel(aerosol.AerosolProcesses(coagulation = True))
        model.run(model.create_input(make_scenario(), dt = 30.0, nstep = 2))
        model.run(model.create_input(make_scenario(), dt = 30.0, nstep = 3))
        self.assertEqual(len(model._step_cache), 1)


@unittest.skipUnless(HAVE_MAM4_JAX, 'mam4_jax is not installed')
class TestEnsembleAndFiles(unittest.TestCase):
    """Unit tests for run_ensemble, the input files, and retrieve_model_state"""

    def setUp(self):
        self.model = mam4_jax.AerosolModel(
            aerosol.AerosolProcesses(coagulation = True))

    def test_run_ensemble_returns_one_output_per_input(self):
        inputs = self.model.create_inputs(
            [make_scenario() for _ in range(3)], dt = 30.0, nstep = 2)
        outputs = self.model.run_ensemble(inputs)
        self.assertEqual([o.scenario_name for o in outputs], ['1', '2', '3'])
        self.assertEqual(len(self.model._step_cache), 1)

    def test_run_ensemble_rejects_a_non_list(self):
        self.assertRaises(TypeError, self.model.run_ensemble, 'not-a-list')

    def test_input_files_round_trip(self):
        input = self.model.create_input(make_scenario(), dt = 30.0, nstep = 7)
        with tempfile.TemporaryDirectory() as dir:
            self.model.write_input_files(input, dir, 'scenario-1')
            self.assertTrue(os.path.exists(os.path.join(dir, 'scenario-1.npz')))
            self.assertTrue(os.path.exists(os.path.join(dir, 'scenario-1.txt')))
            restored = self.model.read_input(dir, 'scenario-1')
        np.testing.assert_allclose(restored.q, input.q)
        np.testing.assert_allclose(restored.dgncur_a, input.dgncur_a)
        self.assertEqual(restored.nstep, input.nstep)
        self.assertEqual(restored.mdo_coag, input.mdo_coag)
        self.assertEqual(restored.gaschem_rate, input.gaschem_rate)

    def test_write_input_files_needs_an_existing_directory(self):
        input = self.model.create_input(make_scenario(), dt = 30.0, nstep = 1)
        self.assertRaises(OSError, self.model.write_input_files,
                          input, '/no/such/directory', 'scenario-1')

    def test_retrieve_model_state_reads_and_runs(self):
        from part2pop import ParticlePopulation
        scenario = make_scenario()
        input = self.model.create_input(scenario, dt = 30.0, nstep = 5)
        with tempfile.TemporaryDirectory() as root:
            dir = os.path.join(root, 'scenario-1')
            os.mkdir(dir)
            self.model.write_input_files(input, dir, 'scenario-1')
            output = mam4_jax.retrieve_model_state(
                'scenario-1', scenario, timestep = 5,
                ensemble_output_dir = root)
        self.assertEqual(output.model_name, 'mam4-jax')
        self.assertIs(output.scenario, scenario)
        self.assertIsInstance(output.particle_population, ParticlePopulation)

    def test_retrieve_model_state_rejects_a_nonpositive_timestep(self):
        self.assertRaises(ValueError, mam4_jax.retrieve_model_state,
                          'scenario-1', make_scenario(), 0)

    def test_invocation_is_not_supported(self):
        from ambrs.aerosol_model import NotImplementedError as NotOverridden
        self.assertRaises(NotOverridden, self.model.invocation, 'exe', 'prefix')


if __name__ == '__main__':
    unittest.main()
