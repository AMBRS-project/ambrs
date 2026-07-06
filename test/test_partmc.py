# unit tests for the ambrs.partmc package

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.scenario import Scenario
import ambrs.partmc as partmc

from math import log10
import numpy as np
import os
import scipy.stats
import tempfile
import unittest

# relevant aerosol and gas species
so4 = aerosol.AerosolSpecies(
    name='SO4',
    molar_mass = 97.071, # NOTE: 1000x smaller than "molecular weight"!
    density = 1770,
    hygroscopicity = 0.507,
)
pom = aerosol.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
soa = aerosol.AerosolSpecies(
    name='OC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
bc = aerosol.AerosolSpecies(
    name='BC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
dst = aerosol.AerosolSpecies(
    name='OIN',
    molar_mass = 135.065,
    density = 1000,
    hygroscopicity = 0.5,
)
ncl = aerosol.AerosolSpecies(
    name='Cl',
    molar_mass = 58.44,
    density = 1000,
    hygroscopicity = 0.5,
)

so2 = gas.GasSpecies(
    name='SO2',
    molar_mass = 64.07,
)
h2so4 = gas.GasSpecies(
    name='H2SO4',
    molar_mass = 98.079,
)
#soag = gas.GasSpecies(
#    name='soag',
#    molar_mass = 12.01,
#)

# reference pressure and height
p0 = 101325 # [Pa]
h0 = 500    # [m]

class TestPartMCInput(unittest.TestCase):
    """Unit tests for ambr.partmc.Input"""

    def setUp(self):
        self.n = 100
        self.ensemble_spec = ppe.EnsembleSpecification(
            name = 'partmc_ensemble',
            aerosols = (so4, pom, soa, bc, dst, ncl),
            gases = (so2, h2so4),#, soag),
            size = aerosol.AerosolModalSizeDistribution(
                modes = [
                    aerosol.AerosolModeDistribution(
                        name = "accumulation",
                        species = [so4, pom, soa, bc, dst, ncl],
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-7, 1.1e-7),
                        log10_geom_std_dev = log10(1.6),
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
                        log10_geom_std_dev = log10(1.6),
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
                        log10_geom_std_dev = log10(1.8),
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
                        log10_geom_std_dev = log10(1.8),
                        mass_fractions = [
                            scipy.stats.uniform(0, 1), # pom
                            scipy.stats.uniform(0, 1), # bc
                        ],
                    ),
                ],
            ),
            gas_concs = (scipy.stats.loguniform(1e5, 1e6) for g in range(3)),
            flux = scipy.stats.loguniform(1e-2*1e-9, 1e1*1e-9),
            relative_humidity = scipy.stats.loguniform(1e-5, 0.99),
            temperature = scipy.stats.uniform(240, 310),
            pressure = p0,
            height = h0,
        )
        self.ensemble = ppe.sample(self.ensemble_spec, self.n)

    def test_create_particle_input(self):
        n_part = 1000
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = 1000,
            n_repeat = 5,
        )
        input = model.create_input(scenario, dt, nstep)

        # timestepping parameters
        self.assertTrue(abs(dt - input.del_t) < 1e-12)
        self.assertTrue(abs(nstep * dt - input.t_max) < 1e-12)

        # aerosol processes
        self.assertFalse(input.do_mosaic)
        self.assertFalse(input.do_nucleation)
        self.assertFalse(input.do_condensation)
        self.assertTrue(input.do_coagulation)

        # FIXME: more stuff vvv

    def test_create_particle_inputs(self):
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        dt = 4.0
        nstep = 100

        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = 1000,
            n_repeat = 5,
        )
        inputs = model.create_inputs(self.ensemble, dt, nstep)
        for i, input in enumerate(inputs):
            scenario = self.ensemble.member(i)

            # timestepping parameters
            self.assertTrue(abs(dt - input.del_t) < 1e-12)
            self.assertTrue(abs(nstep * dt - input.t_max) < 1e-12)

            # aerosol processes
            self.assertFalse(input.do_mosaic)
            self.assertFalse(input.do_nucleation)
            self.assertFalse(input.do_condensation)
            self.assertTrue(input.do_coagulation)

            # FIXME: more stuff vvv

    def test_write_input_files(self):
        n_part = 1000
        processes = aerosol.AerosolProcesses(
            aging = True,
            coagulation = True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = 1000,
            n_repeat = 5,
        )
        input = model.create_input(scenario, dt, nstep)
        temp_dir = tempfile.TemporaryDirectory()
        model.write_input_files(input, temp_dir.name, 'partmc')
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, 'partmc.spec')))
        temp_dir.cleanup()

class TestPartMCHelpers(unittest.TestCase):
    def make_model(self):
        return partmc.AerosolModel(
            processes=aerosol.AerosolProcesses(
                aging=True,
                coagulation=True,
            ),
            n_part=1,
        )

    def make_scenario(self, size):
        return Scenario(
            aerosols=(so4,),
            gases=(so2,),
            size=size,
            gas_concs=(1.0e-9,),
            flux=0.0,
            relative_humidity=0.5,
            temperature=290.0,
            pressure=p0,
            height=h0,
        )

    def test_aerosol_model_constructor_validation(self):
        processes = aerosol.AerosolProcesses(
            aging=True,
            coagulation=True,
        )

        with self.assertRaisesRegex(ValueError, "Unsupported run_type"):
            partmc.AerosolModel(processes=processes, run_type="sectional", n_part=1)

        with self.assertRaisesRegex(ValueError, "n_part must be positive"):
            partmc.AerosolModel(processes=processes, n_part=0)

        with self.assertRaisesRegex(ValueError, "n_part must be positive"):
            partmc.AerosolModel(processes=processes, n_part=None)

        with self.assertRaisesRegex(ValueError, "n_repeat must be non-negative"):
            partmc.AerosolModel(processes=processes, n_part=1, n_repeat=-1)

    def test_create_input_rejects_non_positive_dt(self):
        model = self.make_model()
        scenario = self.make_scenario(
            aerosol.AerosolModalSizePopulation(
                modes=(
                    aerosol.AerosolModePopulation(
                        name="aitken",
                        species=(so4,),
                        number=1.0e8,
                        geom_mean_diam=1.0e-7,
                        log10_geom_std_dev=log10(1.6),
                        mass_fractions=(1.0,),
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "dt must be positive"):
            model.create_input(scenario, 0.0, 1)

    def test_create_input_rejects_non_positive_nstep(self):
        model = self.make_model()
        scenario = self.make_scenario(
            aerosol.AerosolModalSizePopulation(
                modes=(
                    aerosol.AerosolModePopulation(
                        name="aitken",
                        species=(so4,),
                        number=1.0e8,
                        geom_mean_diam=1.0e-7,
                        log10_geom_std_dev=log10(1.6),
                        mass_fractions=(1.0,),
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "nstep must be positive"):
            model.create_input(scenario, 1.0, 0)

    def test_create_input_rejects_non_modal_size(self):
        model = self.make_model()
        scenario = self.make_scenario(size=None)

        with self.assertRaisesRegex(TypeError, "Non-modal aerosol particle size state"):
            model.create_input(scenario, 1.0, 1)

    def test_invocation_returns_expected_command(self):
        self.assertEqual(
            self.make_model().invocation("/usr/local/bin/partmc", "scenario_001"),
            "/usr/local/bin/partmc scenario_001.spec",
        )

    def test_write_input_files_rejects_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_dir = os.path.join(temp_dir, "missing")

            with self.assertRaisesRegex(OSError, "Directory not found"):
                self.make_model().write_input_files(None, missing_dir, "partmc")

    def test_get_ncfile_finds_specific_and_latest_outputs(self):
        scenario_name = "scenario"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, scenario_name, "out")
            os.makedirs(output_dir)
            for filename in [
                "scenario_0002_00000001.nc",
                "scenario_0002_00000003.nc",
                "scenario_0001_00000004.nc",
                "scenario_0002_00000003.txt",
            ]:
                open(os.path.join(output_dir, filename), "w").close()

            self.assertEqual(
                partmc.get_ncfile(
                    scenario_name,
                    3,
                    ensemble_output_dir=temp_dir,
                    repeat_num=2,
                ),
                "scenario_0002_00000003.nc",
            )
            self.assertEqual(
                partmc.get_ncfile(
                    scenario_name,
                    -1,
                    ensemble_output_dir=temp_dir,
                    repeat_num=2,
                ),
                "scenario_0002_00000003.nc",
            )

    def test_get_ncfile_raises_when_output_is_missing(self):
        scenario_name = "scenario"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, scenario_name, "out")
            os.makedirs(output_dir)

            with self.assertRaisesRegex(OSError, "No NetCDF output found"):
                partmc.get_ncfile(
                    scenario_name,
                    1,
                    ensemble_output_dir=temp_dir,
                    repeat_num=1,
                )

    def test_write_aero_modes_rejects_unsupported_diam_type(self):
        processes = aerosol.AerosolProcesses(
            aging=True,
            coagulation=True,
        )
        model = partmc.AerosolModel(processes=processes, n_part=1)
        mode = partmc.AeroMode(
            mode_name="bad_mode",
            mass_frac={"SO4": 1.0},
            diam_type="mobility",
            mode_type="log_normal",
            num_conc=1.0e6,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(TypeError, "Unsupported diam_type"):
                model._write_aero_modes(temp_dir, "aero_init", (mode,))


if __name__ == '__main__':
    unittest.main()
