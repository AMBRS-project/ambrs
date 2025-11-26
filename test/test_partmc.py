# unit tests for the ambrs.partmc package

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.scenario import Scenario
import ambrs.partmc as partmc
# import ambrs.partmc_dq_merge_start as partmc

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

if __name__ == '__main__':
    unittest.main()
