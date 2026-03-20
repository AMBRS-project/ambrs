# unit tests for the ambrs.camp package

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.ppe as ppe
from ambrs.scenario import Scenario
import ambrs.partmc as partmc
from ambrs.camp import CAMP, activity_coefficient

from math import log10
import numpy as np
import os
import scipy.stats
from scipy.optimize import fsolve
import tempfile
import unittest

# aerosol and gas species
gases = [
    {
        "name" : "SOAG",
        "molar_mass" : 12.011,
    },
    {
        "name" : "SO2",
        "molar_mass" : 64.0648
    },
    {
        "name" : "H2SO4",
        "molar_mass" : 98.0784,
    }
]
aerosols = [
    {
        "name" : "SO4",
        "molar_mass" : 115.107340,
        "density" : 1770.0000,
        "hygroscopicity" : 0.507
    },
    {
        "name" : "POM",
        "molar_mass" : 12.011,
        "density" : 1000.0000,
        "hygroscopicity" : 0.010
    },
    {
        "name" : "SOA",
        "molar_mass" : 12.011,
        "density" : 1000.0000,
        "hygroscopicity" : 0.140
    },
    {
        "name" : "BC",
        "molar_mass" : 12.011,
        "density" : 1700.0000,
        "hygroscopicity" : 1.0e-10
    },
    {
        "name" : "DST",
        "molar_mass" : 135.064039,
        "density" : 2600.0000,
        "hygroscopicity" : 0.068
    },
    {
        "name" : "NCL",
        "molar_mass" : 58.442468,
        "density" : 1900.0000,
        "hygroscopicity" : 1.160
    },
    {
        "name" : "MOM",
        "molar_mass" : 250092.672000,
        "density" : 1601.0000,
        "hygroscopicity" : 0.100
    }
]
gases = [gas.GasSpecies(**g) for g in gases]
aerosols = [aerosol.AerosolSpecies(**p) for p in aerosols]

# reference pressure and height
p0 = 101325 # [Pa]
h0 = 500    # [m]

class TestCAMPInput(unittest.TestCase):
    """Unit tests for ambrs.camp"""

    def setUp(self):
        self.n = 100
        self.ensemble_spec = ppe.EnsembleSpecification(
            name = 'partmc_ensemble',
            aerosols = tuple(aerosols),
            gases = tuple(gases),
            size = aerosol.AerosolModalSizeDistribution(
                modes = [
                    aerosol.AerosolModeDistribution(
                        name = "accumulation",
                        species = aerosols,
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-7, 1.1e-7),
                        log10_geom_std_dev = log10(1.6),
                        mass_fractions = [1. if p.name in ['SO4'] else 0. for p in aerosols],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "aitken",
                        species = aerosols,
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(0.5e-8, 3e-8),
                        log10_geom_std_dev = log10(1.6),
                        mass_fractions = [1. if p.name in ['SO4'] else 0. for p in aerosols],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "coarse",
                        species = aerosols,
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-6, 2e-6),
                        log10_geom_std_dev = log10(1.8),
                        mass_fractions = [1. if p.name in ['SO4'] else 0. for p in aerosols],
                    ),
                    aerosol.AerosolModeDistribution(
                        name = "primary carbon",
                        species = aerosols,
                        number = scipy.stats.loguniform(3e7, 2e12),
                        geom_mean_diam = scipy.stats.loguniform(1e-8, 6e-8),
                        log10_geom_std_dev = log10(1.8),
                        mass_fractions = [1. if p.name in ['POM'] else 0. for p in aerosols],
                    ),
                ],
            ),
            gas_concs = (scipy.stats.loguniform(1e5, 1e6) for g in gases),
            flux = scipy.stats.loguniform(1e-2*1e-9, 1e1*1e-9),
            relative_humidity = scipy.stats.loguniform(1e-5, 0.99),
            temperature = scipy.stats.uniform(240, 310),
            pressure = p0,
            height = h0,
        )
        self.ensemble = ppe.sample(self.ensemble_spec, self.n)

        N_star = lambda temperature, **kwargs: fsolve(
            func=activity_coefficient, x0=1.0, args=(temperature,)
        ).item()
        reactions = [
            {
                'type': 'SIMPOL_PHASE_TRANSFER',
                'gas-phase species': 'H2SO4',
                'aerosol phase': 'mixed',
                'aerosol-phase species': 'SO4',
                'B': [0.0, -100.0, 0.0, 0.0],
                'N star': N_star
            }
        ]
        self.camp = CAMP(
            ppe=self.ensemble,
            aero_rep_type='AERO_REP_SINGLE_PARTICLE',
            maximum_computational_particles=1100,
            diffusion_coeff={'H2SO4': lambda temperature, pressure, **kwargs: 0.557e-4 * (temperature**1.75) / pressure},
            reactions=reactions
        )

    def test_create_particle_input(self):
        n_part = 1000
        processes = aerosol.AerosolProcesses(
            do_camp_chem=True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = n_part,
            n_repeat = 5,
            camp_config=self.camp,
        )
        input = model.create_input(scenario, dt, nstep)

        # timestepping parameters
        self.assertTrue(abs(dt - input.del_t) < 1e-12)
        self.assertTrue(abs(nstep * dt - input.t_max) < 1e-12)

        # aerosol processes
        self.assertFalse(input.do_mosaic)
        self.assertFalse(input.do_nucleation)
        self.assertFalse(input.do_condensation)
        self.assertFalse(input.do_coagulation)
        self.assertTrue(input.do_camp_chem)

    def test_create_particle_inputs(self):
        processes = aerosol.AerosolProcesses(
            do_camp_chem=True,
        )
        dt = 4.0
        nstep = 100

        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = 1000,
            n_repeat = 5,
            camp_config=self.camp,
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
            self.assertFalse(input.do_coagulation)
            self.assertTrue(input.do_camp_chem)

    def test_write_input_files(self):
        n_part = 1000
        processes = aerosol.AerosolProcesses(
            do_camp_chem=True,
        )
        scenario = self.ensemble.member(0)
        dt = 4.0
        nstep = 100
        model = partmc.AerosolModel(
            processes = processes,
            run_type = 'particle',
            n_part = n_part,
            n_repeat = 5,
            camp_config=self.camp,
        )
        input = model.create_input(scenario, dt, nstep)
        temp_dir = tempfile.TemporaryDirectory()
        model.write_input_files(input, temp_dir.name, 'partmc')
        self.assertTrue(os.path.exists(os.path.join(temp_dir.name, 'camp.json')))
        temp_dir.cleanup()

if __name__ == '__main__':
    unittest.main()
