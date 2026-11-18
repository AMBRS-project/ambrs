# regression tests: sampled emissions and backgrounds must reach PartMC's input files
#
# AMBRS v2026.0 had two problems here. The aerosol time-series writer read
# event.modes, but AerosolEmissions carries event.size, so any aerosol emission or
# background raised AttributeError. And create_input did not pass the scenario's gas
# background on, so PartMC silently got a zero background.

import os
import tempfile
import unittest

import ambrs.aerosol as aerosol
import ambrs.gas as gas
import ambrs.partmc as partmc
from ambrs.emissions import AerosolEmissions
from ambrs.scenario import Scenario

so4 = aerosol.AerosolSpecies(
    name='SO4',
    molar_mass = 97.071,
    density = 1770,
    hygroscopicity = 0.507,
)
bc = aerosol.AerosolSpecies(
    name='BC',
    molar_mass = 12.01,
    density = 1000,
    hygroscopicity = 0.5,
)
so2 = gas.GasSpecies(
    name='SO2',
    molar_mass = 64.07,
)


def read_series(path):
    """{row name: [values]} for a PartMC time-series file, ignoring comment lines."""
    rows = {}
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                name, *values = line.rstrip('\n').split('\t')
                rows[name] = values
    return rows


class TestPartMCEmissionsAndBackground(unittest.TestCase):

    def make_state(self, name, diam):
        return aerosol.AerosolModalSizeState(
            modes = (
                aerosol.AerosolModeState(
                    name = name,
                    species = (so4, bc),
                    number = 1.0e8,
                    geom_mean_diam = diam,
                    log10_geom_std_dev = 0.2,
                    mass_fractions = (0.75, 0.25),
                ),
            ),
        )

    def make_scenario(self):
        return Scenario(
            aerosols = (so4, bc),
            gases = (so2,),
            size = self.make_state('accumulation', 1.0e-7),
            gas_concs = (1.0e-9,),
            flux = 0.0,
            relative_humidity = 0.5,
            temperature = 290.0,
            pressure = 101325.0,
            height = 500.0,
            gas_emissions = [(0.0, {'SO2': 2.0e-9, 'rate': 1.0})],
            gas_background = [(0.0, {'SO2': 1.5e-9, 'rate': 0.25})],
            aerosol_emissions = [
                AerosolEmissions(
                    time = 0.0,
                    rate = 2.0,
                    size = self.make_state('emitted', 5.0e-8),
                ),
            ],
            aerosol_background = [
                AerosolEmissions(
                    time = 0.0,
                    rate = 0.5,
                    size = self.make_state('background', 2.0e-7),
                ),
            ],
        )

    def write_inputs(self, directory):
        model = partmc.AerosolModel(
            processes = aerosol.AerosolProcesses(aging = True, coagulation = True),
            run_type = 'particle',
            n_part = 10,
            n_repeat = 1,
        )
        model_input = model.create_input(self.make_scenario(), 60.0, 2)
        model.write_input_files(model_input, directory, 'partmc')

    def test_gas_background_reaches_the_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_inputs(directory)
            rows = read_series(os.path.join(directory, 'gas_back.dat'))
        self.assertEqual(float(rows['rate'][0]), 0.25)
        self.assertEqual(float(rows['SO2'][0]), 1.5e-9)

    def test_aerosol_background_reaches_the_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_inputs(directory)
            rows = read_series(os.path.join(directory, 'aero_back.dat'))
            self.assertEqual(float(rows['rate'][0]), 0.5)
            self.assertEqual(rows['dist'], ['aero_back_dist_1_dist.dat'])
            with open(os.path.join(directory, 'aero_back_dist_1_dist.dat')) as f:
                self.assertIn('mode_name background', f.read())

    def test_aerosol_emissions_reach_the_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_inputs(directory)
            rows = read_series(os.path.join(directory, 'aero_emit.dat'))
            self.assertEqual(float(rows['rate'][0]), 2.0)
            self.assertEqual(rows['dist'], ['aero_emit_dist_1_dist.dat'])
            with open(os.path.join(directory, 'aero_emit_dist_1_dist.dat')) as f:
                self.assertIn('mode_name emitted', f.read())


if __name__ == '__main__':
    unittest.main()
