"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from dataclasses import dataclass

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .scenario import Scenario
from .ppe import Ensemble
from typing import Dict, Optional

import os.path

@dataclass
class PartMCAeroData:
    species: str            # name of aerosol species
    density: float          # aerosol species density [kg/m^3]
    ions_in_soln: int       # number of ions in solution [-]
    molecular_weight: float # molecular weight [kg/mol]
    kappa: float            # "kappa" [-]

@dataclass
class PartMCAeroMode:
    mode_name: str              # name of aerosol mode
    mass_frac: Dict[str, float] # mapping of modal species names to mass fractions
    diam_type: str              # type of diameter specified (e.g. 'geometric')
    mode_type: str              # type of distribution (e.g. 'log_normal')
    num_conc: float             # modal number concentration [#/m^-3]

    # geometric diameter parameters
    geom_mean_diam: Optional[float] = None     # modal geometric mean diameter
    log10_geom_std_dev: Optional[float] = None # log_10 of geometric std dev of diameter

# time series data types
type ScalarTimeSeries = list[tuple[float, float], ...] # list of (t, value) pairs
type DictTimeSeries = list[tuple[float, dict], ...] # list of (t, dict) pairs
type AerosolModeTimeSeries = list[tuple[float, PartMCAeroMode], ...] # list of (t, mode) pairs

@dataclass
class PartMCInput:
    """PartMCInput -- represents input for a single PartMC box model simulation"""
    # all fields here are named identically to their respective parameters
    # in the .spec scenario file for the PartMC box model

    run_type: str               # particle, analytic, sectional
    output_prefix: str          # prefix of output files

    restart:  bool              # whether to restart from saved state
    do_select_weighting: bool   # whether to select weighting explicitly

    t_max: float                # total simulation time [s]
    del_t: float                # timestep [s]
    t_output: float             # output interval (0 disables) [s]
    t_progress: float           # progress printing interval (0 disables) [s]

    do_camp_chem: bool          # whether to use CAMP for chemistry

    gas_data: tuple[str, ...]   # tuple of gas species names
    gas_init: dict[str, float]  # dict of initial gas concentrations (keys are
                                # gas species names)

    aerosol_data: tuple[PartMCAeroData, ...] # tuple of aerosol data
    do_fractal: bool                         # whether to do fractal treatment
    aerosol_init: tuple[PartMCAeroMode, ...] # aerosol modal initial condition file

    temp_profile: ScalarTimeSeries       # temperature time series ((t1, T1), (t2, T2), ...)
    pressure_profile: ScalarTimeSeries   # pressure time series
    height_profile: ScalarTimeSeries     # height profile file
    gas_emissions: DictTimeSeries   # gas emissions time series
    gas_background: DictTimeSeries  # background gas concentration time series
    aero_emissions: AerosolModeTimeSeries  # aerosol emissions file
    aero_background: AerosolModeTimeSeries # aerosol background file

    rel_humidity: float         # initial relative humidity [-]
    latitude: float             # latitude [degrees, -90 to 90]
    longitude: float            # longitude [degrees, -180 to 180]
    altitude: float             # altitude [m]
    start_time: int             # start time [s since 00:00 UTC]
    start_day: int              # start day of year [days, UTC]

    do_coagulation: bool        # whether to do coagulation
    do_condensation: bool       # whether to do condensation
    do_mosaic: bool             # whether to do MOSAIC
    do_optical: bool            # whether to compute optical props
    do_nucleation: bool         # whether to do nucleation

    rand_init: int              # random initialization (0 to use time)
    allow_doubling: bool        # whether to allow doubling
    allow_halving: bool         # whether to allow halving
    record_removals: bool       # whether to record particle removals
    do_parallel: bool           # whether to run in parallel

    # particle-specific fields
    n_repeat: Optional[int] = None # number of Monte Carlo repeats
    n_part:   Optional[int] = None # number of particles
    loss_function: Optional[str] = None # loss function specification

    # TODO: analytic-specific fields??
    # TODO: sectional-specific fields??

    # TODO: fractal-specific fields??

    # process-specific fields`
    coag_kernel: Optional[str] = None # coagulation kernel name

    def write_aero_modes_(self, prefix, modes):
        dist_file = f'{prefix}_dist.dat'
        with open(os.path.join(dir, dist_file)) as f:
            for i, mode in enumerate(modes):
                f.write(f'mode_name {mode.mode_name}\n')
                f.write(f'mass_frac {prefix}_comp_{i+1}.dat\n')
                f.write(f'diam_type {mode.diam_type}\n')
                f.write(f'mode_type {mode.mode_type}\n')
                f.write(f'num_conc {mode.num_conc}\n')
                if mode.diam_type == 'geometric':
                    f.write(f'geom_mean_diam {mode.geom_mean_diam}\n')
                    f.write(f'log10_geom_std_dev {mode.log10_geom_std_dev}\n')
                else:
                    raise TypeError(f'Unsupported diam_type for {mode.mode_name} mode: {mode.diam_type}')
                f.write('\n')
        for i, mode in enumerate(modes):
            with open(os.path.join(dir, f'{prefix}_comp_{i+1}.dat')) as f:
                f.write('#\tproportion\n')
                for species, mass_frac in mode.mass_frac.items():
                    f.write(f'{species}\t{mass_frac}\n')

    def write_files(self, dir, prefix):
        """input.write_files(prefix) -> writes a set of PartMC box model input
files, with a main input <prefix>.spec file"""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')

        # write the main (.spec) file
        spec_content = f'run_type {self.run_type}\noutput_prefix {self.output_prefix}\n'

        if self.run_type == 'particle':
            spec_content += f'n_repeat {self.n_repeat}\nn_part {self.n_part}\n'
        if self.restart:
            spec_content += 'restart yes\n'
        else:
            spec_content += 'restart no\n'
        if self.do_select_weighting:
            spec_content += 'do_select_weighting yes\n'
        else:
            spec_content += 'do_select_weighting no\n'

        spec_content += f'\nt_max {self.t_max}\ndel_t {self.del_t}\nt_output {self.t_output}\nt_progress {self.t_progress}\n\n'

        if self.do_camp_chem:
            spec_content += 'do_camp_chem yes\n'
        else:
            spec_content += 'do_camp_chem no\n'

        spec_content += '\ngas_data gas_data.dat\ngas_init gas_init.dat\n\n'

        spec_content += 'aerosol_data aerosol_data.dat\n'
        if self.do_fractal:
            spec_content += 'do_fractal yes\n'
        else:
            spec_content += 'do_fractal no\n'
        spec_content += 'aerosol_init aero_init_dist.dat\n\n'

        spec_content += 'temp_profile temp.dat\npressure_profile pres.dat\nheight_profile height.dat\n'
        spec_content += 'gas_emissions gas_emit.dat\ngas_background gas_back.dat\n'
        spec_content += 'aero_emissions aero_emit.dat\naero_background aero_back.dat\n'

        if self.loss_function:
            spec_content += f'loss_function {self.loss_function}\n'
        else:
            spec_content += 'loss_function none\n'

        spec_content += f'rel_humidity {self.rel_humidity}\n'
        spec_content += f'latitude {self.latitude}\n'
        spec_content += f'longitude {self.longitude}\n'
        spec_content += f'altitude {self.altitude}\n'
        spec_content += f'start_time {self.start_time}\n'
        spec_content += f'start_day {self.start_day}\n'

        if self.do_coagulation:
            spec_content += f'do_coagulation yes\ncoag_kernel {self.coag_kernel}\n'
        else:
            spec_content += 'do_coagulation no\n'

        if self.do_condensation:
            spec_content += 'do_condensation yes\n'
        else:
            spec_content += 'do_condensation no\n'

        if self.do_mosaic:
            spec_content += 'do_mosaic yes\n'
        else:
            spec_content += 'do_mosaic no\n'

        if self.do_optical:
            spec_content += 'do_optical yes\n'
        else:
            spec_content += 'do_optical no\n'

        if self.do_nucleation:
            spec_content += 'do_nucleation yes\n'
        else:
            spec_content += 'do_nucleation no\n'

        spec_content += f'rand_init {self.rand_init}\n'

        if self.allow_doubling:
            spec_content += 'allow_doubling yes\n'
        else:
            spec_content += 'allow_doubling no\n'

        if self.allow_halving:
            spec_content += 'allow_halving yes\n'
        else:
            spec_content += 'allow_halving no\n'

        if self.record_removals:
            spec_content += 'record_removals yes\n'
        else:
            spec_content += 'record_removals no\n'

        if self.do_parallel:
            spec_content += 'do_parallel yes\n'
        else:
            spec_content += 'do_parallel no\n'

        with open(os.path.join(dir, prefix + '.spec'), 'w') as f:
            f.write(spec_content)

        # write auxiliary data files

        # gas_data.dat, gas_init.dat
        with open(os.path.join(dir, 'gas_data.dat')) as f:
            f.write('# list of gas species\n')
            for gas in self.gas_data:
                f.write(f'{gas}\n')
        with open(os.path.join(dir, 'gas_init.dat')) as f:
            f.write('# species\tinitial concentration (ppb)\n')
            for species, conc in self.gas_init.items():
                f.write(f'{species}\t{conc}\n')

        # aero_data.dat, aero_init_dist.dat, aero_init_comp.dat
        with open(os.path.join(dir, 'aero_data.dat')) as f:
            f.write('#\tdens (kg/m^3)\tions in soln (1)\tmolec wght (kg/mole)\tkappa (1)\n')
            for aero in self.aerosol_data:
                f.write(f'{aero.species}\t{aero.density}\t{aero.ions_in_soln}\t{aero.molecular_weight}\t{aero.kappa}\n')
        self.write_aero_modes_(f, 'aero_init', self.aerosol_init)

        # temp.dat, pres.dat, height.dat
        with open(os.path.join(dir, 'temp.dat')) as f:
            f.write('# time (s)\n# temp (K)\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.temp_profile]))
            f.write('\t'.join(['temp'] + [time_series[1] for time_series in self.temp_profile]))
        with open(os.path.join(dir, 'pres.dat')) as f:
            f.write('# time (s)\n# pressure (Pa)\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.pressure_profile]))
            f.write('\t'.join(['pressure'] + [time_series[1] for time_series in self.pressure_profile]))
        with open(os.path.join(dir, 'height.dat')) as f:
            f.write('# time (s)\n# height (m)\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.height_profile]))
            f.write('\t'.join(['height'] + [time_series[1] for time_series in self.height_profile]))

        # gas_emit.dat, gas_back.dat
        gas_emission_species = [self.emissions[0].time_series[1].keys()]
        gas_emission_species.remove('rate')
        with open(os.path.join(dir, 'gas_emit.dat')) as f:
            f.write('# time (s)\n# rate = scaling parameter\n# emissions (mol m^{-2} s^{-1})\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.emissions]))
            f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.emissions]))
            f.write('\t'.join([species_name] + [time_series[1][species_name] \
                              for species_name in gas_emission_species \
                              for time_series in self.emissions]))
        gas_background_species = [self.gas_background[0].time_series[1].keys()]
        gas_emission_species.remove('rate')
        with open(os.path.join(dir, 'gas_back.dat')) as f:
            f.write('# time (s)\n# rate (s^{-1})\n# concentrations (ppb)\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.gas_background]))
            f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.gas_background]))
            f.write('\t'.join([species_name] + [time_series[1][species_name] \
                              for species_name in gas_background_species \
                              for time_series in self.gas_background]))

        # aero_emit.dat, aero_emit_dist_*.dat, aero_emit_comp_*.dat
        with open(os.path.join(dir, 'aero_emit.dat')) as f:
            f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.aero_emissions]))
            f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.aero_emissions]))
            f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(self.aero_emissions))]))
        for i, time_series in enumerate(self.aero_emissions):
            self.write_aero_modes_(f'aero_emit_dist_{i+1}', self.aero_emissions)

        # aero_back.dat, aero_back_dist.dat, aero_back_comp.dat
        with open(os.path.join(dir, 'aero_back.dat')) as f:
            f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
            f.write('\t'.join(['time'] + [time_series[0] for time_series in self.aero_background]))
            f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.aero_background]))
            f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(self.aero_background))]))
        for i, time_series in enumerate(self.aero_background):
            self.write_aero_modes_(f'aero_back_dist_{i+1}', self.aero_background)

def partmc_input_(processes: AerosolProcesses,
                  scenario: Scenario,
                  dt: float,
                  nstep: int) -> PartMCInput:
    if not isinstance(scenario.size, AerosolModalSizeState):
        raise TypeError('Non-modal aerosol particle size state cannot be used to create MAM4 input!')
    # FIXME: do this

def create_partmc_input(processes: AerosolProcesses,
                        scenario: Scenario,
                        dt: float,
                        nstep: int) -> PartMCInput:
    """create_partmc_input(processes, scenario, dt, nstep) -> PartMCInput object
that can create a namelist input file for a PartMC box model simulation

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * scenario: an ambrs.Scenario object created by sampling a modal particle
      size distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    return partmc_input_(processes, scenario, dt, nstep)

def create_partmc_inputs(processes: AerosolProcesses,
                         ensemble: Ensemble,
                         dt: float,
                         nstep: int) -> list[PartMCInput]:
    """create_mam4_inputs(processes, ensemble, dt, nstep) -> list of PartMCInput
objects that can create namelist input files for PartMC box model simulations

Parameters:
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * ensemble: a ppe.Ensemble object created by sampling a modal particle size
      distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if not isinstance(ensemble.size, AerosolModalSizePopulation):
        raise TypeError("ensemble must have a modal size distribution!")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    inputs = []
    for scenario in ensemble:
        inputs.append(partmc_input_(processes, scenario, dt, nstep))
    return inputs
