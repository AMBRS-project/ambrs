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
class PartMCAeroSizeDistribution:
    diam: tuple[float, ...] # diameters of bin edges [m]
    num_conc: tuple[float, ...] # total number concentrations of particles in bins [m^{-3}]

@dataclass
class PartMCAeroMode:
    mode_name: str  # name of aerosol mode
    mass_frac: Dict[str, float | tuple[float, float]]
                    # mapping of modal species names to mass fractions (and possibly also std deviations)
    diam_type: str  # type of diameter specified ('geometric', 'mobility')
    mode_type: str  # type of distribution ('log_normal', 'exp', 'mono', 'sampled')

    #----------------------
    # diam_type parameters
    #----------------------

    # mobility
    temp: Optional[float] = None # temperature at which mobility diameters were measured [K]
    pressure: Optional[float] = None # pressure at which mobility diameters were measured [Pa]

    #----------------------
    # mode_type parameters
    #----------------------

    # log_normal, exp, mono
    num_conc: float = None                     # modal number concentration [#/m^-3]

    # log_normal
    geom_mean_diam: Optional[float] = None     # modal geometric mean diameter
    log10_geom_std_dev: Optional[float] = None # log_10 of geometric std dev of diameter

    # exp
    diam_at_mean_vol: Optional[float] = None   # the diameter corresponding to the mean volume [m]

    # mono
    radius: Optional[float] = None             # the radius R0 for which 2*R0 = D0 [m]

    # sampled
    size_dist: Optional[PartMCAeroSizeDistribution] = None # aerosol size distribution


# time series data types
type ScalarTimeSeries = list[tuple[float, float], ...] # list of (t, value) pairs
type DictTimeSeries = list[tuple[float, dict], ...] # list of (t, dict) pairs
type AerosolModeTimeSeries = list[tuple[float, PartMCAeroMode], ...] # list of (t, mode) pairs

@dataclass
class Input:
    """ambrs.partmc.Input -- represents input for a single PartMC box model simulation"""
    # all fields here are named identically to their respective parameters
    # in the .spec scenario file for the PartMC box model

    run_type: str               # particle, analytic, sectional

    restart:  bool              # whether to restart from saved state
    do_select_weighting: bool   # whether to select weighting explicitly

    t_max: float                # total simulation time [s]
    del_t: float                # timestep [s]
    t_output: float             # output interval (0 disables) [s]
    t_progress: float           # progress printing interval (0 disables) [s]

    do_camp_chem: bool          # whether to use CAMP for chemistry

    gas_data: tuple[str, ...]   # tuple of gas species names
    gas_init: tuple[float, ...] # initial gas concentrations (ordered like gas_data)

    aerosol_data: tuple[PartMCAeroData, ...] # tuple of aerosol data
    do_fractal: bool                         # whether todo fractal treatment
    aerosol_init: tuple[PartMCAeroMode, ...] # aerosol modal initial condition file

    temp_profile: ScalarTimeSeries       # temperature time series ((t1, T1), (t2, T2), ...)
    pressure_profile: ScalarTimeSeries   # pressure time series
    height_profile: ScalarTimeSeries     # height profile file

    rel_humidity: float         # initial relative humidity [-]
    latitude: float             # latitude [degrees, -90 to 90]
    longitude: float            # longitude [degrees, -180 to 180]
    altitude: float             # altitude [m]
    start_time: int             # start time [s since 00:00 UTC]
    start_day: int              # start day of year [days, UTC]

    do_coagulation: bool        # whether to do coagulation
    do_condensation: bool       # whether to do condensation
    do_mosaic: bool             # whether to use MOSAIC
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

    # emissions fields
    gas_emissions: Optional[DictTimeSeries] = None # gas emissions time series
    gas_background: Optional[DictTimeSeries] = None  # background gas concentration time series
    aero_emissions: Optional[AerosolModeTimeSeries] = None  # aerosol emissions time series
    aero_background: Optional[AerosolModeTimeSeries] = None # aerosol background time series

    def _write_aero_modes(self, dir, prefix, modes):
        dist_file = f'{prefix}_dist.dat'
        with open(os.path.join(dir, dist_file), 'w') as f:
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

    def invocation(self, exe: str, prefix: str) -> str:
        """input.invocation(exe, prefix) -> a string defining the command invoking
the input with the given executable and input prefix, assuming that the current
working directory contains any needed input files."""
        return f'{exe} {prefix}.spec'

    def write_files(self, dir: str, prefix: str):
        """input.write_files(dir, prefix) -> writes a set of PartMC box model
input files to the given directory with the given prefix for the main .spec
file"""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')

        # write the main (.spec) file
        spec_content = f'run_type {self.run_type}\noutput_prefix {prefix}\n'

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
        with open(os.path.join(dir, 'gas_data.dat'), 'w') as f:
            f.write('# list of gas species\n')
            for gas in self.gas_data:
                f.write(f'{gas}\n')
        with open(os.path.join(dir, 'gas_init.dat'), 'w') as f:
            f.write('# species\tinitial concentration (ppb)\n')
            for g in range(len(self.gas_data)):
                f.write(f'{self.gas_init[g]}\t{self.gas_data[g]}\n')

        # aero_data.dat, aero_init_dist.dat, aero_init_comp.dat
        with open(os.path.join(dir, 'aero_data.dat'), 'w') as f:
            f.write('#\tdens (kg/m^3)\tions in soln (1)\tmolec wght (kg/mole)\tkappa (1)\n')
            for aero in self.aerosol_data:
                f.write(f'{aero.species}\t{aero.density}\t{aero.ions_in_soln}\t{aero.molecular_weight}\t{aero.kappa}\n')
        self._write_aero_modes(dir, 'aero_init', self.aerosol_init)

        # temp.dat, pres.dat, height.dat
        with open(os.path.join(dir, 'temp.dat'), 'w') as f:
            f.write('# time (s)\n# temp (K)\n')
            f.write('\t'.join(['time'] + [str(time_series[0]) for time_series in self.temp_profile]))
            f.write('\t'.join(['temp'] + [str(time_series[1]) for time_series in self.temp_profile]))
        with open(os.path.join(dir, 'pres.dat'), 'w') as f:
            f.write('# time (s)\n# pressure (Pa)\n')
            f.write('\t'.join(['time'] + [str(time_series[0]) for time_series in self.pressure_profile]))
            f.write('\t'.join(['pressure'] + [str(time_series[1]) for time_series in self.pressure_profile]))
        with open(os.path.join(dir, 'height.dat'), 'w') as f:
            f.write('# time (s)\n# height (m)\n')
            f.write('\t'.join(['time'] + [str(time_series[0]) for time_series in self.height_profile]))
            f.write('\t'.join(['height'] + [str(time_series[1]) for time_series in self.height_profile]))

        # gas_emit.dat, gas_back.dat
        if self.gas_emissions:
            gas_emission_species = [self.gas_emissions[0].time_series[1].keys()]
            gas_emission_species.remove('rate')
            with open(os.path.join(dir, 'gas_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate = scaling parameter\n# emissions (mol m^{-2} s^{-1})\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in self.emissions]))
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.emissions]))
                f.write('\t'.join([species_name] + [time_series[1][species_name] \
                                  for species_name in gas_emission_species \
                                  for time_series in self.emissions]))
        if self.gas_background:
            gas_background_species = [self.gas_background[0].time_series[1].keys()]
            gas_background_species.remove('rate')
            with open(os.path.join(dir, 'gas_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# concentrations (ppb)\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in self.gas_background]))
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.gas_background]))
                f.write('\t'.join([species_name] + [time_series[1][species_name] \
                                  for species_name in gas_background_species \
                                  for time_series in self.gas_background]))

        # aero_emit.dat, aero_emit_dist_*.dat, aero_emit_comp_*.dat
        if self.aero_emissions:
            with open(os.path.join(dir, 'aero_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in self.aero_emissions]))
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.aero_emissions]))
                f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(self.aero_emissions))]))
            for i, time_series in enumerate(self.aero_emissions):
                self._write_aero_modes(dir, f'aero_emit_dist_{i+1}', self.aero_emissions)

        # aero_back.dat, aero_back_dist.dat, aero_back_comp.dat
        if self.aero_background:
            with open(os.path.join(dir, 'aero_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in self.aero_background]))
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in self.aero_background]))
                f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(self.aero_background))]))
            for i, time_series in enumerate(self.aero_background):
                self._write_aero_modes(dir, f'aero_back_dist_{i+1}', self.aero_background)

def _particle_input(n_part: int,
                    processes: AerosolProcesses,
                    scenario: Scenario,
                    dt: float,
                    nstep: int) -> Input:
    if not isinstance(scenario.size, AerosolModalSizeState):
        raise TypeError('Non-modal aerosol particle size state cannot be used to create PartMC input!')
    aero_data = []
    aero_init = []
    return Input(
        run_type = 'particle',
        n_part = n_part,

        restart = False,
        do_select_weighting = False,

        t_max = nstep * dt,
        del_t = dt,
        t_output = 0,
        t_progress = 0,

        do_camp_chem = False,

        gas_data = tuple([gas.name for gas in scenario.gases]),
        gas_init = tuple([gas_conc for gas_conc in scenario.gas_concs]),

        aerosol_data = tuple(aero_data),
        do_fractal = False,
        aerosol_init = tuple(aero_init),

        temp_profile = [(0, scenario.temperature)],
        pressure_profile = [(0, scenario.pressure)],
        height_profile = [(0, scenario.height)],

        rel_humidity = scenario.relative_humidity,
        latitude = 0.0,   # FIXME:
        longitude = 0.0,  # FIXME:
        altitude = 10000, # FIXME:
        start_time = 0,   # FIXME:
        start_day = 0,   # FIXME:

        do_coagulation = processes.coagulation,
        do_condensation = processes.condensation,
        do_mosaic = False,
        do_optical = processes.optics,
        do_nucleation = processes.nucleation,

        rand_init = 0, # FIXME: uses time to initialize random seed
        allow_doubling = False,
        allow_halving = False,
        record_removals = False,
        do_parallel = False,
    )

def create_particle_input(n_part: int,
                          processes: AerosolProcesses,
                          scenario: Scenario,
                          dt: float,
                          nstep: int) -> Input:
    """ambrs.partmc.create_particle_input(n_part, scenario, dt, nstep) ->
ambrs.partmc.Input object that can create input files for a particle-based box
model simulation

Parameters:
    * n_part: the number of computational particles for the simulation
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * scenario: an ambrs.Scenario object created by sampling a modal particle
      size distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if n_part <= 0:
        raise ValueError(f"n_part must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    return _particle_input(n_part, processes, scenario, dt, nstep)

def create_particle_inputs(n_part: int,
                           processes: AerosolProcesses,
                           ensemble: Ensemble,
                           dt: float,
                           nstep: int) -> list[Input]:
    """ambrs.partmc.create_particle_inputs(n_part, ensemble, dt, nstep) -> list
of ambrs.partmc.Input objects that can create input files for particle-based
box model simulations

Parameters:
    * n_part: the number of computational particles for each simulation
    * processes: an ambrs.AerosolProcesses object that defines the aerosol
      processes under consideration
    * ensemble: a ppe.Ensemble object created by sampling a modal particle size
      distribution
    * dt: a fixed time step size for simulations
    * nsteps: the number of steps in each simulation"""
    if n_part <= 0:
        raise ValueError(f"n_part must be positive")
    if not isinstance(ensemble.size, AerosolModalSizePopulation):
        raise TypeError("ensemble must have a modal size distribution!")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nstep <= 0:
        raise ValueError("nstep must be positive")
    inputs = []
    for scenario in ensemble:
        inputs.append(_particle_input(n_part, processes, scenario, dt, nstep))
    return inputs
