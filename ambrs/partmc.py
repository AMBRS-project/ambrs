"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from dataclasses import dataclass

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .scenario import Scenario
from .ppe import Ensemble
from typing import Dict, Optional

@dataclass
class PartMCAeroData:
    species: str            # name of aerosol species
    density: float          # aerosol species density [kg/m^3]
    ions_in_soln: int       # number of ions in solution [-]
    molecular_weight: float # molecular weight [kg/mol]
    kappa: float            # "kappa" [-]

@dataclass
class PartMCAeroDist:
    mode_name: str              # name of aerosol mode
    mass_frac: Dict[str, float] # mapping of modal species names to mass fractions
    diam_type: str              # type of diameter specified (e.g. 'geometric')
    mode_type: str              # type of distribution (e.g. 'log_normal')
    num_conc: float             # modal number concentration [#/m^-3]

    # geometric diameter parameters
    geom_mean_diam: Optional[float] = None     # modal geometric mean diameter
    log10_geom_std_dev: Optional[float] = None # log_10 of geometric std dev of diameter

# time series data types
type ScalarTimeSeries tuple[tuple[float, float], ...] # tuple of (t, value) pairs
type CompositeTimeSeries tuple[tuple[float, dict], ...] # tuple of (t, dict) pairs

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
    gas_init: tuple[float, ...] # tuple of initial gas concentrations (in same
                                # order as gas_data)

    aerosol_data: tuple[PartMCAeroData, ...] # tuple of aerosol data
    do_fractal: bool                         # whether to do fractal treatment
    aerosol_init: tuple[PartMCAeroDist, ...] # aerosol initial condition file

    temp_profile: ScalarTimeSeries       # temperature time series ((t1, T1), (t2, T2), ...)
    pressure_profile: ScalarTimeSeries   # pressure time series
    height_profile: ScalarTimeSeries     # height profile file
    gas_emissions: CompositeTimeSeries   # gas emissions time series
    gas_background: CompositeTimeSeries  # background gas concentration time series
    aero_emissions: CompositeTimeSeries  # aerosol emissions file
    aero_background: CompositeTimeSeries # aerosol background file

    rel_humidity: float         # initial relative humidity [-]
    latitude: float             # latitude [degrees, -90 to 90]
    longitude: float            # longitude [degrees, -180 to 180]
    altitude: float             # altitude [m]
    start_time: int             # start time [s since 00:00 UTC]
    start_day: int              # start day of year [days, UTC]

    do_coagulation: bool        # whether to do coagulation
    coag_kernel: str            # coagulation kernel name
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

    def write_files(self, prefix):
        """input.write_files(prefix) -> writes a set of PartMC box model input
files, with a main input <prefix>.spec file"""
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


        with open(prefix + '.spec', 'w') as f:
            f.write(spec_content)

        # write auxiliary data files

def mam4_input_(processes: AerosolProcesses,
                scenario: Scenario,
                dt: float,
                nstep: int) -> MAM4Input:
    if not isinstance(scenario.size, AerosolModalSizeState):
        raise TypeError('Non-modal aerosol particle size state cannot be used to create MAM4 input!')
    if len(scenario.size.modes) != 4:
        raise TypeError(f'{len(scenario.size.mode)}-mode aerosol particle size state cannot be used to create MAM4 input!')
    return MAM4Input(
        mam_dt = dt,
        mam_nstep = nstep,

        mdo_gaschem = False,
        mdo_gasaerexch = False,
        mdo_rename = False,
        mdo_newnuc = processes.nucleation,
        mdo_coag = processes.coagulation,

        temp = scenario.temperature,
        press = scenario.pressure,
        RH_CLEA = scenario.relative_humidity,

        numc1 = scenario.size.modes[0].number,
        numc2 = scenario.size.modes[1].number,
        numc3 = scenario.size.modes[2].number,
        numc4 = scenario.size.modes[3].number,

        mfso41 = scenario.size.modes[0].mass_fraction("so4"),
        mfpom1 = scenario.size.modes[0].mass_fraction("pom"),
        mfsoa1 = scenario.size.modes[0].mass_fraction("soa"),
        mfbc1  = scenario.size.modes[0].mass_fraction("bc"),
        mfdst1 = scenario.size.modes[0].mass_fraction("dst"),
        mfncl1 = scenario.size.modes[0].mass_fraction("ncl"),

        mfso42 = scenario.size.modes[1].mass_fraction("so4"),
        mfsoa2 = scenario.size.modes[1].mass_fraction("soa"),
        mfncl2 = scenario.size.modes[1].mass_fraction("ncl"),

        mfdst3 = scenario.size.modes[2].mass_fraction("dst"),
        mfncl3 = scenario.size.modes[2].mass_fraction("ncl"),
        mfso43 = scenario.size.modes[2].mass_fraction("so4"),
        mfbc3  = scenario.size.modes[2].mass_fraction("bc"),
        mfpom3 = scenario.size.modes[2].mass_fraction("pom"),
        mfsoa3 = scenario.size.modes[2].mass_fraction("soa"),

        mfpom4 = scenario.size.modes[3].mass_fraction("pom"),
        mfbc4  = scenario.size.modes[3].mass_fraction("bc"),

        # FIXME: what to do about gases?
        qso2 = 0,
        qh2so4 = 0,
        qsoag = 0,
     )

def create_mam4_input(processes: AerosolProcesses,
                      scenario: Scenario,
                      dt: float,
                      nstep: int) -> MAM4Input:
    """create_mam4_input(processes, scenario, dt, nstep) -> MAM4Input object
that can create a namelist input file for a MAM4 box model simulation

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
    return mam4_input_(processes, scenario, dt, nstep)

def create_mam4_inputs(processes: AerosolProcesses,
                       ensemble: Ensemble,
                       dt: float,
                       nstep: int) -> list[MAM4Input]:
    """create_mam4_inputs(processes, ensemble, dt, nstep) -> list of MAM4Input
objects that can create namelist input files for MAM4 box model simulations

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
        inputs.append(mam4_input_(processes, scenario, dt, nstep))
    return inputs
