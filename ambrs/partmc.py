"""ambrs.mam4 -- data types and functions related to the MAM4 box model"""

from .aerosol import AerosolProcesses, AerosolModalSizePopulation, \
                     AerosolModalSizeState
from .aerosol_model import BaseAerosolModel
from .analysis import Output
from .scenario import Scenario
from .ppe import Ensemble
from typing import Dict, Optional

import os
from dataclasses import dataclass

@dataclass
class AeroData:
    species: str            # name of aerosol species
    density: float          # aerosol species density [kg/m^3]
    ions_in_soln: int       # number of ions in solution [-]
    molecular_weight: float # molecular weight [kg/mol]
    kappa: float            # "kappa" [-]

@dataclass 
class AeroSizeDistribution:
    diam: tuple[float, ...] # diameters of bin edges [m]
    num_conc: tuple[float, ...] # total number concentrations of particles in bins [m^{-3}]

@dataclass
class AeroMode:
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
    log10_geom_std_dev: Optional[float] = None # log10 of geometric std dev of diameter

    # exp
    diam_at_mean_vol: Optional[float] = None   # the diameter corresponding to the mean volume [m]

    # mono
    radius: Optional[float] = None             # the radius R0 for which 2*R0 = D0 [m]

    # sampled
    size_dist: Optional[AeroSizeDistribution] = None # aerosol size distribution

# time series data types
type ScalarTimeSeries = list[tuple[float, float], ...] # list of (t, value) pairs
type DictTimeSeries = list[tuple[float, dict], ...] # list of (t, dict) pairs
type AerosolModeTimeSeries = list[tuple[float, AeroMode], ...] # list of (t, mode) pairs

@dataclass
class Input:
    """ambrs.partmc.Input -- an input dataclass for PartMC's box model"""
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

    aerosol_data: tuple[AeroData, ...] # tuple of aerosol data
    do_fractal: bool                         # whether todo fractal treatment
    aerosol_init: tuple[AeroMode, ...] # aerosol modal initial condition file

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

    # weighting fields
    weight_type: str = ''       # type of weighting to use (power, power_source)
    weighting_exponent: int = 0 # exponent to use in weighting curve (-3 to 0)

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

class AerosolModel(BaseAerosolModel):
    def __init__(self,
                 name: str,
                 processes: AerosolProcesses,
                 run_type = 'particle',
                 n_part = None,
                 n_repeat = 0):
        BaseAerosolModel.__init__(self, name, processes)
        if run_type not in ['particle']:
            raise ValueError(f'Unsupported run_type: {run_type}')
        if not n_part or n_part < 1:
            raise ValueError('n_part must be positive!')
        if n_repeat < 0:
            raise ValueError('n_repeat must be non-negative!')
        self.run_type = run_type
        self.n_part = n_part
        self.n_repeat = n_repeat

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        if not isinstance(scenario.size, AerosolModalSizeState):
            raise TypeError('Non-modal aerosol particle size state cannot be used to create PartMC input!')
        aero_data = [AeroData(
            species = s.name,
            density = s.density,
            ions_in_soln = s.ions_in_soln,
            molecular_weight = 1000 * s.molar_mass,
            kappa = s.hygroscopicity,
        ) for s in scenario.aerosols]
        aero_init = [AeroMode(
            mode_name = m.name.replace(' ', '_'),
            mass_frac = {m.species[i].name:m.mass_fractions[i] for i in range(len(m.species))},
            diam_type = 'geometric', # FIXME: could also be 'mobility'
            mode_type = 'log_normal', # FIXME: could also be 'exp', 'mono', 'sampled'
            num_conc = m.number,
            geom_mean_diam = m.geom_mean_diam,
            log10_geom_std_dev = m.log10_geom_std_dev,
        ) for m in scenario.size.modes]
        return Input(
            run_type = self.run_type,
            n_part = self.n_part,
            n_repeat = self.n_repeat,

            restart = False,
            do_select_weighting = False,
            #do_select_weighting = True,
            #weight_type = 'power',
            #weighting_exponent = 0,

            t_max = nstep * dt,
            del_t = dt,
            t_output = nstep * dt,
            t_progress = nstep * dt,

            do_camp_chem = False,

            gas_data = tuple([gas.name for gas in scenario.gases]),
            gas_init = tuple([gas_conc for gas_conc in scenario.gas_concs]),

            aerosol_data = tuple(aero_data),
            do_fractal = False,
            aerosol_init = tuple(aero_init),

            temp_profile = [(0, scenario.temperature)],
            pressure_profile = [(0, scenario.pressure)],
            height_profile = [(0, scenario.height)],

            rel_humidity = 0.5,#scenario.relative_humidity,
            latitude = 0,       # FIXME:
            longitude = 0,      # FIXME:
            altitude = 0,       # FIXME:
            start_time = 21600, # FIXME:
            start_day = 200,    # FIXME:

            do_coagulation = self.processes.coagulation,
            do_condensation = False, # this is cloud condensation, not for aerosols
            do_mosaic = False,
            do_optical = self.processes.optics,
            do_nucleation = self.processes.nucleation,

            rand_init = 0, # FIXME: uses time to initialize random seed
            allow_doubling = False,
            allow_halving = False,
            record_removals = False,
            do_parallel = False,
        )

    def invocation(self,
                   exe: str,
                   prefix: str) -> str:
        return f'{exe} {prefix}.spec'

    def write_input_files(self,
                          input,
                          dir: str,
                          prefix: str):
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        output_dir = os.path.join(dir, 'out')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # write the main (.spec) file
        output_prefix = os.path.join('out', prefix)
        spec_content = f'run_type {input.run_type}\noutput_prefix {output_prefix}\n'

        # simulation metadata
        if input.run_type == 'particle':
            spec_content += f'n_repeat {input.n_repeat if input.n_repeat else 1}\nn_part {input.n_part}\n'
        if input.restart:
            spec_content += 'restart yes\n'
        else:
            spec_content += 'restart no\n'
        if input.do_select_weighting:
            spec_content += 'do_select_weighting yes\n'
            spec_content += f'weight_type {input.weight_type}\n'
            spec_content += f'weighting_exponent {input.weighting_exponent}\n'
        else:
            spec_content += 'do_select_weighting no\n'
        spec_content += '\n'

        # time info
        spec_content += f't_max {input.t_max}\ndel_t {input.del_t}\nt_output {input.t_output}\nt_progress {input.t_progress}\n'
        spec_content += '\n'

        # chemistry
        if input.do_camp_chem:
            spec_content += 'do_camp_chem yes\n'
        else:
            spec_content += 'do_camp_chem no\n'
        spec_content += '\n'

        # gas data
        spec_content += 'gas_data gas_data.dat\ngas_init gas_init.dat\n'
        spec_content += '\n'

        # aerosol data
        spec_content += 'aerosol_data aero_data.dat\n'
        if input.do_fractal:
            spec_content += 'do_fractal yes\n'
        else:
            spec_content += 'do_fractal no\n'
        spec_content += 'aerosol_init aero_init_dist.dat\n'
        spec_content += '\n'

        # atmospheric environment data
        spec_content += 'temp_profile temp.dat\npressure_profile pres.dat\nheight_profile height.dat\n'
        spec_content += 'gas_emissions gas_emit.dat\ngas_background gas_back.dat\n'
        spec_content += 'aero_emissions aero_emit.dat\naero_background aero_back.dat\n'
        if input.loss_function:
            spec_content += f'loss_function {input.loss_function}\n'
        else:
            spec_content += 'loss_function none\n'
        spec_content += '\n'

        spec_content += f'rel_humidity {input.rel_humidity}\n'
        spec_content += f'latitude {input.latitude}\n'
        spec_content += f'longitude {input.longitude}\n'
        spec_content += f'altitude {input.altitude}\n'
        spec_content += f'start_time {input.start_time}\n'
        spec_content += f'start_day {input.start_day}\n'
        spec_content += '\n'

        # processes
        if input.do_coagulation:
            spec_content += f'do_coagulation yes\ncoag_kernel {input.coag_kernel if input.coag_kernel else 'zero'}\n'
        else:
            spec_content += 'do_coagulation no\n'
        if input.do_condensation:
            spec_content += 'do_condensation yes\n'
        else:
            spec_content += 'do_condensation no\n'
        if input.do_mosaic:
            spec_content += 'do_mosaic yes\n'
            # do_optical is only parsed when do_mosaic is set
            if input.do_optical:
                spec_content += 'do_optical yes\n'
            else:
                spec_content += 'do_optical no\n'
        else:
            spec_content += 'do_mosaic no\n'
        if input.do_nucleation:
            spec_content += 'do_nucleation yes\n'
        else:
            spec_content += 'do_nucleation no\n'
        spec_content += '\n'

        # misc simulation parameters
        spec_content += f'rand_init {input.rand_init}\n'
        if input.allow_doubling:
            spec_content += 'allow_doubling yes\n'
        else:
            spec_content += 'allow_doubling no\n'
        if input.allow_halving:
            spec_content += 'allow_halving yes\n'
        else:
            spec_content += 'allow_halving no\n'
        if input.record_removals:
            spec_content += 'record_removals yes\n'
        else:
            spec_content += 'record_removals no\n'
        if input.do_parallel:
            spec_content += 'do_parallel yes'
        else:
            spec_content += 'do_parallel no'

        with open(os.path.join(dir, prefix + '.spec'), 'w') as f:
            f.write(spec_content)

        # write auxiliary data files

        # gas_data.dat, gas_init.dat
        with open(os.path.join(dir, 'gas_data.dat'), 'w') as f:
            f.write('# list of gas species\n')
            for gas in input.gas_data:
                f.write(f'{gas}\n')
        with open(os.path.join(dir, 'gas_init.dat'), 'w') as f:
            f.write('# species\tinitial concentration (ppb)\n')
            for g in range(len(input.gas_data)):
                f.write(f'{input.gas_data[g]}\t{input.gas_init[g]}\n')

        # aero_data.dat, aero_init_dist.dat, aero_init_comp.dat
        with open(os.path.join(dir, 'aero_data.dat'), 'w') as f:
            f.write('#\tdens (kg/m^3)\tions in soln (1)\tmolec wght (kg/mole)\tkappa (1)\n')
            for aero in input.aerosol_data:
                f.write(f'{aero.species}\t{aero.density}\t{aero.ions_in_soln}\t{aero.molecular_weight}\t{aero.kappa}\n')
        self._write_aero_modes(dir, 'aero_init', input.aerosol_init)

        # temp.dat, pres.dat, height.dat
        with open(os.path.join(dir, 'temp.dat'), 'w') as f:
            f.write('# time (s)\n# temp (K)\n')
            f.write('\t'.join(['time'] + [str(pair[0]) for pair in input.temp_profile]) + '\n')
            f.write('\t'.join(['temp'] + [str(pair[1]) for pair in input.temp_profile]))
        with open(os.path.join(dir, 'pres.dat'), 'w') as f:
            f.write('# time (s)\n# pressure (Pa)\n')
            f.write('\t'.join(['time'] + [str(pair[0]) for pair in input.pressure_profile]) + '\n')
            f.write('\t'.join(['pressure'] + [str(pair[1]) for pair in input.pressure_profile]))
        with open(os.path.join(dir, 'height.dat'), 'w') as f:
            f.write('# time (s)\n# height (m)\n')
            f.write('\t'.join(['time'] + [str(pair[0]) for pair in input.height_profile]) + '\n')
            f.write('\t'.join(['height'] + [str(pair[1]) for pair in input.height_profile]))

        # gas_emit.dat
        if input.gas_emissions:
            gas_emission_species = [input.gas_emissions[0].time_series[1].keys()]
            gas_emission_species.remove('rate')
            with open(os.path.join(dir, 'gas_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate = scaling parameter\n# emissions (mol m^{-2} s^{-1})\n')
                f.write('\t'.join(['time'] + [pair[0] for pair in input.gas_emissions]) + '\n')
                f.write('\t'.join(['rate'] + [pair[1]['rate'] for pair in input.gas_emissions]) + '\n')
                f.write('\t'.join([species_name] + [emit.pair[1][species_name] \
                                  for species_name in gas_emission_species \
                                  for pair in input.gas_emissions]))
        else:
            # write a gas emissions file with zero data
            with open(os.path.join(dir, 'gas_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate = scaling parameter\n# emissions (mol m^{-2} s^{-1})\n')
                f.write('time\t0.0\n')
                f.write('rate\t1.0\n')
                for gas in input.gas_data:
                    f.write(f'{gas}\t0.0\n')

        # gas_back.dat
        if input.gas_background:
            gas_background_species = [input.gas_background[0].time_series[1].keys()]
            gas_background_species.remove('rate')
            with open(os.path.join(dir, 'gas_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# concentrations (ppb)\n')
                f.write('\t'.join(['time'] + [pair[0] for pair in input.gas_background]) + '\n')
                f.write('\t'.join(['rate'] + [pair[1]['rate'] for pair in input.gas_background]) + '\n')
                f.write('\t'.join([species_name] + [pair[1][species_name] \
                                  for species_name in gas_background_species \
                                  for pair in input.gas_background]))
        else:
            # write a gas background file with zero data
            with open(os.path.join(dir, 'gas_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate = scaling parameter\n# emissions (mol m^{-2} s^{-1})\n')
                f.write('time\t0.0\n')
                f.write('rate\t1.0\n')
                for gas in input.gas_data:
                    f.write(f'{gas}\t0.0\n')

        # aero_emit.dat, aero_emit_dist_*.dat, aero_emit_comp_*.dat
        if input.aero_emissions:
            with open(os.path.join(dir, 'aero_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in input.aero_emissions]) + '\n')
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in input.aero_emissions]) + '\n')
                f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(input.aero_emissions))]))
            for i, time_series in enumerate(input.aero_emissions):
                input._write_aero_modes(dir, f'aero_emit_dist_{i+1}', input.aero_emissions)
        else:
            # write a zero-scaled aerosol emissions file
            with open(os.path.join(dir, 'aero_emit.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('time\t0.0\n')
                f.write('rate\t0.0\n')
                f.write('dist\taero_init_dist.dat\n')

        # aero_back.dat, aero_back_dist.dat, aero_back_comp.dat
        if input.aero_background:
            with open(os.path.join(dir, 'aero_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('\t'.join(['time'] + [time_series[0] for time_series in input.aero_background]) + '\n')
                f.write('\t'.join(['rate'] + [time_series[1]['rate'] for time_series in input.aero_background]) + '\n')
                f.write('\t'.join(['dist'] + [f'aero_emit_dist_{i+1}.dat' for i in range(len(input.aero_background))]))
            for i, time_series in enumerate(input.aero_background):
                input._write_aero_modes(dir, f'aero_back_dist_{i+1}', input.aero_background)
        else:
            # write a zero-scaled aerosol background file
            with open(os.path.join(dir, 'aero_back.dat'), 'w') as f:
                f.write('# time (s)\n# rate (s^{-1})\n# aerosol distribution filename\n')
                f.write('time\t0.0\n')
                f.write('rate\t0.0\n')
                f.write('dist\taero_init_dist.dat\n')

    def _write_aero_modes(self,
                          dir: str,
                          prefix: str,
                          modes):
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
            with open(os.path.join(dir, f'{prefix}_comp_{i+1}.dat'), 'w') as f:
                f.write('#\tproportion\n')
                for species, mass_frac in mode.mass_frac.items():
                    f.write(f'{species}\t{mass_frac}\n')

    def read_output_files(self,
                          input,
                          dir: str,
                          prefix: str) -> Output:
        n_repeat = self.n_repeat
        timestep = -1 # FIXME: for now, we use the last timestep in the output
        if timestep == -1:
            nc_files = [f for f in os.listdir(dir) if filename.endswith('.nc')]
            nc_files.sort()
            nc_file = nc_files[-1]
        else:
            nc_file = os.path.join(dir, prefix + '_' + str(timestep).zfill(4) + '_' + '1'.zfill(8) + '.nc')
            if not os.path.exists(nc_file):
                raise OSError('Could not open
        dNdlnD_repeat = np.zeros([len(lnDs), n_repeat])
        for i, repeat in enumerate(range(1, n_repeat+1)):
            output_dir = os.path.join(dir, 'out')
            output_file = prefix + '_' + str(timestep)
            nc_output = read_partmc.get_ncfile(partmc_dir, timestep, ensemble_number=repeat)
            dNdlnD_repeats[:,ii] = get_partmc_dsd_onefile(lnDs,ncfile,density_type=density_type)
        return Output(
            model = self.name,
            input = input,
            dNdlnD = dNdlnD,
            )
