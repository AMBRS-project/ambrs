"""ambrs.carma_jax -- data types and functions related to CARMA-JAX, a JAX
port of the CARMA sectional (bin) aerosol microphysics model
(https://github.com/reflective-org/carma-jax).

CARMA-JAX is a Python/JAX library with no input files and no command line, so it
is stepped *in process* (see AerosolModel.run and run_ensemble) rather than
through ambrs.runners.PoolRunner, and its results are turned into an
ambrs.analysis.Output by retrieve_model_state, exactly as for the other models.

This adapter wires up CARMA-JAX's **coagulation**, its production-quality path:
a single internally-mixed involatile particle group on a geometric mass grid,
with the physical Brownian kernel built from the scenario's temperature and
pressure. CARMA-JAX's condensational growth and sulfate nucleation exist on its
dev branch but their environment builder is not yet part of the installed
package, so requesting those processes raises rather than silently doing
nothing (see the CARMA_JAX.md notes).

CARMA-JAX is an optional dependency: importing ambrs never requires it. When it
is absent this module still imports, but constructing an AerosolModel raises
ImportError. `_CARMA_JAX_AVAILABLE` records whether the library was importable.
"""

from .aerosol import AerosolProcesses, AerosolModalSizeState
from .aerosol_model import BaseAerosolModel, NotImplementedError
from .analysis import Output
from .gas import build_gas_mixture
from .scenario import Scenario

from dataclasses import dataclass, field
from typing import Any
import os.path
import warnings

import numpy as np
from scipy.special import erf

# CARMA-JAX is optional. Importing carma.precision enables float64 globally as a
# side effect (jax.config.update("jax_enable_x64", True)), so it must run before
# any other use of JAX. OSError is caught alongside ImportError so a broken or
# partial install cannot break `import ambrs`.
try:
    import carma.precision as _precision  # noqa: F401  (import first!)
    from carma.precision import DTYPE as _DTYPE
    from carma.bins import setup_bins as _setup_bins
    from carma.coagulation.setup_coag import setup_coag as _setup_coag
    from carma.step import make_step_coag as _make_step_coag
    from carma.setup_atm import setup_atm as _setup_atm
    from carma.setup_vf import setup_vf_jit as _setup_vf_jit
    from carma.setup_ckern import setup_ckern_jit as _setup_ckern_jit
    from carma.config import CarmaConfig as _CarmaConfig
    from carma.config import ElementConfig as _ElementConfig
    from carma.config import GroupConfig as _GroupConfig
    from carma.enums import ElementType as _ElementType
    from carma.enums import GridType as _GridType
    from carma.constants import RPA2CGS as _RPA2CGS       # Pa -> dyne/cm^2
    from carma.constants import SMALL_PC as _SMALL_PC     # concentration floor
    _CARMA_JAX_AVAILABLE = True
except (ImportError, OSError):
    _CARMA_JAX_AVAILABLE = False


def lognormal_to_bins(number, geom_mean_diam, log10_geom_std_dev, rlow, rup):
    """Integrate a single log-normal mode onto CARMA's bin grid, exactly.

The number placed in each bin is the analytic integral of the log-normal number
distribution between the bin's boundary radii, so the mode's total number is
preserved to the extent the grid covers it (a warning is issued when more than
1% falls outside the grid).

Parameters:
    * number: modal number concentration [# cm^-3]
    * geom_mean_diam: geometric mean diameter [m]
    * log10_geom_std_dev: log10 of the geometric standard deviation
    * rlow, rup: bin boundary radii [cm], each shape (nbin,)

Returns the number per bin [# cm^-3], shape (nbin,).
"""
    r_g = 0.5 * geom_mean_diam * 100.0   # geometric mean radius [cm]
    ln_sigma = np.log(10.0 ** log10_geom_std_dev)
    scale = np.sqrt(2.0) * ln_sigma
    upper = erf(np.log(np.asarray(rup) / r_g) / scale)
    lower = erf(np.log(np.asarray(rlow) / r_g) / scale)
    binned = 0.5 * number * (upper - lower)
    covered = np.sum(binned)
    if number > 0.0 and covered < 0.99 * number:
        warnings.warn(
            'carma_jax: only %.1f%% of a mode (GMD %.3g m) falls inside the '
            'bin grid; enlarge nbin or move rmin to cover it.'
            % (100.0 * covered / number, geom_mean_diam),
            stacklevel = 2)
    return binned


def blend_compositions(modes):
    """Blend the modes' dry compositions into a single species mixture.

CARMA-JAX's coagulation configuration carries one internally-mixed involatile
element, so a multi-mode scenario's compositions are blended into one mixture,
weighted by each mode's volume (from its own number, diameter and width).
Aerosol water is dropped. Returns (names, fractions) tuples; warns when the
modes' compositions actually differ, since the blend is then approximate.
"""
    totals = {}
    signatures = set()
    for mode in modes:
        ln_sigma = np.log(10.0 ** mode.log10_geom_std_dev)
        volume = (mode.number * mode.geom_mean_diam ** 3
                  * np.exp(4.5 * ln_sigma ** 2))
        dry = [(s.name, float(f))
               for s, f in zip(mode.species, mode.mass_fractions)
               if s.name != 'H2O' and float(f) > 0.0]
        signatures.add(tuple(sorted(dry)))
        for name, frac in dry:
            totals[name] = totals.get(name, 0.0) + volume * frac
    if len(signatures) > 1:
        warnings.warn(
            'carma_jax: modes have differing compositions; CARMA carries a '
            'single internally-mixed element, so they were blended '
            '(volume-weighted) into one mixture.', stacklevel = 2)
    total = sum(totals.values())
    if total <= 0.0:
        return ('SO4',), (1.0,)
    names = tuple(sorted(totals))
    return names, tuple(totals[n] / total for n in names)


@dataclass
class Input:
    """ambrs.carma_jax.Input -- an input dataclass for the CARMA-JAX box model.

CARMA-JAX has no native input file format, so this holds the per-bin state
directly, together with the atmospheric state and timestepping needed to drive
a run. Bin geometry lives on the AerosolModel (it is fixed at construction).
"""

    pc: Any              # particle number per bin [# cm^-3], shape (1, nbin, 1)

    # atmospheric state
    temp: float          # temperature [K]
    press: float         # pressure [Pa]
    rh: float            # relative humidity [0-1] (recorded; coagulation ignores it)

    # timestepping
    dt: float            # time step [s]
    nstep: int           # number of time steps

    # the blended dry composition reported in the output population
    species_names: tuple = field(default_factory = tuple)
    species_fracs: tuple = field(default_factory = tuple)

    # bookkeeping (not consumed by the solver)
    aerosols: tuple = field(default_factory = tuple)
    scenario: Any = None


class AerosolModel(BaseAerosolModel):
    """ambrs.carma_jax.AerosolModel -- an in-process AMBRS adapter for CARMA-JAX.

Create inputs as for any other model, then step them in process with run() (or
run_ensemble() for a whole ensemble) rather than with ambrs.runners.PoolRunner,
which drives external executables."""

    def __init__(self,
                 processes: AerosolProcesses,
                 nbin: int = 47,
                 rmin: float = 2e-10,
                 rmrat: float = 2.0,
                 density: float = 1770.0):
        """ambrs.carma_jax.AerosolModel.__init__(processes, ...)

Parameters beyond the AerosolProcesses flags:
    * nbin: number of size bins
    * rmin: radius of the smallest bin [m] (default 0.2 nm)
    * rmrat: mass ratio between adjacent bins (default 2, mass doubling);
      the defaults span particle radii of about 0.2 nm to 8 um
    * density: particle density [kg m^-3], used for the bin mass grid and the
      coagulation kernel

Only coagulation is wired in this adapter (see the module docstring), so any
other enabled microphysical process raises ValueError rather than silently
doing nothing.
"""
        if not _CARMA_JAX_AVAILABLE:
            raise ImportError(
                'carma_jax is not installed, so ambrs.carma_jax.AerosolModel '
                'cannot be used. Install it with\n'
                '    pip install "carma-jax @ git+https://github.com/reflective-org/carma-jax.git@dev"')
        unsupported = [name for name in
                       ('condensation', 'nucleation', 'gas_phase_chemistry',
                        'aqueous_chemistry', 'aging', 'optics')
                       if getattr(processes, name, False)]
        if unsupported:
            raise ValueError(
                'ambrs.carma_jax supports coagulation only; %s not yet wired '
                '(CARMA-JAX has growth and sulfate nucleation on its dev '
                'branch, but their environment builder is not part of the '
                'installed package).' % ', '.join(unsupported))
        BaseAerosolModel.__init__(self, 'carma-jax', processes)

        import jax.numpy as jnp

        self.nbin = int(nbin)
        self.density = float(density)
        rho_cgs = self.density / 1000.0          # kg/m^3 -> g/cm^3
        rmin_cm = float(rmin) * 100.0            # m -> cm

        (self.r, self.rmass, vol, dr, dm,
         self.rup, self.rlow, rmassup) = _setup_bins(
            rmin = rmin_cm, rmrat = float(rmrat), nbin = self.nbin,
            rho = rho_cgs)

        groups = (_GroupConfig(
            name = 'aerosol', ishape = 1, ienconc = 0, is_ice = False,
            is_cloud = False, is_sulfate = False, do_vtran = False,
            do_drydep = False, ifallrtn = 1, irhswell = 0,
            rmrat = float(rmrat), eshape = 1.0, rmin = rmin_cm,
            r = self.r, rmass = self.rmass, vol = vol, dr = dr, dm = dm,
            rmassup = rmassup, rup = self.rup, rlow = self.rlow,
            rrat = jnp.ones(self.nbin, dtype = _DTYPE),
            rprat = jnp.ones(self.nbin, dtype = _DTYPE),
            arat = jnp.ones(self.nbin, dtype = _DTYPE)),)
        elements = (_ElementConfig(
            name = 'mixed-aerosol',
            rho = jnp.full(self.nbin, rho_cgs, dtype = _DTYPE),
            igroup = 0, itype = int(_ElementType.I_INVOLATILE),
            icomposition = 0, isolute = -1, kappa = 0.0),)

        # the coagulation pair tables are pure-Python and cost ~1 s; built once
        coag = _setup_coag(self.nbin, 1, 1, groups, elements,
                           np.array([[0]], dtype = np.int32),
                           np.array([[0]], dtype = np.int32))

        self._config = _CarmaConfig(
            nbin = self.nbin, nelem = 1, ngroup = 1, ngas = 0, nsolute = 0,
            elements = elements, groups = groups, gases = (), solutes = (),
            coag = coag,
            do_coag = bool(processes.coagulation), do_grow = False,
            do_vtran = False, do_vdiff = False, do_thermo = False,
            do_substep = False, do_explised = False, do_incloud = False,
            do_clearsky = False, do_detrain = False, do_pheat = False,
            do_pheatatm = False, do_cnst_rlh = False,
            itbnd_pc = 1, ibbnd_pc = 1,
            maxsubsteps = 1, minsubsteps = 1, maxretries = 5, conmax = 0.0,
            cstick = 1.0, gsticki = 1.0, gstickl = 1.0, tstick = 1.0,
            dt_threshold = 0.0,
            igash2o = -1, igash2so4 = -1, igasso2 = -1)
        self._step = _make_step_coag(self._config)

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        """ambrs.carma_jax.AerosolModel.create_input(scenario, dt, nstep) ->
ambrs.carma_jax.Input describing a CARMA-JAX simulation of the given scenario.

Every mode is integrated analytically onto the bin grid and the modes are
summed; their compositions are blended into the single internally-mixed
mixture CARMA carries (see blend_compositions).

Parameters:
    * scenario: an ambrs.Scenario object defining an individual scenario
    * dt: a fixed time step size for simulations
    * nstep: the number of steps in each simulation"""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if nstep <= 0:
            raise ValueError("nstep must be positive")
        if not isinstance(scenario.size, AerosolModalSizeState):
            raise TypeError('Non-modal aerosol particle size state cannot be '
                            'used to create carma-jax input!')

        binned = np.zeros(self.nbin)
        for mode in scenario.size.modes:
            # an ambrs mode's number is [# m^-3]; CARMA bins number in [# cm^-3]
            binned += lognormal_to_bins(
                mode.number * 1e-6, mode.geom_mean_diam,
                mode.log10_geom_std_dev, self.rlow, self.rup)
        pc = np.full((1, self.nbin, 1), float(_SMALL_PC))
        pc[0, :, 0] = np.maximum(binned, float(_SMALL_PC))

        names, fracs = blend_compositions(scenario.size.modes)
        return Input(
            pc = pc,
            temp = scenario.temperature,
            press = scenario.pressure,
            rh = scenario.relative_humidity,
            dt = dt,
            nstep = nstep,
            species_names = names,
            species_fracs = fracs,
            aerosols = tuple(scenario.aerosols),
            scenario = scenario,
        )

    def _environment(self, temp, press):
        """The atmosphere-dependent pieces of a run: air properties and the
physical Brownian coagulation kernel, built from temperature [K] and pressure
[Pa] the way CARMA's own drivers do (setup_atm -> setup_vf -> setup_ckern)."""
        import jax.numpy as jnp

        dz = _DTYPE(1.0e5)   # a nominal 1 km box; zmet is 1 on a Cartesian grid
        t = jnp.asarray([temp], dtype = _DTYPE)
        p = jnp.asarray([press * float(_RPA2CGS)], dtype = _DTYPE)
        pl = jnp.asarray([press * float(_RPA2CGS)] * 2, dtype = _DTYPE)
        zc = jnp.asarray([0.5 * dz], dtype = _DTYPE)
        zl = jnp.asarray([0.0, dz], dtype = _DTYPE)
        rhoa, _dz, zmet, _zmetl, rmu, _thcond, _rhoa_wet = _setup_atm(
            t, p, pl, zc, zl, _GridType.I_CART)

        r_wet = jnp.broadcast_to(
            jnp.asarray(self.r, dtype = _DTYPE)[None, :, None],
            (1, self.nbin, 1))
        rhop_wet = jnp.full((1, self.nbin, 1),
                            _DTYPE(self.density / 1000.0))
        ones = jnp.ones((self.nbin, 1), dtype = _DTYPE)
        rmass_2d = jnp.asarray(self.rmass, dtype = _DTYPE)[:, None]

        vf, re, bpm = _setup_vf_jit(t, rhoa, zmet, rmu, r_wet, rhop_wet,
                                    ones, ones)
        ckernel = _setup_ckern_jit(
            t, rhoa, zmet, rmu, r_wet, ones, ones, bpm, rmass_2d, re, vf,
            _DTYPE(self._config.cstick),
            use_vw = jnp.zeros((1, 1), dtype = bool))
        return t, zmet, ckernel

    def run(self, input, scenario_name: str = 'carma-jax') -> Output:
        """ambrs.carma_jax.AerosolModel.run(input, scenario_name) ->
ambrs.analysis.Output describing the final state of the simulation.

Steps the input's state input.nstep times (one JIT-compiled coagulation step,
reused across every scenario) and converts the result into an Output, exactly
as retrieve_model_state does for the executable models."""
        import jax.numpy as jnp

        t, zmet, ckernel = self._environment(input.temp, input.press)
        pc = jnp.asarray(input.pc, dtype = _DTYPE)
        gc = jnp.zeros((1, 0), dtype = _DTYPE)   # this configuration has no gases
        pcl, gcl, told = pc, gc, t
        for _ in range(input.nstep):
            pc, gc, t, pcl, gcl, _pconmax = self._step(
                pc, gc, t, pcl, gcl, told, zmet, ckernel, float(input.dt))

        return Output(
            model_name = self.name,
            scenario_name = scenario_name,
            scenario = input.scenario,
            timestep = input.nstep,
            particle_population = self.build_population(
                np.asarray(pc[0, :, 0]), input.species_names,
                input.species_fracs),
            # this CARMA configuration carries no gas phase
            gas_mixture = build_gas_mixture({'units': 'kg_per_kg'}),
            thermodynamics = {'T': input.temp, 'p': input.press,
                              'RH': input.rh},
        )

    def build_population(self, number_per_bin, species_names, species_fracs):
        """Build a part2pop ParticlePopulation from CARMA's per-bin state.

Each meaningfully-populated bin becomes one particle: diameter from the bin
centre, the blended dry composition (CARMA carries one internally-mixed
element, so composition is uniform across bins), and number converted back to
the [# m^-3] the rest of ambrs uses.
"""
        from part2pop import ParticlePopulation, make_particle
        from part2pop.species.registry import get_species

        diameters = 2.0 * np.asarray(self.r) / 100.0   # radius [cm] -> D [m]
        num_concs = np.asarray(number_per_bin) * 1e6   # [# cm^-3] -> [# m^-3]
        # CARMA floors empty bins at SMALL_PC rather than zero; drop them
        meaningful = np.asarray(number_per_bin) > 1e3 * float(_SMALL_PC)
        if not np.any(meaningful):
            meaningful = num_concs == num_concs.max()

        species = tuple(get_species(name, None) for name in species_names)
        population = ParticlePopulation(
            species = species, spec_masses = [], num_concs = [], ids = [],
            species_modifications = {})
        part_id = 0
        for k in range(self.nbin):
            if not meaningful[k]:
                continue
            particle = make_particle(
                diameters[k], species, list(species_fracs),
                specdata_path = None, species_modifications = {},
                D_is_wet = False)
            part_id += 1
            population.set_particle(particle, part_id, float(num_concs[k]))
        return population

    def write_input_files(self, input, dir: str, prefix: str) -> None:
        """ambrs.carma_jax.AerosolModel.write_input_files(input, dir, prefix)

CARMA-JAX reads no input files, so this records the state a run started from,
for provenance and so a run can be reproduced later: the arrays go to
<dir>/<prefix>.npz and the scalar settings to a readable <dir>/<prefix>.txt.
The originating Scenario is not serialized."""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        np.savez(
            os.path.join(dir, prefix + '.npz'),
            pc = input.pc, temp = input.temp, press = input.press,
            rh = input.rh, dt = input.dt, nstep = input.nstep,
            species_names = np.asarray(input.species_names),
            species_fracs = np.asarray(input.species_fracs),
            nbin = self.nbin, density = self.density,
        )
        with open(os.path.join(dir, prefix + '.txt'), 'w') as f:
            f.write('# generated by ambrs.carma_jax.AerosolModel.write_input_files\n')
            for name in ('temp', 'press', 'rh', 'dt', 'nstep'):
                f.write(f'{name} = {getattr(input, name)}\n')
            f.write(f'species_names = {list(input.species_names)}\n')
            f.write(f'species_fracs = {list(input.species_fracs)}\n')
            f.write(f'nbin = {self.nbin}\ndensity = {self.density}\n')

    def read_input(self, dir: str, prefix: str, scenario = None) -> Input:
        """ambrs.carma_jax.AerosolModel.read_input(dir, prefix) ->
ambrs.carma_jax.Input reconstructed from the files write_input_files wrote.

The originating Scenario isn't stored in the files, so pass it here if the
resulting Input is to be used to build an Output."""
        filename = os.path.join(dir, prefix + '.npz')
        if not os.path.exists(filename):
            raise OSError(f'CARMA-JAX input file not found: {filename}')
        d = np.load(filename)
        if int(d['nbin']) != self.nbin:
            raise ValueError(
                f'recorded input has {int(d["nbin"])} bins but this model was '
                f'built with {self.nbin}; construct the model to match')
        return Input(
            pc = d['pc'], temp = float(d['temp']), press = float(d['press']),
            rh = float(d['rh']), dt = float(d['dt']), nstep = int(d['nstep']),
            species_names = tuple(str(s) for s in d['species_names'].tolist()),
            species_fracs = tuple(float(f) for f in d['species_fracs'].tolist()),
            scenario = scenario,
        )

    def run_ensemble(self, inputs: list) -> list:
        """ambrs.carma_jax.AerosolModel.run_ensemble(inputs) -> list of Outputs

Runs each input in process, in order, reusing the single compiled coagulation
step. Scenarios are named by their 1-based index, as PoolRunner names its
scenario directories."""
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of scenario inputs')
        width = len(str(len(inputs)))
        return [self.run(input, scenario_name = f'{i + 1:0{width}d}')
                for i, input in enumerate(inputs)]

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'carma-jax runs in process and has no executable to invoke; step it '
            'with AerosolModel.run() or run_ensemble() instead of PoolRunner.')


def retrieve_model_state(
        scenario_name: str,
        scenario: Scenario,
        timestep: int,
        processes: AerosolProcesses = None,
        ensemble_output_dir: str = 'carma_jax_runs',
        **model_options) -> Output:
    """ambrs.carma_jax.retrieve_model_state(scenario_name, scenario, timestep, ...)
-> ambrs.analysis.Output describing the state of a CARMA-JAX simulation.

This mirrors partmc.retrieve_model_state and mam4.retrieve_model_state, but
because CARMA-JAX runs in process there is no output on disk to read: the state
recorded by write_input_files in <ensemble_output_dir>/<scenario_name>/ is
loaded and stepped forward to `timestep`. Running a scenario directly with
AerosolModel.run() is the cheaper path when the Input is still in hand.

Parameters:
    * scenario_name: names the scenario's directory and the resulting Output
    * scenario: the Scenario the run came from, recorded in the Output
    * timestep: the number of steps to advance (1 gives the initial state)
    * processes: the AerosolProcesses to run with (default: coagulation on)
    * ensemble_output_dir: the directory holding the per-scenario directories
    * model_options: forwarded to AerosolModel (nbin, rmin, rmrat, density)"""
    if timestep < 1:
        raise ValueError('timestep must be positive; 1 gives the initial state')
    model = AerosolModel(processes or AerosolProcesses(coagulation = True),
                         **model_options)
    dir = os.path.join(ensemble_output_dir, scenario_name)
    input = model.read_input(dir, scenario_name, scenario = scenario)
    input.nstep = timestep - 1   # timestep 1 == initial conditions
    return model.run(input, scenario_name = scenario_name)
