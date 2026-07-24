"""ambrs.tomas_jax -- data types and functions related to TOMAS-JAX, a JAX
re-implementation of the TwO-Moment Aerosol Sectional (TOMAS) box model
(https://github.com/reflective-org/tomas-jax).

Unlike PartMC and MAM4, TOMAS-JAX is a Python/JAX library rather than a compiled
executable: there are no native input files to write and no command to invoke.
It is therefore stepped *in process* (see AerosolModel.run and run_ensemble)
rather than through ambrs.runners.PoolRunner, and its results are turned into an
ambrs.analysis.Output by retrieve_model_state, exactly as for the other models.

TOMAS-JAX is an optional dependency: importing ambrs never requires it. When it
is absent this module still imports, but constructing an AerosolModel raises
ImportError. `_TOMAS_AVAILABLE` records whether the library was importable.
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

# TOMAS-JAX is optional. Its config module must be imported before anything else
# touches JAX, because importing it enables float64 globally
# (jax.config.update("jax_enable_x64", True)).
#
# OSError is caught alongside ImportError so that a partial or broken install
# (for example a wheel missing its bundled data files) cannot break `import ambrs`.
try:
    import tomas_jax.core.config as _tomas_config  # noqa: F401  (import first!)
    from tomas_jax.core.config import (
        xk_boundaries,
        ICOMP, N_GAS_SPECIES,
        SRTSO4, SRTSO2, SRTNH4, SRTH2O, SRTORG1,
        MW_H2SO4, MW_SO2, AVOGADRO, PI,
    )
    from tomas_jax.solvers.condensation import make_step

    # AMBRS aerosol species -> TOMAS species index. TOMAS carries sulfate, a run
    # of organics (SRTORG1..IORG), ammonium and water; see UNMAPPED_SPECIES below
    # for everything else.
    AEROSOL_SPECIES_MAP = {
        'SO4': SRTSO4,
        'NH4': SRTNH4,
        'H2O': SRTH2O,
        # organics are lumped into the first organic slot
        'OC': SRTORG1, 'MSA': SRTORG1,
        'ARO1': SRTORG1, 'ARO2': SRTORG1, 'ALK1': SRTORG1, 'OLE1': SRTORG1,
        'API1': SRTORG1, 'API2': SRTORG1, 'LIM1': SRTORG1, 'LIM2': SRTORG1,
    }

    # AMBRS gas species -> index in TOMAS's gas array Gc. NH3 has no Gc slot; it
    # is passed to the nucleation scheme as nh3_conc instead (see create_input).
    GAS_SPECIES_MAP = {
        'H2SO4': SRTSO4,
        'SO2': SRTSO2,
    }

    # molar masses [g/mol] used to convert gas number concentrations to the
    # kg-per-grid-cell units TOMAS stores in Gc
    _GAS_MOLAR_MASS = {
        'H2SO4': MW_H2SO4,
        'SO2': MW_SO2,
    }

    _TOMAS_AVAILABLE = True
except (ImportError, OSError):
    _TOMAS_AVAILABLE = False

# AMBRS aerosol species with no TOMAS analog. TOMAS has no black carbon, dust or
# inorganic-salt slot, so this mass is lumped into the organic slot: total mass
# is conserved but the speciation is approximate. Callers are warned.
UNMAPPED_SPECIES = frozenset({'BC', 'OIN', 'NO3', 'Cl', 'Na', 'Ca', 'CO3'})

# TOMAS applies its processes in this order within a step
_PROCESS_ORDER = ('so2_chemistry', 'nucleation', 'coagulation', 'condensation',
                  'dilution')


def _gas_conc_to_kg_per_cell(conc_molec_cm3, boxvol, molar_mass_g_mol):
    """Convert a gas number concentration [molec cm^-3] to [kg per grid cell]."""
    return conc_molec_cm3 * boxvol * (molar_mass_g_mol / 1000.0) / AVOGADRO


def map_species_fractions(species_names, mass_fractions):
    """Map an AMBRS mode's (species_names, mass_fractions) onto TOMAS species
indices, conserving total mass.

Returns (frac_by_index, remapped): a {tomas_index: mass_fraction} dict, and the
names that had no TOMAS slot and were lumped into the organic slot.
"""
    frac_by_index = {}
    remapped = []
    for name, frac in zip(species_names, mass_fractions):
        index = AEROSOL_SPECIES_MAP.get(name)
        if index is None:
            index = SRTORG1  # conserve mass by lumping into the organic slot
            remapped.append(name)
        frac_by_index[index] = frac_by_index.get(index, 0.0) + float(frac)
    if remapped:
        warnings.warn(
            'tomas_jax: aerosol species %s have no TOMAS-JAX slot; their mass '
            'was lumped into the organic slot, so total mass is conserved but '
            'the speciation is approximate.' % ', '.join(remapped),
            stacklevel = 2,
        )
    return frac_by_index, tuple(remapped)


def lognormal_to_bins(xk, number, geom_mean_diam, log10_geom_std_dev,
                      boxvol, dens = 1770.0):
    """Integrate a single log-normal mode onto TOMAS's mass-doubling grid.

Mirrors TOMAS-JAX's own initializer but takes the parameters an AMBRS
AerosolModeState carries: a geometric mean diameter in metres and the base-10
logarithm of the geometric standard deviation.

Parameters:
    * xk: bin boundary masses [kg], shape (nbins+1,)
    * number: modal number concentration [# cm^-3]
    * geom_mean_diam: geometric mean diameter [m]
    * log10_geom_std_dev: log10 of the geometric standard deviation
    * boxvol: grid-cell volume [cm^3]
    * dens: density relating bin mass to diameter [kg m^-3]

Returns Nk, the number per bin [# per grid cell], shape (nbins,).
"""
    xk = np.asarray(xk, dtype = float)
    nbins = len(xk) - 1
    gsd = 10.0 ** log10_geom_std_dev
    gmd_um = geom_mean_diam * 1e6 # the integral below works in micrometres
    Nk = np.zeros(nbins)
    for k in range(nbins):
        Dl = 1e6 * ((6.0 * xk[k]) / (dens * PI)) ** (1.0 / 3.0)
        Dh = 1e6 * ((6.0 * xk[k + 1]) / (dens * PI)) ** (1.0 / 3.0)
        Dk = np.sqrt(Dl * Dh)
        Nk[k] = ((number * boxvol)
                 / (np.sqrt(2 * PI) * Dk * np.log(gsd))
                 * np.exp(-(np.log(Dk / gmd_um) ** 2 / (2 * np.log(gsd) ** 2)))
                 * (Dh - Dl))
    return Nk


def bin_diameters(xk, dens = 1770.0):
    """The diameter at the centre of each TOMAS bin [m].

TOMAS's grid is defined by bin boundary *masses*, so a size distribution should be
evaluated on the diameters those bins correspond to: sampling more finely than the
model resolves leaves empty bins between populated ones, which shows up as a comb
of spikes rather than a smooth distribution.

Parameters:
    * xk: bin boundary masses [kg], shape (nbins+1,)
    * dens: density relating bin mass to diameter [kg m^-3]

Returns the bin-centre diameters [m], shape (nbins,).
"""
    xk = np.asarray(xk, dtype = float)
    return np.cbrt(np.sqrt(xk[:-1] * xk[1:]) / dens * 6.0 / np.pi)


def distribute_mass(Nk, xk, frac_by_index):
    """Distribute each bin's mass across TOMAS species indices.

Each bin's particles are given the geometric-mean bin mass sqrt(xk[k]*xk[k+1]),
matching TOMAS-JAX's own initializer.

Parameters:
    * Nk: number per bin [# per grid cell], shape (nbins,)
    * xk: bin boundary masses [kg], shape (nbins+1,)
    * frac_by_index: {tomas_index: mass_fraction} for this mode

Returns Mk [kg per grid cell], shape (nbins, ICOMP).
"""
    xk = np.asarray(xk, dtype = float)
    Mk = np.zeros((len(xk) - 1, ICOMP))
    bin_mass = np.asarray(Nk, dtype = float) * np.sqrt(xk[:-1] * xk[1:])
    for index, frac in frac_by_index.items():
        Mk[:, index] += bin_mass * frac
    return Mk


@dataclass
class Input:
    """ambrs.tomas_jax.Input -- an input dataclass for the TOMAS-JAX box model.

TOMAS-JAX has no native input file format, so this dataclass holds the model's
in-memory state arrays directly, together with the atmospheric state, the
timestepping parameters, the active processes, and the precursor concentrations
needed to drive a run.
"""

    # TOMAS state arrays
    Nk: Any            # number per bin [# per grid cell], shape (nbins,)
    Mk: Any            # mass per bin per species [kg per grid cell], (nbins, ICOMP)
    Gc: Any            # gas concentrations [kg per grid cell], (N_GAS_SPECIES,)
    xk: Any            # bin boundary masses [kg], shape (nbins+1,)

    # atmospheric state
    temp: float        # temperature [K]
    pres: float        # pressure [Pa]
    rh: float          # relative humidity [0-1]
    boxvol: float      # grid-cell volume [cm^3]
    alpha: float       # mass accommodation coefficient [-]

    # timestepping
    dt: float          # time step [s]
    nstep: int         # number of time steps

    # active processes, in TOMAS operator-split order (a subset of
    # 'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution')
    processes: tuple[str, ...]

    # nucleation precursors (the ricco_dunne scheme)
    org_conc: float    # condensable organic vapor [molec/cm^3]
    nh3_conc: float    # ammonia [molec/cm^3]
    fion: float        # ion-pair production rate [ion pairs/cm^3/s]
    fn_scale: float    # nucleation rate scale factor [-]

    # H2SO4 gas added to Gc[SRTSO4] each step [molec/cm^3/s]
    h2so4_production: float

    # bookkeeping (not consumed by the solver)
    aerosols: tuple = field(default_factory=tuple)          # scenario aerosol species
    dropped_species: tuple = field(default_factory=tuple)   # species with no TOMAS slot
    scenario: Any = None    # originating Scenario, used to build the Output


class AerosolModel(BaseAerosolModel):
    """ambrs.tomas_jax.AerosolModel -- an in-process AMBRS adapter for TOMAS-JAX.

Create inputs as for any other model, then step them in process with run() (or
run_ensemble() for a whole ensemble) rather than with ambrs.runners.PoolRunner,
which drives external executables.
"""

    def __init__(self,
                 processes: AerosolProcesses,
                 boxvol: float = 1e6,
                 alpha: float = 1.0,
                 density: float = 1770.0,
                 cond_method: str = 'ppm_jit',
                 nucl_scheme: str = 'ricco_dunne',
                 org_conc: float = 0.0,
                 nh3_conc: float = 0.0,
                 fion: float = 0.0,
                 fn_scale: float = 1.0,
                 h2so4_production: float = 0.0):
        """ambrs.tomas_jax.AerosolModel.__init__(processes, ...)

Parameters beyond the AerosolProcesses flags:
    * boxvol: grid-cell volume [cm^3], which scales number and mass to the
      per-cell units TOMAS works in
    * alpha: mass accommodation coefficient [-]
    * density: density relating bin mass to diameter [kg m^-3]
    * cond_method: H2SO4 condensation solver ('ppm_jit', 'tfl_jit', 'ppm', 'tfl')
    * nucl_scheme: nucleation scheme ('ricco_dunne' or 'zhao2024')
    * org_conc, nh3_conc, fion, fn_scale: nucleation precursors [molec cm^-3,
      molec cm^-3, ion pairs cm^-3 s^-1, dimensionless]. An ambrs Scenario
      doesn't carry these, so they are model-level settings; a Scenario NH3
      concentration, if present, overrides nh3_conc.
    * h2so4_production: H2SO4 gas produced each step [molec cm^-3 s^-1]
"""
        if not _TOMAS_AVAILABLE:
            raise ImportError(
                'tomas_jax is not installed, so ambrs.tomas_jax.AerosolModel '
                'cannot be used. Install it with\n'
                '    pip install "tomas-jax @ git+https://github.com/reflective-org/tomas-jax.git@dev"')
        BaseAerosolModel.__init__(self, 'tomas-jax', processes)
        self.boxvol = boxvol
        self.alpha = alpha
        self.density = density
        self.cond_method = cond_method
        self.nucl_scheme = nucl_scheme
        self.org_conc = org_conc
        self.nh3_conc = nh3_conc
        self.fion = fion
        self.fn_scale = fn_scale
        self.h2so4_production = h2so4_production

    def active_processes(self) -> tuple:
        """The TOMAS process names implied by self.processes, in TOMAS's
operator-split order."""
        enabled = {
            'coagulation': self.processes.coagulation,
            'condensation': self.processes.condensation,
            'nucleation': self.processes.nucleation,
        }
        return tuple(p for p in _PROCESS_ORDER if enabled.get(p, False))

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        """ambrs.tomas_jax.AerosolModel.create_input(scenario, dt, nstep) ->
ambrs.tomas_jax.Input describing a TOMAS-JAX simulation of the given scenario.

Every mode is integrated onto TOMAS's mass-doubling grid and its composition
mapped onto TOMAS species indices; the modes are then summed. Gas
concentrations are assumed to be number concentrations [molec cm^-3].

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
                            'used to create tomas-jax input!')

        xk = np.asarray(xk_boundaries(), dtype = float)
        Nk = np.zeros(len(xk) - 1)
        Mk = np.zeros((len(xk) - 1, ICOMP))
        dropped = set()
        for mode in scenario.size.modes:
            fracs, remapped = map_species_fractions(
                tuple(s.name for s in mode.species), mode.mass_fractions)
            dropped.update(remapped)
            # an ambrs mode's number is [# m^-3]; TOMAS bins number in [# cm^-3]
            mode_Nk = lognormal_to_bins(
                xk, mode.number * 1e-6, mode.geom_mean_diam,
                mode.log10_geom_std_dev, self.boxvol, self.density)
            Nk += mode_Nk
            Mk += distribute_mass(mode_Nk, xk, fracs)

        # gas phase: map the gases TOMAS carries into Gc, and let a scenario NH3
        # concentration stand in for the nucleation precursor
        Gc = np.zeros(N_GAS_SPECIES)
        nh3_conc = self.nh3_conc
        for gas, conc in zip(scenario.gases, scenario.gas_concs):
            index = GAS_SPECIES_MAP.get(gas.name)
            if index is not None:
                Gc[index] += _gas_conc_to_kg_per_cell(
                    conc, self.boxvol, _GAS_MOLAR_MASS[gas.name])
            elif gas.name == 'NH3':
                nh3_conc = conc

        return Input(
            Nk = Nk, Mk = Mk, Gc = Gc, xk = xk,
            temp = scenario.temperature,
            pres = scenario.pressure,
            rh = scenario.relative_humidity,
            boxvol = self.boxvol,
            alpha = self.alpha,
            dt = dt,
            nstep = nstep,
            processes = self.active_processes(),
            org_conc = self.org_conc,
            nh3_conc = nh3_conc,
            fion = self.fion,
            fn_scale = self.fn_scale,
            h2so4_production = self.h2so4_production,
            aerosols = tuple(scenario.aerosols),
            dropped_species = tuple(sorted(dropped)),
            scenario = scenario,
        )

    def step_function(self, processes):
        """The compiled TOMAS step for a set of processes.

Compiling is expensive, so steps are cached on the model and reused across every
scenario in an ensemble.
"""
        if not hasattr(self, '_step_cache'):
            self._step_cache = {}
        key = (processes, self.cond_method, self.nucl_scheme)
        if key not in self._step_cache:
            self._step_cache[key] = make_step(
                list(processes),
                cond_method = self.cond_method,
                nucl_scheme = self.nucl_scheme)
        return self._step_cache[key]

    def run(self, input, scenario_name: str = 'tomas-jax') -> Output:
        """ambrs.tomas_jax.AerosolModel.run(input, scenario_name) ->
ambrs.analysis.Output describing the final state of the simulation.

Steps the input's state input.nstep times and converts the result into an
Output, exactly as retrieve_model_state does for the executable models. TOMAS-JAX
runs in process, so there are no output files to read.
"""
        import jax.numpy as jnp

        Nk = jnp.asarray(input.Nk)
        Mk = jnp.asarray(input.Mk)
        Gc = jnp.asarray(input.Gc)
        xk = jnp.asarray(input.xk)

        # TOMAS's step doesn't add the H2SO4 source term itself
        produced = _gas_conc_to_kg_per_cell(
            input.h2so4_production * input.dt, input.boxvol, MW_H2SO4)
        nucleation_args = {}
        if 'nucleation' in input.processes:
            nucleation_args = dict(
                org_conc = input.org_conc, nh3_conc = input.nh3_conc,
                fion = input.fion, fn_scale = input.fn_scale)

        step = self.step_function(input.processes) if input.processes else None
        for _ in range(input.nstep):
            if input.h2so4_production:
                Gc = Gc.at[SRTSO4].add(produced)
            if step is not None:
                Nk, Mk, Gc = step(
                    Nk, Mk, Gc, xk, input.temp, input.pres, input.boxvol,
                    input.rh, input.alpha, input.dt, **nucleation_args)

        return Output(
            model_name = self.name,
            scenario_name = scenario_name,
            scenario = input.scenario,
            timestep = input.nstep,
            particle_population = self.build_population(
                np.asarray(Nk), np.asarray(Mk), np.asarray(xk), input.boxvol),
            gas_mixture = self.build_gas_mixture(
                np.asarray(Gc), input.temp, input.pres, input.boxvol),
            thermodynamics = {'T': input.temp, 'p': input.pres, 'RH': input.rh},
        )

    def build_population(self, Nk, Mk, xk, boxvol):
        """Build a part2pop ParticlePopulation from TOMAS's per-bin state.

Each non-empty bin becomes one particle, whose composition is that bin's mass
across the dry TOMAS species and whose number concentration is Nk/boxvol
converted back to the [# m^-3] the rest of ambrs uses.

TOMAS is a two-moment model, so a bin's mean particle mass is Mk/Nk rather than
the geometric bin mass; taking the diameter from Mk/Nk is what makes the
population's total mass match the model's.
"""
        from part2pop import ParticlePopulation, make_particle
        from part2pop.species.registry import get_species

        # dry species only: part2pop adds water from the ambient humidity
        names = ['SO4', 'OC', 'NH4']
        masses = np.column_stack([
            Mk[:, SRTSO4],
            Mk[:, SRTORG1:SRTNH4].sum(axis = 1), # organics -> a single OC proxy
            Mk[:, SRTNH4],
        ])
        dry_mass = masses.sum(axis = 1)

        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            mean_mass = np.where(Nk > 0.0, dry_mass / Nk, 0.0)
        # (near-)empty bins carry negligible number; fall back to the bin's own
        # geometric mass so their diameter is still well defined
        geometric_mass = np.sqrt(xk[:-1] * xk[1:])
        mean_mass = np.where(mean_mass > 0.0, mean_mass, geometric_mass)
        diameters = np.cbrt(mean_mass / self.density * 6.0 / np.pi) # [m]
        num_concs = (Nk / boxvol) * 1e6                             # [# m^-3]

        # drop species that carry no mass, but always keep sulfate so that a
        # population is never empty
        totals = masses.sum(axis = 0)
        keep = [i for i in range(len(names)) if i == 0 or totals[i] > 0.0]
        names = [names[i] for i in keep]
        masses = masses[:, keep]

        species = tuple(get_species(name, None) for name in names)
        population = ParticlePopulation(
            species = species, spec_masses = [], num_concs = [], ids = [],
            species_modifications = {})
        part_id = 0
        for k in range(len(diameters)):
            if num_concs[k] <= 0.0:
                continue
            total = masses[k].sum()
            fractions = (masses[k] / total) if total > 0.0 \
                else np.eye(len(names))[0] # nominal sulfate for an empty bin
            particle = make_particle(
                diameters[k], species, list(fractions),
                specdata_path = None, species_modifications = {},
                D_is_wet = False)
            part_id += 1
            population.set_particle(particle, part_id, float(num_concs[k]))
        return population

    def build_gas_mixture(self, Gc, temp, pres, boxvol):
        """Build a GasMixture from TOMAS's gas array Gc [kg per grid cell]."""
        R = 8.314462618 # [J mol^-1 K^-1]
        moles_of_air = pres * (boxvol * 1e-6) / (R * temp)
        return build_gas_mixture({
            'SO2': float((Gc[SRTSO2] / (MW_SO2 / 1000.0)) / moles_of_air),
            'H2SO4': float((Gc[SRTSO4] / (MW_H2SO4 / 1000.0)) / moles_of_air),
            'units': 'mole_ratio',
        })

    def write_input_files(self, input, dir: str, prefix: str) -> None:
        """ambrs.tomas_jax.AerosolModel.write_input_files(input, dir, prefix)

TOMAS-JAX reads no input files, so this records the state a run started from,
for provenance and so a run can be reproduced later: the arrays go to
<dir>/<prefix>.npz and the scalar settings to a readable <dir>/<prefix>.txt. The
originating Scenario is not serialized."""
        if not os.path.exists(dir):
            raise OSError(f'Directory not found: {dir}')
        np.savez(
            os.path.join(dir, prefix + '.npz'),
            Nk = input.Nk, Mk = input.Mk, Gc = input.Gc, xk = input.xk,
            temp = input.temp, pres = input.pres, rh = input.rh,
            boxvol = input.boxvol, alpha = input.alpha,
            dt = input.dt, nstep = input.nstep,
            processes = np.asarray(input.processes),
            org_conc = input.org_conc, nh3_conc = input.nh3_conc,
            fion = input.fion, fn_scale = input.fn_scale,
            h2so4_production = input.h2so4_production,
            dropped_species = np.asarray(input.dropped_species),
        )
        with open(os.path.join(dir, prefix + '.txt'), 'w') as f:
            f.write('# generated by ambrs.tomas_jax.AerosolModel.write_input_files\n')
            f.write(f'processes = {list(input.processes)}\n')
            for name in ('temp', 'pres', 'rh', 'boxvol', 'alpha', 'dt', 'nstep',
                         'org_conc', 'nh3_conc', 'fion', 'fn_scale',
                         'h2so4_production'):
                f.write(f'{name} = {getattr(input, name)}\n')
            f.write(f'dropped_species = {list(input.dropped_species)}\n')

    def read_input(self, dir: str, prefix: str, scenario = None) -> Input:
        """ambrs.tomas_jax.AerosolModel.read_input(dir, prefix) ->
ambrs.tomas_jax.Input reconstructed from the files write_input_files wrote.

The originating Scenario isn't stored in the files, so pass it here if the
resulting Input is to be used to build an Output."""
        filename = os.path.join(dir, prefix + '.npz')
        if not os.path.exists(filename):
            raise OSError(f'TOMAS-JAX input file not found: {filename}')
        d = np.load(filename)
        text = lambda key: tuple(str(s) for s in d[key].tolist()) \
            if d[key].size else ()
        return Input(
            Nk = d['Nk'], Mk = d['Mk'], Gc = d['Gc'], xk = d['xk'],
            temp = float(d['temp']), pres = float(d['pres']),
            rh = float(d['rh']), boxvol = float(d['boxvol']),
            alpha = float(d['alpha']), dt = float(d['dt']),
            nstep = int(d['nstep']), processes = text('processes'),
            org_conc = float(d['org_conc']), nh3_conc = float(d['nh3_conc']),
            fion = float(d['fion']), fn_scale = float(d['fn_scale']),
            h2so4_production = float(d['h2so4_production']),
            dropped_species = text('dropped_species'),
            scenario = scenario,
        )

    def run_ensemble(self, inputs: list) -> list:
        """ambrs.tomas_jax.AerosolModel.run_ensemble(inputs) -> list of Outputs

Runs each input in process, in order, reusing a single compiled step across the
whole ensemble. Scenarios are named by their 1-based index, as PoolRunner names
its scenario directories."""
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of scenario inputs')
        width = len(str(len(inputs)))
        return [self.run(input, scenario_name = f'{i + 1:0{width}d}')
                for i, input in enumerate(inputs)]

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'tomas-jax runs in process and has no executable to invoke; step it '
            'with AerosolModel.run() or run_ensemble() instead of PoolRunner.')


def retrieve_model_state(
        scenario_name: str,
        scenario: Scenario,
        timestep: int,
        processes: AerosolProcesses = None,
        ensemble_output_dir: str = 'tomas_jax_runs',
        **model_options) -> Output:
    """ambrs.tomas_jax.retrieve_model_state(scenario_name, scenario, timestep, ...)
-> ambrs.analysis.Output describing the state of a TOMAS-JAX simulation.

This mirrors partmc.retrieve_model_state and mam4.retrieve_model_state, but
because TOMAS-JAX runs in process there is no output on disk to read: the state
recorded by write_input_files in <ensemble_output_dir>/<scenario_name>/ is loaded
and stepped forward to `timestep`. Running a scenario directly with
AerosolModel.run() is the cheaper path when the Input is still in hand.

Parameters:
    * scenario_name: names the scenario's directory and the resulting Output
    * scenario: the Scenario the run came from, recorded in the Output
    * timestep: the number of steps to advance (1 gives the initial state)
    * processes: the AerosolProcesses to run with; taken from the recorded input
      when omitted
    * ensemble_output_dir: the directory holding the per-scenario directories
    * model_options: forwarded to AerosolModel (boxvol, density, ...)"""
    if timestep < 1:
        raise ValueError('timestep must be positive; 1 gives the initial state')
    model = AerosolModel(processes or AerosolProcesses(), **model_options)
    dir = os.path.join(ensemble_output_dir, scenario_name)
    input = model.read_input(dir, scenario_name, scenario = scenario)
    if processes is None:
        # honour whatever the recorded run was configured with
        model.processes = AerosolProcesses(
            coagulation = 'coagulation' in input.processes,
            condensation = 'condensation' in input.processes,
            nucleation = 'nucleation' in input.processes,
        )
    else:
        input.processes = model.active_processes()
    input.nstep = timestep - 1 # timestep 1 == initial conditions
    return model.run(input, scenario_name = scenario_name)
