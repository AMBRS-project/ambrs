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
        NBINS, ICOMP, N_GAS_SPECIES,
        SRTSO4, SRTSO2, SRTNH4, SRTH2O, SRTORG1, IORG,
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

    def run(self, input, scenario_name: str = 'tomas-jax') -> Output:
        raise NotImplementedError('tomas_jax.AerosolModel.run not yet implemented!')

    def write_input_files(self, input, dir: str, prefix: str) -> None:
        raise NotImplementedError('tomas_jax.AerosolModel.write_input_files not yet implemented!')

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'tomas-jax runs in process and has no executable to invoke; step it '
            'with AerosolModel.run() or run_ensemble() instead of PoolRunner.')
