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
    _TOMAS_AVAILABLE = True
except (ImportError, OSError):
    _TOMAS_AVAILABLE = False


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
                 processes: AerosolProcesses):
        if not _TOMAS_AVAILABLE:
            raise ImportError(
                'tomas_jax is not installed, so ambrs.tomas_jax.AerosolModel '
                'cannot be used. Install it with\n'
                '    pip install "tomas-jax @ git+https://github.com/reflective-org/tomas-jax.git@dev"')
        BaseAerosolModel.__init__(self, 'tomas-jax', processes)

    def create_input(self,
                     scenario: Scenario,
                     dt: float,
                     nstep: int) -> Input:
        raise NotImplementedError('tomas_jax.AerosolModel.create_input not yet implemented!')

    def run(self, input, scenario_name: str = 'tomas-jax') -> Output:
        raise NotImplementedError('tomas_jax.AerosolModel.run not yet implemented!')

    def write_input_files(self, input, dir: str, prefix: str) -> None:
        raise NotImplementedError('tomas_jax.AerosolModel.write_input_files not yet implemented!')

    def invocation(self, exe: str, prefix: str) -> str:
        raise NotImplementedError(
            'tomas-jax runs in process and has no executable to invoke; step it '
            'with AerosolModel.run() or run_ensemble() instead of PoolRunner.')
