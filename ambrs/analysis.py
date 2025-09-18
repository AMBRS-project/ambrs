"""ambrs.analysis - functions and tools for analyzing output from box models
"""

from dataclasses import dataclass
import numpy as np
import scipy.stats
import pandas as pd
from typing import Optional, Dict, List

# PyParticle public API
from PyParticle import ParticlePopulation  # type: ignore
from PyParticle.optics.builder import build_optical_population  # type: ignore
from PyParticle import analysis as ppa  # type: ignore

from .gas import GasMixture
from .scenario import Scenario

###############################################################################
# NOTE FOR DEVELOPERS
# --------------------
# This module previously contained bespoke implementations of size distribution,
# CCN, and optical coefficient calculations. Those routines have been removed
# in favor of delegating all scientific variable computation to
# PyParticle.analysis.compute_variable ("ppa.compute_variable").
#
# Contracts relied upon here:
#   ppa.compute_variable(population, varname, var_cfg) -> dict with:
#       - one or more axis arrays (e.g. 'D', 's', 'wvls', 'rh_grid')
#       - a main data array under key = varname (e.g. 'dNdlnD','Nccn','b_ext')
# Any failure (missing files, unknown variable) should raise a clear exception;
# we DO NOT fabricate fallback data.
###############################################################################


@dataclass
class Output:
    model_name: str
    scenario_name: str
    scenario: Scenario
    # time: float
    timestep: int
    particle_population: ParticlePopulation
    gas_mixture: GasMixture
    thermodynamics: dict
    environment: dict = None
    species_modifications: dict = None
    # diagnostics: Optional(dict)
    def compute_variable(self, varname: str, var_cfg: Optional[Dict] = None) -> Dict:
        """Delegate variable computation to PyParticle.analysis.

        Parameters
        ----------
        varname : str
            Canonical variable name (e.g. 'dNdlnD', 'Nccn', 'b_ext').
        var_cfg : dict, optional
            Configuration overrides (axes, ranges, etc.).
        """
        return ppa.compute_variable(
            population=self.particle_population,  # PyParticle >= latest uses keyword 'population'
            varname=varname,
            var_cfg=var_cfg or {}
        )

def nmae(output_list1: List[Output], output_list2: List[Output], varname: str, var_cfg: Optional[Dict] = None) -> float:
    """Normalized mean absolute error between two lists of Output objects.

    All outputs are collapsed (ravel) across their returned arrays for the
    specified variable.
    """
    var_cfg = var_cfg or {}
    x1 = []
    x2 = []
    for o1, o2 in zip(output_list1, output_list2):
        d1 = o1.compute_variable(varname, var_cfg)
        d2 = o2.compute_variable(varname, var_cfg)
        v1 = d1.get(varname, d1)
        v2 = d2.get(varname, d2)
        x1.append(np.ravel(v1))
        x2.append(np.ravel(v2))
    if not x1:
        return np.nan
    a1 = np.concatenate(x1)
    a2 = np.concatenate(x2)
    denom = np.sum(np.abs(a1))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(a2 - a1)) / denom)

# fixme: generalize kl_divergence to distribution wrt to any independent variable?
# fixme: kl_divergence is comaprison between single time-step for single scenario
def kl_divergence(output1: Output, output2: Output, var_cfg: Optional[Dict] = None) -> float:
    """KL-divergence between two size distributions using PyParticle dispatcher.

    Parameters
    ----------
    output1, output2 : Output
        Model outputs at the same timestep / scenario.
    var_cfg : dict, optional
        Configuration overrides for 'dNdlnD'. Recognized extra key:
          * 'backward' (bool): if True, compute D_KL(P2 || P1) instead of P1||P2.
    """
    var_cfg = (var_cfg or {}).copy()
    backward = bool(var_cfg.pop('backward', False))
    d1 = output1.compute_variable('dNdlnD', var_cfg)
    d2 = output2.compute_variable('dNdlnD', var_cfg)
    v1 = np.asarray(d1.get('dNdlnD', d1))
    v2 = np.asarray(d2.get('dNdlnD', d2))
    s1 = np.sum(v1)
    s2 = np.sum(v2)
    if s1 <= 0 or s2 <= 0:
        return np.nan
    P1 = v1 / s1
    P2 = v2 / s2
    return float(scipy.stats.entropy(P2, P1) if backward else scipy.stats.entropy(P1, P2))

