"""aerosol - aerosol species and particle size distribution machinery

We use frozen scipy.stats distributions (specifically the rv_continuous ones)
for sampling. These frozen distributions fix the shape parameters, location, and
scale factors for each random variable.
"""

import numpy as np
import scipy.stats

from dataclasses import dataclass
from typing import TypeVar

# this type represents a frozen scipy.stats.rv_continous distribution
# (this frozen type isn't made available by the scipy.stats package)
RVFrozenDistribution = TypeVar('RVFrozenDistribution')

@dataclass
class AerosolProcesses:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    # processes that can be enabled/disabled
    aging: bool = False
    coagulation: bool = False
    mosaic: bool = False
    nucleation: bool = False

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
specific parameters (no state information)"""
    name: str          # name of the species
    molar_mass: float  # etc

@dataclass(frozen=True)
class AerosolModeDistribution:
    """AerosolModeDistribution: the definition of a single internally-mixed,
log-normal aerosol mode (configuration only--no state information)"""
    name: str
    species: tuple[AerosolSpecies, ...]
    number: RVFrozenDistribution                     # modal number concentration distribution
    geom_mean_diam: RVFrozenDistribution             # geometric mean diameter distribution
    mass_fractions: tuple[RVFrozenDistribution, ...] # species mass fraction distributions

@dataclass
class AerosolModePopulation:
    """AerosolModePopulation: a particle population representing a single
internally-mixed, log-normal aerosol mode, sampled from a specific mode
distribution"""
    number: np.array
    geom_mean_diam: np.array
    mass_fractions: tuple[np.array, ...] # species-specific mass fractions

@dataclass
class AerosolModalSizeDistribution:
    """AerosolModalSizeDistribution: an aerosol particle size distribution
specified in terms of a fixed number of log-normal modes"""
    modes: tuple[AerosolModeDistribution, ...]
    
@dataclass
class AerosolModalSizePopulation:
    """AerosolModalSizePopulation: an aerosol population sampled from a specific
modal particle size distribution"""
    modes: tuple[AerosolModePopulation, ...]
