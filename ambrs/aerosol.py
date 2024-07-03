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

#----------------------------
# Modal aerosol descriptions
#----------------------------

@dataclass
class AerosolModeState:
    """AerosolModeState: the state information for a single internally-mixed,
log-normal aerosol mode"""
    name: str
    species: tuple[AerosolSpecies, ...]
    number: float                     # modal number concentration
    geom_mean_diam: float             # geometric mean diameter
    mass_fractions: tuple[float, ...] # species mass fractions

@dataclass(frozen=True)
class AerosolModeDistribution:
    """AerosolModeDistribution: the definition of a single internally-mixed,
log-normal aerosol mode (distribution only--no state information)"""
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
    name: str
    species: tuple[AerosolSpecies, ...]
    number: np.array
    geom_mean_diam: np.array
    mass_fractions: tuple[np.array, ...] # species-specific mass fractions

    def member(i: int) -> AerosolModeState:
        """population.member(i) -> extracts mode state information from ith
population member"""
        return AerosolModeState(
            name = self.name,
            species = self.species,
            number = self.number[i],
            geom_mean_diam = self.geom_mean_diam[i],
            mass_fractions = tuple([mass_frac[i] for mass_frac in self.mass_fractions]))

@dataclass
class AerosolModalSizeState:
    """AerosolModalSizeState: state information for modal aerosol particle size,
specified in terms of a fixed number of log-normal modes"""
    modes: tuple[AerosolModeState, ...]

@dataclass(frozen=True)
class AerosolModalSizeDistribution:
    """AerosolModalSizeDistribution: an aerosol particle size distribution
specified in terms of a fixed number of log-normal modes"""
    modes: tuple[AerosolModeDistribution, ...]
    
@dataclass
class AerosolModalSizePopulation:
    """AerosolModalSizePopulation: an aerosol population sampled from a specific
modal particle size distribution"""
    modes: tuple[AerosolModePopulation, ...]

    def member(i: int) -> AerosolModalSizeState:
        """population.member(i) -> extracts size state information from ith
population member"""
        return AerosolModalSizeState(
            modes = tuple([mode.member(i) for mode in self.modes]))
