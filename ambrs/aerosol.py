"""aerosol - aerosol species and particle size distribution machinery

We use frozen scipy.stats distributions (specifically the rv_continuous ones)
for sampling. These frozen distributions fix the shape parameters, location, and
scale factors for each random variable.
"""

import numpy as np
import scipy.stats

from dataclasses import dataclass
from typing import Optional, TypeVar

# this type represents a frozen scipy.stats.rv_continous distribution
# (this frozen type isn't made available by the scipy.stats package)
RVFrozenDistribution = TypeVar('RVFrozenDistribution')

@dataclass
class AerosolProcesses:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    # processes that can be enabled/disabled
    aging: bool = False
    aqueous_chemistry: bool = False
    coagulation: bool = False
    condensation: bool = False
    gas_phase_chemistry: bool = False
    optics: bool = False
    nucleation: bool = False

@dataclass(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
specific parameters (no state information)"""
    name: str                   # name of the species
    molar_mass: float           # [g/mol]
    density: float              # density [kg/m^3]
    ions_in_soln: int = 0       # number of ions in solution [-]
    hygroscopicity: float = 0.0 # "kappa" [-]
    aliases: Optional[tuple[str, ...]] = None # tuple of alternative species names

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
    log10_geom_std_dev: float         # log10 of geometric std dev of diameter
    mass_fractions: tuple[float, ...] # species mass fractions

    def mass_fraction(self, species_name: str):
        """returns the mass fraction corresponding to the given species name,
or throws a ValueError."""
        index = [s.name for s in self.species].index(species_name)
        return self.mass_fractions[index]

@dataclass(frozen=True)
class AerosolModeDistribution:
    """AerosolModeDistribution: the definition of a single internally-mixed,
log-normal aerosol mode (distribution only--no state information)"""
    name: str
    species: tuple[AerosolSpecies, ...]
    number: RVFrozenDistribution                     # modal number concentration distribution
    geom_mean_diam: RVFrozenDistribution             # geometric mean diameter distribution
    log10_geom_std_dev: float                        # mode-specific logarithmic diameter std dev
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
    log10_geom_std_dev: np.array
    mass_fractions: tuple[np.array, ...] # species-specific mass fractions

    def __len__(self) -> int:
        return len(self.number)

    def __iter__(self) -> AerosolModeState: # for modal state in mode population
        for i in range(len(self.number)):
            yield self.member(i)

    def member(self, i: int) -> AerosolModeState:
        """population.member(i) -> extracts mode state information from ith
population member"""
        return AerosolModeState(
            name = self.name,
            species = self.species,
            number = self.number[i],
            geom_mean_diam = self.geom_mean_diam[i],
            log10_geom_std_dev = self.log10_geom_std_dev[i],
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

    def __len__(self) -> int:
        return len(self.modes[0])

    def __iter__(self) -> AerosolModalSizeState: # for modal size state in population
        for i in range(len(self.modes[0].number)):
            yield AerosolModalSizeState(
                modes = tuple([mode.member(i) for mode in self.modes]))

    def member(self, i: int) -> AerosolModalSizeState:
        """population.member(i) -> extracts size state information from ith
population member"""
        return AerosolModalSizeState(
            modes = tuple([mode.member(i) for mode in self.modes]))
