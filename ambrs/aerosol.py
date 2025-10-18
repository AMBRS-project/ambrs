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

@dataclass(frozen=False) # Unfroze. Reason for frozen?
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
specific parameters (no state information)"""
    name: str                   # name of the species
    molar_mass: float           # [g/mol]
    density: float              # density [kg/m^3]
    ions_in_soln: int = 0       # number of ions in solution [-]
    hygroscopicity: float = 0.0 # "kappa" [-]
    aliases: Optional[tuple[str, ...]] = None # tuple of alternative species names

    def __post_init__(self): # checks initialized fields
        # check the name of the species
        valid_aerosol_species = [
            'SO4',
            'NO3',
            'Cl',
            'NH4',
            'MSA',
            'ARO1',
            'ARO2',
            'ALK1',
            'OLE1',
            'API1',
            'API2',
            'LIM1',
            'LIM2',
            'CO3',
            'Na',
            'Ca',
            'OIN',
            'OC',
            'BC',
            'H2O',
        ]
        
        # if self.name not in valid_aerosol_species:
        #     raise NameError(f'Invalid aerosol species name: {self.name}\nValid names are {valid_aerosol_species}')
        if self.molar_mass <= 0.0:
            raise ValueError(f'Non-positive molar mass: {self.molar_mass}')
        if self.density <= 0.0:
            raise ValueError(f'Non-positive density: {self.density}')
        if self.ions_in_soln > 0.0 and self.hygroscopicity > 0.0:
            raise ValueError('Only one of ions_in_soln and hygroscopicity may be given!')

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
        #print(self.species)
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
    log10_geom_std_dev: RVFrozenDistribution #float                        # mode-specific logarithmic diameter std dev # FIXME: changed to RV
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
    

#----------------------------------------
# Aerosol Conversion Utility Functions
#----------------------------------------

def vol_to_num(v, Dgn, sigmag):
    '''
    Convert volume concentration to number concentration based on lognormal mode statistics.
    '''
    # n = 6*m / (np.pi * rho * Dgn**3 * np.exp(4.5 * np.log(sigmag)**2))
    n = v / (Dgn**3 * (np.pi / 6.0) * np.exp(4.5*(np.log(sigmag))**2))
    return n
