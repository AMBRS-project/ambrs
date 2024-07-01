"""aerosol - aerosol species and particle size distribution machinery

We use scipy.stats distributions (specifically the rv_continuous ones) for
sampling. Because scipy provides an extensible framework for sampling, any
type of sampling process (including sampling from ESMs like E3SM) is possible.
"""

import numpy as np
import scipy.stats

from dataclasses import dataclass

@dataclass
class AerosolProcesses:
    """AerosolProcesses: a definition of a set of aerosol processes under consideration"""
    # processes that can be enabled/disabled
    aging: bool
    coagulation: bool
    mosaic: bool
    nucleation: bool

@dataclass
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
specific parameters (no state information)"""
    name: str          # name of the species
    molar_mass: float  # etc

@dataclass
class AerosolModeDistribution:
    """AerosolModeDistribution: the definition of a single internally-mixed,
log-normal aerosol mode (configuration only--no state information)"""
    name: str
    species: tuple[AerosolSpecies]
    number: scipy.stats.rv_continuous                # modal number concentration distribution
    geom_mean_diam: scipy.stats.rv_continuous        # geometric mean diameter distribution
    mass_fractions: tuple[scipy.stats.rv_continuous] # species mass fraction distributions

@dataclass
class AerosolModePopulation:
    """AerosolModePopulation: a particle population representing a single
internally-mixed, log-normal aerosol mode, sampled from a specific mode
distribution"""
    number: np.array
    geom_mean_diam: np.array
    mass_fractions: np.array

@dataclass
class AerosolModalSizeDistribution:
    """AerosolModalSizeDistribution: an aerosol particle size distribution
specified in terms of a fixed number of log-normal modes"""
    modes: tuple[AerosolModeDistribution]
    
@dataclass
class AerosolModalSizePopulation:
    """AerosolModalSizePopulation: an aerosol population sampled from a specific
modal particle size distribution"""
    modes: tuple[AerosolModePopulation]
