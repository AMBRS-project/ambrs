"""ambrs.ppe - types and functions supporting the creation of perturbed-parameter
ensembles (PPE).

We use frozen scipy.stats distributions (specifically the rv_continuous ones)
for sampling. These frozen distributions fix the shape parameters, location, and
scale factors for each random variable.
"""

import numpy as np
import pyDOE
import scipy.stats

from dataclasses import dataclass
from .aerosol import AerosolModalSizeDistribution, AerosolModalSizePopulation,\
                     RVFrozenDistribution

@dataclass(frozen=True)
class EnsembleSpecification:
    """EnsembleSpecification: a set of distributions from which members of a
PPE are sampled"""
    name: str
    size: AerosolModalSizeDistribution
    flux: RVFrozenDistribution
    relative_humidity: RVFrozenDistribution
    temperature: RVFrozenDistribution

@dataclass(frozen=True)
class Ensemble:
    """Ensemble: an ensemble defined by values sampled from the distributions of
a specific EnsembleSpecification"""
    specification: EnsembleSpecification  # used to construct Ensemble
    size: AerosolModalSizePopulation
    flux: np.array
    relative_humidity: np.array
    temperature: np.array

@dataclass
class Scenario:
    size: AerosolModalSizeScenario
    flux: float
    relative_humidity: float
    temperature: float

def sample(specification: EnsembleSpecification, n: int) -> Ensemble:
    """sample(spec, n) -> n-member ensemble sampled from spec"""
    modal_size = None
    if isinstance(specification.size, AerosolModalSizeDistribution):
        modal_size = AerosolModalSizePopulation(
            modes=tuple([
                AerosolModePopulation(
                    number=mode.number.rvs(n),
                    geom_mean_diam=mode.geom_mean_diam.rvs(n),
                    mass_fractions=tuple([f.rvs(n) for f in mode.mass_fractions]),
                ) for mode in specification.size.modes]),
        )
    return Ensemble(
        specification = specification,
        size = modal_size,
        flux = spec.flux.rvs(n),
        relative_humidity = spec.relative_humidity.rvs(n),
        temperature = spec.temperature.rvs(n),
    )

def lhs(specification: EnsembleSpecification,
        n: int,
        criterion = None,
        iterations = None) -> Ensemble:
    """lhs(spec, n, [criterion, iterations]) -> n-member ensemble generated
from latin hypercube sampling applied to the given specification. The optional
arguments are passed along to pyDOE's lhs function, which creates the
distribution from which ensemble members are sampled."""
    n_factors = 3 # size-independent factors: flux + relative_humidity + temperature
    lhd = None # latin hypercube distribution (created depending on particle
               # size representation)
    modal_size = None
    if isinstance(spec.size, AerosolModalSizeDistribution):
        # count up mode-related factors
        for mode in spec.size.modes:
            n_factors += 2 + len(mode.mass_fractions)
        # lhd is a 2D array with indices (sample index, factor index)
        lhd = pyDOE.lhs(n_factors, n, criterion, iterations)
        num_species = len(spec.size.species)
        modal_size = AerosolModalSizePopulation(
            modes=tuple([
                AerosolModePopulation(
                    number=mode.number.pdf(lhd[:,(2+num_species)*m]),
                    geom_mean_diam=mode.geom_mean_diam.pdf(lhd[:,(2+num_species+1)*m]),
                    mass_fractions=tuple(
                        [mass_fraction.pdf(lhd[:,(2*num_species+2)*m+f])
                         for f,mass_fraction in enumerate(mode.mass_fractions)]),
                ) for m, mode in enumerate(spec.size.modes)]),
        )
    return Ensemble(
        specification = spec,
        size = modal_size,
        flux = spec.flux.rvs(n),
        relative_humidity = spec.relative_humidity.pdf(lhd[:,-2]),
        temperature = spec.temperature.pdf(lhd[:,-1]),
    )
