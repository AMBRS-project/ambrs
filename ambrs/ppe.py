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
from math import log10, pow
from typing import Optional
from .aerosol import \
    AerosolModalSizeState, AerosolModalSizeDistribution, AerosolModalSizePopulation,\
    RVFrozenDistribution
from .scenario import Scenario

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
    specification: Optional[EnsembleSpecification] # if used for creation
    size: AerosolModalSizePopulation
    flux: np.array
    relative_humidity: np.array
    temperature: np.array

    def __len__(self):
        return len(ensemble.size)

    def __iter__(self):
        for i in range(len(self)):
            yield self.member(i)

    def member(self, i: int) -> Scenario:
        """ensemble.member(i) -> extracts Scenario from ith ensemble member"""
        return Scenario(
            size = self.size.member(i),
            flux = self.flux[i],
            relative_humidity = self.relative_humidity[i],
            temperature = self.temperature[i])

#------------------------------------------------
# Ensembles constructed by aggregating scenarios
#------------------------------------------------

def ensemble_from_scenarios(scenarios: list[Scenario]):
    """ensemble_from_scenarios(scenarios) -> ensemble consisting of exactly the
specified scenarios (which must all have the same particle size representation)"""
    n = len(scenarios)
    if n == 0:
        raise ValueError("No scenarios provided for ensemble!")

    # assemble size-independent data
    flux = np.array(n)
    flux[:] = [scenario.flux for scenario in scenarios]
    relative_humidity = np.array(n)
    relative_humidity[:] = [scenario.relative_humidity for scenario in scenarios]
    temperature = np.array(n)
    temperature[:] = [scenario.temperature for scenario in scenarios]

    # handle particle size data
    if instance(scenarios[0].size, AerosolModalSizeState):
        modes=[]
        for mode in scenarios[0].size.modes:
            modes.append(AerosolModePopulation(
                name = mode.name,
                species = mode.species,
                number = np.array(n),
                geom_mean_diam = np.array(n),
                mass_fractions = tuple([np.array(n) for mf in mode.mass_fractions])))
        for m, mode in enumerate(modes):
            mode.number[:] = [scenario.modes[m].number for scenario in scenarios]
            mode.geom_mean_diam[:] = [scenario.modes[m].geom_mean_diam for scenario in scenarios]
            mode.mass_fractions[:] = [mf for mf in scenario.modes[m].mass_fractions for scenario in scenarios]
        return Ensemble(
            size = AerosolModalSizePopulation(
                modes = tuple(modes),
            ),
            flux = flux,
            relative_humidity = relative_humidity,
            temperature = temperature,
        )
    else:
        raise TypeError("Invalid particle size information in scenarios!")

#-------------------------------------------------
# Ensembles constructed by sampling distributions
#-------------------------------------------------

def sample(specification: EnsembleSpecification, n: int) -> Ensemble:
    """sample(spec, n) -> n-member ensemble sampled from a specification"""
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
    """lhs(specification, n, [criterion, iterations]) -> n-member ensemble
generated from latin hypercube sampling applied to the given specification. The
optional arguments are passed along to pyDOE's lhs function, which creates the
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
                    number=mode.number.ppf(lhd[:,(2+num_species)*m]),
                    geom_mean_diam=mode.geom_mean_diam.ppf(lhd[:,(2+num_species+1)*m]),
                    mass_fractions=tuple(
                        [mass_fraction.ppf(lhd[:,(2*num_species+2)*m+f])
                         for f, mass_fraction in enumerate(mode.mass_fractions)]),
                ) for m, mode in enumerate(spec.size.modes)]),
        )
    return Ensemble(
        specification = spec,
        size = modal_size,
        flux = spec.flux.rvs(n),
        relative_humidity = spec.relative_humidity.ppf(lhd[:,-2]),
        temperature = spec.temperature.ppf(lhd[:,-1]),
    )

#---------------------------
# Swept-parameter ensembles
#---------------------------

@dataclass
class LinearParameterSweep:
    """LinearParameterSweep(a, b, n) - an n-step, uniformly-spaced sweep in the
parameter range [a, b]"""
    a: float
    b: float
    n: int

    def __len__(self): # `len(sweep)` returns number of values swept
        return self.n

    def __iter__(self): # `for p in sweep` allows iteration over assumed values
        assert(self.b - self.a > 0)
        step = (self.b - self.a)/self.n
        for i in range(self.n):
            yield self.a + i*step

@dataclass
class LogarithmicParameterSweep:
    """LogarithmicParameterSweep(a, b, n) - an n-step, log10-spaced sweep in the
parameter range [a, b]"""
    a: float
    b: float
    n: int

    def __len__(self): # returns number of values swept
        return self.n

    def __iter__(self): # allows iteration over assumed values
        assert(self.b - self.a > 0)
        log_a = log10(self.a)
        log_b = log10(self.b)
        step = (log_b - log_a)/step.n
        for i in range(self.n):
            yield pow(10, log_a + i*step)

@dataclass
class AerosolModeParameterSweep:
    """AerosolModeParameterSweep: a set of parameter sweep ranges for a single,
internally-mixed aerosol mode"""
    number: Optional[LinearParameterSweep | LogarithmicParameterSweep]
    geom_mean_diam: Optional[LinearParameterSweep | LogarithmicParameterSweep]
    mass_fraction: Optional[tuple[LinearParameterSweep | LogarithmicParameterSweep, ...]]

@dataclass
class AerosolModalSizeParameterSweep:
    """AerosolModalSizeParameterSweep: a set of parameter sweep ranges for modal
aerosol particle size"""
    modes: Optional[tuple[AerosolModeParameterSweep, ...]]

@dataclass
class ParameterSweep:
    """ParameterSweep: a set of aerosol parameter "sweep" ranges used by the
sweep function to construct an ensemble from a reference state.

Each sweep range is ultimately specified by one of the following:
* LinearParameterSweep(a, b, step) - a uniformly-spaced sweep in the parameter
  range [a, b]
* LogarithmicParameterSweep(a, b, step) - a logarithmically-spaced sweep in
  the parameter range [a, b]

If no sweep is specified for a given parameter, that parameter assumes a
value specified by the reference_state passed to the sweep function.
"""
    size: Optional[AerosolModalSizeParameterSweep]
    flux: Optional[LinearParameterSweep | LogarithmicParameterSweep]
    relative_humidity: Optional[LinearParameterSweep | LogarithmicParameterSweep]
    temperature: Optional[LinearParameterSweep | LogarithmicParameterSweep]

def sweep(reference_state: Scenario, sweeps: ParameterSweep) -> Ensemble:
    """sweep(reference_state, sweeps) -> ensemble generated by initializing a
"reference state" and performing sweeps for the parameters specified in the
given ParameterSweep. The size of the ensemble is determined by the specified
parameter sweeps"""
    if isinstance(sweeps.size, AerosolModalSizeParameterSweep):
        pass # TODO:
