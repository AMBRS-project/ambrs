"""ambrs.ppe - types and functions supporting the creation of perturbed-parameter
ensembles (PPE).

We use frozen scipy.stats distributions (specifically the rv_continuous ones)
for sampling. These frozen distributions fix the shape parameters, location, and
scale factors for each random variable.
"""

import numpy as np
import pyDOE
import scipy.stats
import itertools # for cartesian products of parameter sweeps

from dataclasses import dataclass
from math import log10, pow
from typing import Optional
from .aerosol import \
    AerosolModalSizeState, AerosolModeState, AerosolModePopulation, \
    AerosolModalSizeDistribution, AerosolModalSizePopulation, AerosolSpecies, \
    RVFrozenDistribution
from .gas import GasSpecies
from .scenario import Scenario

@dataclass(frozen=True)
class EnsembleSpecification:
    """EnsembleSpecification: a set of distributions from which members of a
PPE are sampled"""
    name: str
    aerosols: tuple[AerosolSpecies, ...]
    gases: tuple[GasSpecies, ...]
    size: AerosolModalSizeDistribution
    gas_concs: tuple[RVFrozenDistribution, ...] # ordered like gases
    flux: RVFrozenDistribution
    relative_humidity: RVFrozenDistribution
    temperature: RVFrozenDistribution
    pressure: RVFrozenDistribution #float # <-- these are fixed per ensemble # !!! FIXME: Changed from float to RV
    height: float   # <--
    gas_emissions: Optional[list[tuple[float, dict], ...]] = None
    gas_background: Optional[list[tuple[float, dict], ...]] = None

@dataclass(frozen=True)
class Ensemble:
    """Ensemble: an ensemble defined by values sampled from the distributions of
a specific EnsembleSpecification"""
    aerosols: tuple[AerosolSpecies, ...]
    gases: tuple[GasSpecies, ...]
    size: AerosolModalSizePopulation
    gas_concs: tuple[np.array, ...] # ordered like gases
    flux: np.array
    relative_humidity: np.array
    temperature: np.array
    pressure: np.array
    height: float
    gas_emissions: Optional[list[tuple[float, dict], ...]] = None
    gas_background: Optional[list[tuple[float, dict], ...]] = None
    specification: Optional[EnsembleSpecification] = None # if used for creation

    def __len__(self):
        return len(self.size)

    def __iter__(self):
        for i in range(len(self)):
            yield self.member(i)

    def member(self, i: int) -> Scenario:
        """ensemble.member(i) -> extracts Scenario from ith ensemble member"""
        return Scenario(
            aerosols = self.aerosols,
            gases = self.gases,
            size = self.size.member(i),
            gas_concs = tuple([conc[i] for conc in self.gas_concs]),
            flux = self.flux[i],
            relative_humidity = self.relative_humidity[i],
            temperature = self.temperature[i],
            pressure = self.pressure[i], # FIXME: indexed pressure like prev two vars
            height = self.height,
            gas_emissions = self.gas_emissions,
            gas_background = self.gas_background,
        )

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
    flux = np.array([scenario.flux for scenario in scenarios])
    relative_humidity = np.array([scenario.relative_humidity for scenario in scenarios])
    temperature = np.array([scenario.temperature for scenario in scenarios])
    pressure = np.array([scenario.pressure for scenario in scenarios])

    # handle particle size data
    size = None
    if isinstance(scenarios[0].size, AerosolModalSizeState):
        modes=[]
        for m, mode in enumerate(scenarios[0].size.modes):
            num_species = len(mode.species)
            modes.append(AerosolModePopulation(
                name = mode.name,
                species = mode.species,
                number = np.array([scenario.size.modes[m].number \
                                   for scenario in scenarios]),
                geom_mean_diam = np.array([scenario.size.modes[m].geom_mean_diam \
                                           for scenario in scenarios]),
                log10_geom_std_dev = np.array([scenario.size.modes[m].log10_geom_std_dev \
                                              for scenario in scenarios]),
                mass_fractions = tuple(
                    [np.array([scenario.size.modes[m].mass_fractions[s] \
                               for scenario in scenarios]) \
                               for s in range(num_species)]
                ),
            ))
        size = AerosolModalSizePopulation(
            modes = tuple(modes),
        )
    else:
        raise TypeError("Invalid particle size information in scenarios!")
    gas_concs = []
    for g in range(len(scenarios[0].gas_concs)):
        gas_concs.append(np.array([scenario.gas_concs[g] for scenario in scenarios]))
    return Ensemble(
        aerosols = scenarios[0].aerosols,
        gases = scenarios[0].gases,
        size = size,
        gas_concs = tuple(gas_concs),
        flux = flux,
        relative_humidity = relative_humidity,
        temperature = temperature,
        pressure = pressure, #scenarios[0].pressure,
        height = scenarios[0].height,
        gas_emissions = scenarios[0].gas_emissions,
        gas_background = scenarios[0].gas_background,
    )

#-------------------------------------------------
# Ensembles constructed by sampling distributions
#-------------------------------------------------

def sample(specification: EnsembleSpecification, n: int) -> Ensemble:
    """sample(spec, n) -> n-member ensemble sampled from a specification"""
    size = None
    if isinstance(specification.size, AerosolModalSizeDistribution):
        size = AerosolModalSizePopulation(
            modes=tuple([
                AerosolModePopulation(
                    name = mode.name,
                    species = mode.species,
                    number = mode.number.rvs(n),
                    geom_mean_diam = mode.geom_mean_diam.rvs(n),
                    # log10_geom_std_dev = np.array([mode.log10_geom_std_dev.rvs(n) for i in range(n)]),
                    log10_geom_std_dev = mode.log10_geom_std_dev.rvs(n),
                    mass_fractions = tuple([f.rvs(n) for f in mode.mass_fractions]),
                ) for mode in specification.size.modes]),
        )
        # normalize mass fractions
        for mode in size.modes:
            factor = sum([mass_fraction for mass_fraction in mode.mass_fractions])
            mode.mass_fractions = tuple([mf/factor for mf in mode.mass_fractions])
    return Ensemble(
        aerosols = specification.aerosols,
        gases = specification.gases,
        specification = specification,
        size = size,
        gas_concs = tuple([gas_conc.rvs(n) for gas_conc in specification.gas_concs]),
        flux = specification.flux.rvs(n),
        relative_humidity = specification.relative_humidity.rvs(n),
        temperature = specification.temperature.rvs(n),
        pressure = specification.pressure.rvs(n),
        height = specification.height,
        gas_emissions = specification.gas_emissions,
        gas_background = specification.gas_background,
    )

def lhs(specification: EnsembleSpecification,
        n: int,
        criterion = None,
        iterations = None) -> Ensemble:
    """lhs(specification, n, [criterion, iterations]) -> n-member ensemble
generated from latin hypercube sampling applied to the given specification. The
optional arguments are passed along to pyDOE's lhs function, which creates the
distribution from which ensemble members are sampled."""
    num_gases = len(specification.gases)
    n_factors = num_gases + 4 # size-independent factors: num_gases + flux + relative_humidity + temperature (!!!FIXME) + presssure
    lhd = None # latin hypercube distribution (created depending on particle
               # size representation)
    size = None
    if isinstance(specification.size, AerosolModalSizeDistribution):
        # count up mode-related factors
        for mode in specification.size.modes:
            n_factors += 3 + len(mode.mass_fractions) # !!!FIXME Changed offset to 3 from 2 throughout to accommodate sigmag dist
        # lhd is a 2D array with indices (sample index, factor index)
        lhd = pyDOE.lhs(n_factors, n, criterion, iterations)
        num_species = [len(mode.mass_fractions) for mode in specification.size.modes]
        size = AerosolModalSizePopulation(
            modes=tuple([
                AerosolModePopulation(
                    name = mode.name,
                    species = mode.species,
                    number=mode.number.ppf(lhd[:,(3+num_species[m])*m]),
                    geom_mean_diam=mode.geom_mean_diam.ppf(lhd[:,(3+num_species[m])*m+1]),
                    log10_geom_std_dev=np.array(mode.log10_geom_std_dev.ppf(lhd[:,(3+num_species[m])*m+2])), # !!! FIXME: Changed np.array --> np.log10
                    mass_fractions=tuple(
                        [mass_fraction.ppf(lhd[:,(3+num_species[m])*m+f])
                         for f, mass_fraction in enumerate(mode.mass_fractions)]),
                ) for m, mode in enumerate(specification.size.modes)]),
        )

        # FIXME: laura's janky fix to set some mass fractions to zero
        for mode in size.modes:
            for qq in range(len(mode.mass_fractions)):
                print(mode.mass_fractions[qq])
                for ii in range(len(mode.mass_fractions[qq])):
                    if np.isnan(mode.mass_fractions[qq][ii]):
                        mode.mass_fractions[qq][ii] = 0.
        
        # FIXME: laura's janky fix to normalize mass fractions
        for mode in size.modes:
            factor = sum([mass_fraction for mass_fraction in mode.mass_fractions])
            mode.mass_fractions = tuple([mf/factor for mf in mode.mass_fractions])
            
    gas_concs = []
    for g, gas_conc in enumerate(specification.gas_concs):
        gas_concs.append(gas_conc.ppf(lhd[:,-4-(num_gases-g)]))
    return Ensemble(
        aerosols = specification.aerosols,
        gases = specification.gases,
        specification = specification,
        size = size,
        gas_concs = tuple(gas_concs),
        flux = specification.flux.ppf(lhd[:,-4]), # !!!FIXME Following indices shifted down by 1 to accommodate pressure
        relative_humidity = specification.relative_humidity.ppf(lhd[:,-3]),
        temperature = specification.temperature.ppf(lhd[:,-2]),
        pressure = specification.pressure.ppf(lhd[:,-1]),
        height = specification.height,
        gas_emissions = specification.gas_emissions,
        gas_background = specification.gas_background,
    )

#---------------------------
# Swept-parameter ensembles
#---------------------------

@dataclass(frozen=True)
class LinearParameterSweep:
    """LinearParameterSweep(a, b, n) - an n-step, uniformly-spaced sweep in the
parameter range [a, b]"""
    a: float
    b: float
    n: int

    def __len__(self) -> int: # `len(sweep)` returns number of values swept
        return self.n

    def __iter__(self) -> float: # `for p in sweep` allows iteration over assumed values
        assert(self.b - self.a > 0)
        step = (self.b - self.a)/self.n
        for i in range(self.n):
            yield self.a + i*step

@dataclass(frozen=True)
class LogarithmicParameterSweep:
    """LogarithmicParameterSweep(a, b, n) - an n-step, log10-spaced sweep in the
parameter range [a, b]"""
    a: float
    b: float
    n: int

    def __len__(self) -> int: # returns number of values swept
        return self.n

    def __iter__(self) -> float: # allows iteration over assumed values
        assert(self.b - self.a > 0)
        log_a = log10(self.a)
        log_b = log10(self.b)
        step = (log_b - log_a)/self.n
        for i in range(self.n):
            yield pow(10, log_a + i*step)

@dataclass(frozen=True)
class AerosolModeParameterSweeps:
    """AerosolModeParameterSweep: a set of parameter sweep ranges for a single,
internally-mixed aerosol mode"""
    species: tuple[AerosolSpecies, ...]
    number: Optional[LinearParameterSweep | LogarithmicParameterSweep] = None
    geom_mean_diam: Optional[LinearParameterSweep | LogarithmicParameterSweep] = None
    mass_fractions: Optional[tuple[LinearParameterSweep | LogarithmicParameterSweep, ...]] = None

@dataclass(frozen=True)
class AerosolModalSizeParameterSweeps:
    """AerosolModalSizeParameterSweeps: a set of parameter sweep ranges for
modal aerosol particle size"""
    modes: Optional[tuple[Optional[AerosolModeParameterSweeps], ...]] = None

@dataclass(frozen=True)
class AerosolParameterSweeps:
    """AerosolParameterSweeps: a set of aerosol parameter "sweep" ranges used by
the sweep function to construct an ensemble from a reference state.

Each sweep range is ultimately specified by one of the following:
* LinearParameterSweep(a, b, step) - a uniformly-spaced sweep in the parameter
  range [a, b]
* LogarithmicParameterSweep(a, b, step) - a logarithmically-spaced sweep in
  the parameter range [a, b]

If no sweep is specified for a given parameter, that parameter assumes a
value specified by the reference_state passed to the sweep function.
"""
    size: Optional[AerosolModalSizeParameterSweeps] = None
    gas_concs: Optional[tuple[LinearParameterSweep | LogarithmicParameterSweep, ...]] = None
    flux: Optional[LinearParameterSweep | LogarithmicParameterSweep] = None
    relative_humidity: Optional[LinearParameterSweep | LogarithmicParameterSweep] = None
    temperature: Optional[LinearParameterSweep | LogarithmicParameterSweep] = None

def modal_size_factors(ref_size: AerosolModalSizeState,
                       sweeps: AerosolModalSizeParameterSweeps) -> list:
    """Returns a list of numpy arrays containing all values assumed by
modal parameter sweeps for every mode in a modal particle size description.
These arrays represent "factors" in the cartesian product representing all
possible combinations of parameters."""
    factors = []
    for m, mode in enumerate(ref_size.modes):
        if sweeps and sweeps.modes and sweeps.modes[m] and sweeps.modes[m].number:
            numbers = [n for n in sweeps.modes[m].number]
        else:
            numbers = [ref_size.modes[m].number]
        factors.append(numbers)
        if sweeps and sweeps.modes and sweeps.modes[m] and sweeps.modes[m].geom_mean_diam:
            diameters = [d for d in sweeps.modes[m].geom_mean_diam]
        else:
            diameters = [ref_size.modes[m].geom_mean_diam]
        factors.append(diameters)
        if sweeps and sweeps.modes and sweeps.modes[m] and sweeps.modes[m].mass_fractions:
            for mf in sweeps.modes[m].mass_fractions:
                mass_fracs = [f for f in mf]
                factors.append(mass_fracs)
        else:
            for mf in ref_size.modes[m].mass_fractions:
                factors.append([mf])
    return factors

def sweep(reference_state: Scenario, sweeps: AerosolParameterSweeps) -> Ensemble:
    """sweep(reference_state, sweeps) -> ensemble generated by initializing a
"reference state" and performing sweeps for the parameters specified in the
given ParameterSweep. The size of the ensemble is determined by the specified
parameter sweeps"""
    # We form a cartesian product of all parameter sweeps ("factors") to obtain
    # the set of all possible parameter combinations.
    if isinstance(reference_state.size, AerosolModalSizeState): # modal
        if sweeps and sweeps.size and not isinstance(sweeps.size, AerosolModalSizeParameterSweeps):
            raise TypeError('sweep: Reference state and sweeps have mismatched particle size representations!')
        else:
            factors = modal_size_factors(reference_state.size, sweeps.size)
    else:
        raise TypeError(f'sweep: Unsupported particle size representation: {reference_state.size.__class__}')
    if sweeps.gas_concs:
        for gas_conc in sweeps.gas_concs:
            factors.append([c for c in gas_conc])
    else:
        for c in range(len(reference_state.gas_concs)):
            factors.append([reference_state.gas_concs[c]])
    if sweeps.flux:
        factors.append([F for F in sweeps.flux])
    else:
        factors.append([reference_state.flux])
    if sweeps.relative_humidity:
        factors.append([rh for rh in sweeps.relative_humidity])
    else:
        factors.append([reference_state.relative_humidity])
    if sweeps.temperature:
        factors.append([T for T in sweeps.temperature])
    else:
        factors.append([reference_state.temperature])

    # form all parameter combinations to populate an ensemble
    all_params = list(itertools.product(*factors))
    members = []
    for params in all_params:
        index = 0
        size = None
        if isinstance(reference_state.size, AerosolModalSizeState):
            mode_states = []
            for mode in reference_state.size.modes:
                num_species = len(mode.species)
                mass_fractions = [params[index+2+s] for s in range(num_species)]
                mode_states.append(
                    AerosolModeState(
                        name = mode.name,
                        species = mode.species,
                        number = params[index],
                        geom_mean_diam = params[index+1],
                        log10_geom_std_dev = mode.log10_geom_std_dev,
                        mass_fractions = tuple(mass_fractions),
                    ),
                )
                index += 2 + num_species
            size = AerosolModalSizeState(
                modes = mode_states,
            )
        else:
            raise TypeError(f'Unsupported particle size state: {reference_state.size.__class__}')
        gas_concs = []
        for g in range(len(reference_state.gas_concs)):
            gas_concs.append(params[index + g])
        index += len(gas_concs)
        member = Scenario(
            aerosols = reference_state.aerosols,
            gases = reference_state.gases,
            size = size,
            gas_concs = tuple(gas_concs),
            flux = params[index],
            relative_humidity = params[index + 1],
            temperature = params[index + 2],
            pressure = reference_state.pressure,
            height = reference_state.height,
            gas_emissions = reference_state.gas_emissions,
            gas_background = reference_state.gas_background,
        )
        index += 3
        members.append(member)

    return ensemble_from_scenarios(members)
