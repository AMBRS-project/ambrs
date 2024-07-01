"""ambrs.ppe - types and functions supporting the creation of perturbed-parameter
ensembles (PPE).

We use scipy.stats distributions (specifically the rv_continuous ones) for
sampling. Because scipy provides an extensible framework for sampling, any
type of sampling process (including sampling from ESMs like E3SM) is possible.
"""

import numpy as np
import scipy.stats

from dataclasses import dataclass
from .aerosol import AerosolModalSizeDistribution, AerosolModalSizePopulation

@dataclass
class EnsembleSpecification:
    """EnsembleSpecification: a set of distributions from which members of a
PPE are sampled"""
    name: str
    modalSize: AerosolModalSizeDistribution
    flux: scipy.stats.rv_continuous
    relativeHumidity: scipy.stats.rv_continuous
    temperature: scipy.stats.rv_continuous

@dataclass
class Ensemble:
    """Ensemble: an ensemble defined by values sampled from the distributions of
a specific EnsembleSpecification"""
    specification: EnsembleSpecification  # used to construct Ensemble
    modalSize: AerosolModalSizePopulation
    flux: np.array
    relativeHumidity: np.array
    temperature: np.array

# TODO: demonstrate how to incorporate experiment design like latin hypercube
def sample(spec: EnsembleSpecification,
           n:    int) -> Ensemble:
    """sample(spec, n) -> n-member ensemble sampled from spec"""
    modalSize = None
    if spec.modalSize:
        modalSize = AerosolModalSizePopulation(
            modes=tuple([
                AerosolModePopulation(
                    number=spec.modalSize.number.rvs(n),
                    geom_mean_diam=spec.modalSize.geom_mean_diam.rvs(n),
                    mass_fractions=tuple([f.rvs(n) for f in spec.modalSize.mass_fractions]),
                ) for m in distribution.modes]),
        )
    return Ensemble(
        specification = spec,
        modalSize = modalSize,
        flux = spec.flux.rvs(n),
        relativeHumidity = spec.relativeHumidity.rvs(n),
        temperature = spec.temperature.rvs(n),
    )
