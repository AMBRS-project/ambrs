"""ambrs.analysis - functions and tools for analyzing output from box models
"""

from dataclasses import dataclass
import netCDF4
import numpy as np
import scipy.stats
from typing import Any
from PyParticle import ParticlePopulation, builder

@dataclass
class Output:
    """ambrs.analysis.Output: a set of output gathered from a box model that can
be used in post-processing and analysis within AMBRS."""
    input: Any        # input object corresponding to this output
    model: str        # name of the box model that produced the output
    population_settings: dict # dictionary containing specifications (e.g., output_filename, timestep, N_bins, etc.)
    population: ParticlePopulation # particle population representing output
    
    # bins: np.array    # array representing logarithmically spaced particle size bins
    # dNdlnD: np.array  # particle populations array of particles binned by (logarithmic) size
    # ccn: float = None # a measure of cloud concentration number (NOTE: not yet required)
    
    def _populate(self,population_settings,N_particles=None):
        # if N_particles=None, use whatever the model tracks
        #   - probably what you want for a sectional or particle-based model
        #   - probably not what you want for modal model
        if self.model.lower() == 'partmc':
            population = builder.partmc.build(population_settings)
        elif self.model.lower() == 'mam4':
            population = builder.mam4.build(population_settings)
        
        self.population = population
        self.population_settings = population_settings
        
    def compute_dNdlnD(self, 
               method='hist', # how to destimate pdf: hist, kde (later), ...
               diam_bins=np.logspace(-9,-4,31), wetsize=True): # if false, look at dNdlnD wrt to dry diameter
        # fixme: may want to extend to compute mean + std dev for PartMC ensembles
        Ds = []    
        for part_id in self.population.ids:
            particle = self.population.get_particle(part_id)
        # for part_id in population.ids:
        #     particle = population.get_particle(part_id)
            if wetsize:
                Ds.append(particle.get_Dwet())
            else:
                Ds.append(particle.get_Ddry())
        if method == 'hist':
            dNdlnD,_ = np.histogram(
                np.log(Ds),bins=diam_bins,
                weights=self.population.num_concs)
        
        return dNdlnD
                

def kl_divergence(dNdlnD1: np.array,
                  dNdlnD2: np.array, 
                  backward=False) -> float:
    """kl_divergence(dNdlnD1, dNdlnD2, backward = False) -> KL-divergence
representing the difference in two particle size distributions represented by
the particle size histograms dNdlnD1 and dNdlnD2. The KL-divergence is computed
as the Shannon entropy of the probability distributions corresponding to these
size distributions.

Optional parameters:
    * backwards: if True, the arguments to the Shannon entropy are reversed in
      the calculation of the KL-divergence.
"""
    P1 = dNdlnD1/sum(dNdlnD1)
    P2 = dNdlnD2/sum(dNdlnD2)
    if backward:
        return scipy.stats.entropy(P2, P1)
    else:
        return scipy.stats.entropy(P1, P2)

