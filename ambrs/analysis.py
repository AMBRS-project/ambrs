"""ambrs.analysis - functions and tools for analyzing output from box models
"""

from dataclasses import dataclass
import netCDF4
import numpy as np
import scipy.stats
from typing import Any

@dataclass
class Output:
    """ambrs.analysis.Output: a set of output gathered from a box model that can
be used in post-processing and analysis within AMBRS."""
    input: Any        # input object corresponding to this output
    model: str        # name of the box model that produced the output
    bins: np.array    # array representing logarithmically spaced particle size bins
    dNdlnD: np.array  # particle populations array of particles binned by (logarithmic) size
    ccn: float = None # a measure of cloud concentration number (NOTE: not yet required)

def kl_divergence(dNdlnD1: np.array,
                  dNdlnD2: np.array) -> float:
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

