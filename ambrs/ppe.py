"""ambrs.ppe - types and functions supporting the creation of perturbed-parameter ensembles (PPE)."""

from dataclasses import dataclass
from pyDOE import lhs # latin hypercube sampling

@dataclass
class EnsembleSpecification:
    """EnsembleSpecification: a specification for a specific PPE"""
    name: str

