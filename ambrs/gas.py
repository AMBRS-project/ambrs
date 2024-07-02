"""ambrs.gas - Gas species related data types"""

from dataclasses import dataclass

@dataclass(frozen=True)
class GasSpecies:
    """GasSpecies: the definition of a gas species in terms of species-
specific parameters (no state information)"""
    name: str          # name of the species
    molar_mass: float  # etc

