"""ambrs.gas - Gas species related data types"""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class GasSpecies:
    """GasSpecies: the definition of a gas species in terms of species-
specific parameters (no state information)"""
    name: str          # name of the species
    molar_mass: float  # [g/mol]
    aliases: Optional[tuple[str, ...]] = None # tuple of alternative species names

    @staticmethod
    def find(species_list: list, species_name: str) -> int:
        """GasSpecies.find(species_list, species_name) -> index of the gas species with the
given name if found within the given list of GasSpecies, or -1 if not found.
This function first tries to match the name of the species, and then tries to
match any given aliases"""
        for s, species in enumerate(species_list):
            if species.name == species_name or \
               (species.aliases and species_name in species.aliases):
                return s
        return -1
