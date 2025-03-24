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

    def __post_init__(self): # checks initialized fields
        # check the name of the species
        valid_gas_species = [
            'H2SO4',
            'HNO3',
            'HCl',
            'NH3',
            'NO',
            'NO2',
            'NO3',
            'N2O5',
            'HONO',
            'HNO4',
            'O3',
            'O1D',
            'O3P',
            'OH',
            'HO2',
            'H2O2',
            'CO',
            'SO2',
            'CH4',
            'C2H6',
            'CH3O2',
            'ETHP',
            'HCHO',
            'CH3OH',
            'ANOL',
            'CH3OOH',
            'ETHOOH',
            'ALD2',
            'HCOOH',
            'RCOOH',
            'C2O3',
            'PAN',
            'ARO1',
            'ARO2',
            'ALK1',
            'OLE1',
            'API1',
            'API2',
            'LIM1',
            'LIM2',
            'PAR',
            'AONE',
            'MGLY',
            'ETH',
            'OLET',
            'OLEI',
            'TOL',
            'XYL',
            'CRES',
            'TO2',
            'CRO',
            'OPEN',
            'ONIT',
            'ROOH',
            'RO2',
            'ANO2',
            'NAP',
            'XO2',
            'XPAR',
            'ISOP',
            'ISOPRD',
            'ISOPP',
            'ISOPN',
            'ISOPO2',
            'API',
            'LIM',
            'DMS',
            'MSA',
            'DMSO',
            'DMSO2',
            'CH3SO2H',
            'CH3SCH2OO',
            'CH3SO2',
            'CH3SO3',
            'CH3SO2OO',
            'CH3SO2CH2OO',
            'SULFHOX',
            'SOAG',
        ]
        # if self.name.upper() not in valid_gas_species:
        #     raise NameError(f'Invalid gas species name: {self.name}\nValid names are {valid_gas_species}')
        if self.molar_mass <= 0.0:
            raise ValueError(f'Non-positive molar mass: {self.molar_mass}')

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
