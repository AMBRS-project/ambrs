"""ambrs.gas - Gas species related data types"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

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

@dataclass 
class GasMixture:
    species: Tuple[GasSpecies, ...]
    mole_ratio: np.ndarray 
    
    # fixme: maybe remove the following
    def _add_gas(self,GasSpec,mole_ratio):
        if GasSpec.name not in [spec.name for spec in self.species]:
            self.species.append(GasSpec)
            idx, = -1
            self.mole_ratio = np.append(self.mole_ratio, 0.)
        else:
            idx, = np.where([GasSpec.name==spec.name for spec in self.species])
        self.mole_ratio[idx] = mole_ratio
        
    def _add_H2O_from_RH(self,RH):
        mole_ratio_h2o = RH # fixme: need T. 
        H2O = GasSpecies(name='H2O',molar_mass=18.)
        self._add_gas(H2O,mole_ratio_h2o)
    
def build_gas_mixture(gas_cfg: dict) -> GasMixture:
    # fixme: need automated way to populate defaults.
    
    gas_spec_names = ['SO2','H2SO4','SOAG']
    # fixme: for now, just have SO2 and H2SO4
    so2 = GasSpecies(
        name='SO2',
        molar_mass = 64.07,
    )
    h2so4 = GasSpecies(
        name='H2SO4',
        molar_mass = 98.079,
    )
    
    # # Duncan's assumptions:
    # soag = GasSpecies(
    #     name='SOAG',
    #     molar_mass = 12.011,
    # )

    # Shrivastava et al., 2015 assumes 250 g/mol for https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014JD022563
    soag = GasSpecies(
        name='SOAG',
        molar_mass = 250.0,
    )
    
    gases = [so2,h2so4,soag]
    # todo: move to constants module?
    molar_mass_dry_air = 28.97
    
    # fixme: hard coded with MAM4 gases for now
    species = [so2,h2so4]
    if gas_cfg['units'] == 'ppb':
        # mole_ratio = np.array([gas_cfg.get('SO2',0.),gas_cfg.get('H2SO4',0.)])/1e9
        mole_ratio = np.array([gas_cfg.get(specname, 0.) for specname in gas_spec_names])/1e9
        # gas_cfg.get('SO2',0.),gas_cfg.get('H2SO4',0.)])/1e9
    elif gas_cfg['units'] == 'ratio' or gas_cfg['units'] == 'mole_ratio' or gas_cfg['units'] == 'mol_ratio':
        # mole_ratio = np.array([gas_cfg.get('SO2',0.),gas_cfg.get('H2SO4',0.)])
        mole_ratio = np.array([gas_cfg.get(specname, 0.) for specname in gas_spec_names])

    elif gas_cfg['units'] == 'kg_per_kg':
        # mole_ratio = np.array([
        #     gas_cfg.get('SO2',0.)*molar_mass_dry_air/so2.molar_mass,
        #     gas_cfg.get('H2SO4',0.)*molar_mass_dry_air/h2so4.molar_mass])
        mole_ratio = np.array([gas_cfg.get(specname, 0.)*molar_mass_dry_air/
                               gases[GasSpecies.find(gases,specname)].molar_mass
                               for specname in gas_spec_names])                 
            # GasSpecies.find(species,specname).molar_mass for specname in gas_spec_names])
    else:
        raise ValueError(f"Unsupported gas units: {gas_cfg['units']}, supported units are 'ppb', 'mole_ratio', and 'kg_per_kg'")
    
    return GasMixture(species=species,mole_ratio=mole_ratio)
