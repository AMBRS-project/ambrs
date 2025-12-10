"""ambrs.scenario - types and functions supporting the creation of single
simulation scenarios.
"""

from dataclasses import dataclass
from .aerosol import AerosolModalSizeState, AerosolSpecies
from .gas import GasSpecies
from typing import Dict, Optional

@dataclass
class Scenario:
    """Scenario: an abstract description of an aerosol configuration, expressed
in terms of state information"""
    aerosols: tuple[AerosolSpecies, ...] # aerosol species
    gases: tuple[GasSpecies, ...]        # gas species
    size: AerosolModalSizeState          # could also be sectional, stochastic, etc
    gas_concs: tuple[float, ...]         # gas number concentrations (ordered like gases)
    flux: float
    relative_humidity: float
    temperature: float
    pressure: float
    height: float
    gas_emissions: Optional[list[tuple[float, dict], ...]] = None
    gas_background: Optional[list[tuple[float, dict], ...]] = None
